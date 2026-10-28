# EXECUTOR ISOLATION PREP (READ-ONLY) — quote capture must not starve the order path

Date: 2026-10-25. **Read-only.** No edits, staging, commits, DB changes, push,
or deploy. `CAPTURE_SUBMISSION_QUOTE` stays **False**.

Authorized by bare "Authorize prep for executor isolation". This maps the change
and its evidence. It implements nothing.

Prior design context already on disk (read and reconciled, not re-derived):
`reports/DESIGN_CAPTURE_ISOLATION_2026-10-25.md`,
`reports/ADDENDUM_TIMEOUT_POOL_COUPLING_2026-10-25.md`,
`reports/SCOPE_ISOLATION_LIMITS_2026-10-25.md`.

---

## 1. The problem, located in code

Quote capture runs **inline, awaited, before the order is submitted**, on the
**shared asyncio default executor**:

- Capture call site: `src/exchange.py:843` — inside `create_order`, gated by
  `CAPTURE_SUBMISSION_QUOTE`, `await self.get_latest_quote(symbol)`.
- `get_latest_quote` (`src/exchange.py:447`) does
  `await asyncio.to_thread(self.data_client.get_crypto_latest_quote, ...)`
  (`exchange.py:467`).
- `asyncio.to_thread` → `loop.run_in_executor(None, ...)` → the loop's **single
  default pool**.

`exchange.py` has **20** `asyncio.to_thread` sites: **7** on `trading_client`
(incl. `submit_order`) and **2** on `data_client`, all on that same pool.

**Consequence:** a hung capture occupies a worker the order path needs. Because
the capture is *awaited before* `submit_order`, a hang delays that order
directly; repeated hangs saturate the pool and delay *unrelated* orders.

## 2. Evidence re-measured at HEAD (not quoted from the old notes)

| Measurement | Result |
| --- | --- |
| Default pool size on this host | `min(32, cpu+4)` = **8** (cpu=4) |
| Shared pool, 8 hung calls, then submit | submit waited **1.90 s** |
| Dedicated 2-worker quote pool hung, submit on default pool | submit waited **0.00 s** |

So the coupling is real **and** the isolation is structural. (My first attempt
measured 0.00 s because I created coroutines without scheduling them; corrected
by `create_task`. Recording that so the number is not misread.)

## 3. Timeout hardening is not cleanly available (re-verified)

| Check | Result |
| --- | --- |
| `RESTClient.__init__` | no `timeout` param |
| `RESTClient._request(...)` | `(method, path, data, base_url, api_version)` — **no timeout** |
| `CryptoHistoricalDataClient.__init__` | no `timeout` param |
| `data_client._session` | exists, is a `requests.Session` (private) |

A socket timeout would require reaching into the private `_session` — fragile and
version-dependent. **Isolation is the fix; timeout is optional hardening.**

## 4. The proposed change (for a separate authorization)

Scope: `src/exchange.py` only, plus tests. No trading behavior change.

1. **Dedicated pool** — create a small `ThreadPoolExecutor` (1–2 workers,
   `thread_name_prefix="quote-capture"`) owned by `AlpacaExchange`, used **only**
   by `get_latest_quote`. Route the capture through
   `loop.run_in_executor(self._quote_pool, ...)` instead of `asyncio.to_thread`.
2. **Caller bound** — wrap the capture in `asyncio.wait_for(..., timeout=N)` so
   the order proceeds promptly even if the thread lingers. (Bounds latency, not
   the worker — see §6.)
3. **Lifecycle** — shut the pool down in `AlpacaExchange.close()`
   (`exchange.py:274`), which is already called from `bot.shutdown()`
   (`bot.py:610`) via `bot.py:4213`. `__init__` (`exchange.py:207`) creates it.
4. **Optional** — a socket timeout if achievable without private attributes.
   Do **not** rely on it; isolation is the guarantee.

## 5. Test plan (the missing assertion is the point)

Existing `tests/test_submission_quote_capture.py` covers: quote fields, never
raises, gating on the flag, quote-failure-does-not-block-order, persistence,
migration. **It does not measure submit latency under a hung quote** — verified
by grep, no `latency|elapsed|starv|hung|monotonic` in that file.

New tests to add:

| # | Test | Assertion |
| --- | --- | --- |
| T1 | hung quote on dedicated pool leaves submit unaffected | `submit_order` completes promptly while capture threads hang |
| T2 | `wait_for` releases the caller | capture raises timeout; `create_order` still returns the order |
| T3 | pool is shut down on `close()` | no leaked threads after `close()` |
| T4 | pool is *not* the default executor | capture runs on `quote-capture` thread, not a default worker |

T1 is the one the old design note explicitly called out as missing.

## 6. What this does and does not fix (keep honest)

**Fixes (structural):** worker starvation. Capture can no longer occupy a worker
`submit_order` needs.

**Does not fix:**
- **Shared API-key rate-limit quota** — both clients use the same key; a burst of
  captures still consumes order-path quota. `_rate_limit_remaining` is inert
  (assigned, never read).
- **Shared event loop / process / host.**
- **`create_order`'s own unbounded socket call** — unchanged by isolating capture.

So isolation is **necessary but not complete**. Capture stays **False** in
deployment until this lands and is verified.

## 7. Impact assessment

| Question | Answer |
| --- | --- |
| Changes trading behavior? | **No** — measurement path only |
| Touches protected paths? | `src/exchange.py` is **not** in the protected list (`strategies.py`, `risk.py`, `committee.py`, `db.py`); still a core file, so treat with care |
| New dependencies? | **No** (stdlib `concurrent.futures`) |
| Requires DB change? | **No** |
| Test count delta | **+4** (604 → 608 expected) |
| Risk if pool leaks | bounded — shut down in `close()`; 1–2 threads |

## 8. Sequencing

Executor isolation is a prerequisite for enabling capture, **not** for the A/B/C
accounting work (already committed). Order: isolation → verify → then a *separate*
decision on `CAPTURE_SUBMISSION_QUOTE` default.

## Boundary

This prep authorizes **nothing**. Implementation is a separate authorization.
`CAPTURE_SUBMISSION_QUOTE` stays **False**. No push, no deploy.

## Safety state

Branch `w1-instrumentation-accounting`, HEAD `cc92b37` (5 ahead), `main` `7a8ca12`
unchanged; 0 staged, 0 unstaged.
