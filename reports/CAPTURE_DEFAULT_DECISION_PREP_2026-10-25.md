# PREP — CAPTURE_SUBMISSION_QUOTE DEFAULT DECISION (READ-ONLY)

Date: 2026-10-25. **Read-only.** No edits, staging, commits, DB changes, push,
or deploy. The default remains **`False`**.

Authorized by bare "Authorize prep for capture default decision". This frames the
decision and its evidence. It changes nothing and recommends no flip.

**Supersedes** `reports/DECISION_QUOTE_CAPTURE_DEFAULT_2026-10-25.md`, which is
now stale: it assumed default `True`, predates the flip to `False`, and predates
the executor-isolation fix. Two of its premises are no longer true.

---

## 1. What changed since that memo (why it is stale)

| Memo claim (2026-10-25 earlier) | Now |
| --- | --- |
| Default is `True` | **`False`** (`src/config.py:428`) |
| `get_latest_quote` is time-unbounded | **bounded** — `asyncio.wait_for(_QUOTE_CAPTURE_TIMEOUT_SEC=2.0)` |
| Capture shares the default pool | **isolated** — dedicated 2-worker pool, commit `8945afd` |
| A test pins default `True` | test now pins **`False`** (`test_capture_flag_defaults_off`, line 242) |

So the memo's central tension ("keep `True` but add a timeout") has been
**resolved by construction**: the timeout and the isolation both exist now.

## 2. What the flag does (verified at HEAD `8945afd`)

When on, `create_order` (`src/exchange.py:887-890`) performs one extra data-
endpoint round-trip immediately before submit and attaches the result to the
order row. Verified properties:

- **Gated**: `if getattr(settings, "CAPTURE_SUBMISSION_QUOTE", False)`.
- **Failure-isolated**: `except Exception` → `None`; never raises, never blocks on
  an *error*.
- **Time-bounded**: `wait_for` releases the caller after 2.0 s on a *hang*.
- **Worker-isolated**: runs on the dedicated `quote-capture` pool.
- **Never read by a decision**: see §3.

## 3. CRITICAL FINDING — the capture is currently WRITE-ONLY

The quote is persisted but **nothing reads it back**. Verified three ways:

```
grep -rn "OrderRecord.mid_at_submit|OrderRecord.spread_at_submit|OrderRecord.quote_ts"  → empty
grep -rn "mid_at_submit|spread_at_submit" src/ scripts/  → only writes + migration + model
grep -n  "mid_at_submit|spread_at_submit|quote_ts" scripts/analyze_deployed_trades.py → empty
```

- The only **write** sites are `bot.py:1438-1439` and `db.py:304-305`.
- `get_recent_order_records` (`db.py:775`) returns `order_id, decision_id, symbol,
  side, qty, filled_qty, filled_avg_price, status, type, time_in_force,
  client_order_id, submitted_at, filled_at` — **the three quote columns are NOT
  selected.**
- No `classify_submission_quote` helper exists (grep: no hits).
- The rebuilt analyzer does not read them.

**Consequence:** enabling the flag today would **collect data no tooling
consumes**. The W1 payoff ("separate slippage-vs-mid from half-spread") requires a
read/analysis path that does **not yet exist**. This is the single most important
input to the decision, and it is independent of the risk question.

## 4. Residual risk if enabled (isolation does NOT remove all of it)

| Risk | Status after `8945afd` |
| --- | --- |
| Hung capture delays that order | **bounded** (2.0 s `wait_for`) |
| Hung capture starves the worker pool | **fixed** (dedicated pool) |
| Capture consumes the shared API-key rate-limit quota | **REMAINS** — same `api_key`; `_rate_limit_remaining` is inert (assigned `exchange.py:231`, never read) |
| Shared loop / process / host | **REMAINS** (inherent) |
| `create_order`'s own unbounded socket call | **REMAINS** (out of scope) |

At the observed order rate (24 closed trades over ~22 days) the quota risk is
small. It is not zero, and it is invisible to the bot.

## 5. The decision, stated plainly

Three options. I recommend none of them here — the point of this prep is to make
the choice explicit, not to push one.

**Option 1 — Keep `False` (current).** No quote data. Zero added risk. W1's
instrumentation ships but is dormant. Defensible because nothing reads the data
yet (§3), so enabling now buys nothing measurable.

**Option 2 — Enable `True` now.** Collects quotes for future analysis. Adds the
residual §4 risks. Buys nothing **until a read path exists** (§3), so the data
collected now would sit unused — though it would be available retroactively once
a consumer is written.

**Option 3 — Enable only after a read path lands.** Sequence: build the
read/analysis consumer → verify it → then enable. The most economical path to
actual measurement value, and it avoids accumulating unusable rows.

**Prerequisite common to 2 and 3:** a read path. That is a separate, unimplemented
change (`get_recent_order_records` does not select the columns).

## 6. What enabling would/would not buy today

| Question | Answer |
| --- | --- |
| Does enabling make slippage measurable now? | **No** — no consumer (§3) |
| Does enabling add risk? | **Yes** — the §4 residual (small at this turnover) |
| Is the added risk new in kind? | **No** — same class as existing unbounded order-path calls |
| Does isolation make `True` safe? | **Not fully** — quota risk remains |
| Is `False` harmful? | **No** — it is the current, safe state |

## 7. Recommended next artifact (not this prep)

If the decision is Option 3, the next read-only prep is a **read-path design**:
which consumer should read `mid_at_submit`/`spread_at_submit`/`quote_ts` (the
analyzer, or `get_recent_order_records`), and what it should compute
(slippage-vs-mid, half-spread decomposition, stale-quote exclusion via
`quote_ts`). That is where the W1 value actually gets realised.

## Boundary

This prep authorizes **nothing** and changes no default. `CAPTURE_SUBMISSION_QUOTE`
stays **False**. No push, no deploy.

## Safety state

Branch `w1-instrumentation-accounting`, HEAD `8945afd` (6 ahead), `main` `7a8ca12`
unchanged; 0 staged, 0 unstaged.
