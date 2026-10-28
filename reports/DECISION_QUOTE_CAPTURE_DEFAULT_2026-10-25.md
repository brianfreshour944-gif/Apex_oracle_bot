# DECISION MEMO — CAPTURE_SUBMISSION_QUOTE DEFAULT

Date: 2026-10-25. **Read-only analysis. No code changed.** This resolves the
open question flagged for Commit A so the default is an intentional choice.

---

## Question

`CAPTURE_SUBMISSION_QUOTE` currently defaults to **`True`** (`src/config.py:427`).
Should the committed default be `True` or `False`?

## What the flag does

When on, `create_order()` performs one extra network round-trip to Alpaca's
**data** endpoint (`CryptoLatestQuoteRequest`) immediately before submitting the
order (`src/exchange.py:842`). The result is stored on the order row as
`mid_at_submit` / `spread_at_submit` / `quote_ts`. Measurement only — the value
is never read by any order decision (verified: only `_persist_order_record`
consumes it).

## Findings relevant to the decision

**1. The fetch sits on the order critical path, immediately before submit.**
It is awaited before the order is constructed and submitted. It is
failure-isolated (`except Exception` → returns `None`), so an *error* cannot
block an order.

**2. But it is NOT time-bounded, and neither is anything else here.**
- `get_latest_quote` uses bare `asyncio.to_thread(...)` — **not** the
  `circuit_breaker` wrapper every trading call uses (`exchange.py:286/383/539/851`).
- There is **no** `asyncio.wait_for` / `asyncio.timeout` around it.
- The underlying `alpaca.common.rest` client calls
  `self._session.request(method, url, **opts)` with **no `timeout=`** argument, so
  `requests`' default (no timeout) applies.

Consequence: a *hang* (as opposed to an error) on the data endpoint would delay
order submission. `except Exception` does not catch a hang.

**3. However, this is NOT a regression the flag introduces.**
The existing trading calls (`submit_order`, `get_account`, `get_positions`) also
use the circuit breaker *without* a timeout, and the breaker adds no timeout of
its own (`src/circuit_breaker.py:62`). So the codebase already has no
time-bounded network call on the order path. The flag **adds one more such
call**, it does not create a new class of risk.

**4. Net effect of `True`:** one extra API request and its latency per order,
on top of a path that already has unbounded network calls. For a low-turnover
bot (the P3 target profile) this is negligible in throughput terms.

**5. `False` means no quote data at all** — which defeats W1's entire purpose
(measuring slippage-vs-half-spread from a contemporaneous mid). W1's value
depends on this being on.

**6. A test pins the current default.** `tests/test_submission_quote_capture.py:238`
(`test_capture_flag_defaults_on`) asserts `True`. Changing the default requires
updating that test.

**7. Minor inconsistency (harmless).** `exchange.py` uses
`getattr(settings, "CAPTURE_SUBMISSION_QUOTE", False)` while the config field
defaults to `True`. The effective default is `True` (the field is always
defined), so the `getattr` fallback never fires. Worth tidying but not a defect.

## Recommendation

**Keep the default `True`.** Reasons:

- It is the point of W1; `False` collects nothing.
- The marginal risk (one more unbounded call) is not a *new* risk class — the
  order path already has them.
- A low-turnover bot's order rate makes the extra request immaterial.

**But pair it with a bounded timeout**, as a separate, explicitly-scoped change:
wrap the capture in `asyncio.wait_for(..., timeout=N)` so a data-endpoint hang
cannot delay an order. That is a behaviour change and needs its own review — it
should **not** ride along in Commit A unnoticed.

If you prefer the most conservative default for a first deployment, `False` is
defensible: collect nothing until the timeout hardening lands, then flip it on.
Either is reasonable; the point is that it should be chosen, not inherited.

## Scope

This memo changes nothing. It informs the Commit A decision. The
timeout-hardening suggestion is **out of scope** for the four planned commits and
would need separate authorization.
