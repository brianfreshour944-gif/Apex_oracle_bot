# W1 — SUBMISSION-QUOTE INSTRUMENTATION (IMPLEMENTED, NOT DEPLOYED)

Date: 2026-10-25. Authorized scope: **W1 instrumentation only** — capture
submission-time quote data so execution cost becomes measurable. No deployment,
no orders, no restart, no commit. Tested without deploying or placing trades.

Base: HEAD `7a8ca12` (branch `main`), **0 staged**.

---

## 1. Correction to P1 (important)

The P1 report stated that `mid_at_submit` / `spread_at_submit` / `quote_ts`
"already exist in the uncommitted working tree". **That was wrong.** A repo-wide
search shows those names appear **nowhere** before this change — the claim came
from stale prior-session memory. W1 was therefore **greenfield**: nothing to
enable, everything to build. The P1 conclusion is unaffected (the columns were
absent from the deployed snapshot, which is what mattered); only the "already
exists" remark was inaccurate.

## 2. What was added (four files + one test)

| File | Change |
| --- | --- |
| `src/exchange.py` | `get_latest_quote()`; quote snapshot inside `create_order` before submit |
| `src/db.py` | `OrderRecord.mid_at_submit/spread_at_submit/quote_ts`; `save_order_record` kwargs; additive migration |
| `src/bot.py` | `_persist_order_record` carries the quote from the adapter result |
| `src/config.py` | `CAPTURE_SUBMISSION_QUOTE` (default `True`) |
| `tests/test_submission_quote_capture.py` | 14 tests (new) |

### `exchange.py`
- New `get_latest_quote(symbol)` → `{bid, ask, mid, spread, quote_ts}` or `None`.
  Uses `CryptoLatestQuoteRequest` via the existing `data_client`.
  **Never raises** (catches everything, returns `None`).
- **Deliberately does not call `load()`.** It returns `None` if the data client
  is not yet loaded. Reason: calling `load()` from inside an order path would
  re-initialize `trading_client` — a live credential round-trip that could
  replace an in-flight client. (This was caught by the existing test suite; see
  §5.)
- `create_order` snapshots the quote **before** `submit_order`, gated by
  `CAPTURE_SUBMISSION_QUOTE`, wrapped in its own try/except, and attaches it to
  the result as `order_info["submission_quote"]` on both the fresh-submit and
  recovered-existing-order paths.

### `db.py`
- Three nullable columns on `orders`; migration entries `FLOAT/FLOAT/DATETIME`.
- `save_order_record` accepts the three kwargs (default `None`).

### `bot.py`
- `_persist_order_record` (the single ledger-write funnel) derives
  `mid_at_submit`/`spread_at_submit`/`quote_ts` from
  `order_result["submission_quote"]`. Missing/partial quote → **NULL**.

## 3. Design invariants (the point of the change)

1. **Contemporaneous, not post-fill.** The quote is taken immediately before
   submission, so `mid_at_submit` is a real reference for slippage — unlike the
   stale hourly bars P1 had to use.
2. **Never a fabricated zero.** A missing quote persists as `NULL`, never `0.0`.
   This is the exact trap P1 identified with fees, now closed for quotes.
3. **Cannot affect trading.** Capture is gated, failure-isolated, and returns
   `None` on any problem. A quote outage cannot block, delay, or alter an order.
4. **No live calls in tests.** The suite injects a fake data client; no endpoint
   is contacted, no order is submitted.

## 4. Verification (all local; nothing deployed)

| Check | Result |
| --- | --- |
| W1 test suite | **14 passed** |
| Full suite | **592 passed** (578 baseline + 14 new) |
| Ruff | **All checks passed!** |
| Migration on the **real snapshot schema** (a `/tmp` copy) | 4 columns added; **84 rows preserved**; all legacy rows **NULL**; idempotent |
| Real snapshot `52e39180…` | **byte-identical, untouched** |
| Staged | **0** |

The migration test is the decisive one: applied to a copy of the actual
deployed snapshot's schema, `ALTER TABLE` adds the columns, no row is rewritten,
and every historical row keeps `NULL` (never a fabricated zero quote).

## 5. A defect my first draft introduced — and fixed

My first `get_latest_quote` called `await self.load()` when `data_client` was
`None`. Running the full suite showed **12 failures in `tests/test_exchange.py`**:
those tests stub `trading_client` but leave `data_client` unset, so `load()`
replaced the stub with a **real** `TradingClient` and the test attempted a live
paper order (visible in the captured `APIError: insufficient balance`).

I confirmed these failures were **mine**, not pre-existing, by stashing my
changes and re-running: `29 passed` on HEAD. Fix: `get_latest_quote` no longer
calls `load()`. After the fix: **592 passed, 0 failed**.

This is worth recording because it shows the isolation property is real — an
order-path helper that could trigger a live client init is exactly the kind of
side effect instrumentation must not have.

## 6. What W1 does and does not do

**Does:** make spread and a contemporaneous slippage reference *capturable* on
new orders; preserve fee provenance (`fee_source`, item 1).

**Does not:** produce a cost distribution. As you noted, that needs enough **new**
fills. Until then the fields are populated but not yet informative. W1 makes the
next snapshot *measurable*; it does not make it *measured*.

## 7. Limitations

- `quote_ts` depends on Alpaca returning a quote timestamp; if absent, it stores
  `NULL` (not fabricated).
- `spread_at_submit` is in **price units** (`ask - bid`); conversion to bps needs
  `mid` at read time.
- The snapshot's existing 84 rows will remain `NULL` — correctly, since they were
  written before this instrumentation existed. **No back-fill was performed.**
- `exchange.py` is a protected path; this change was made under your explicit W1
  authorization.

## 8. Safety boundary

- **No deployment, no orders, no restart, no commit.** 0 staged.
- No live endpoint contacted by tests (fake data client only).
- Real snapshot `52e39180…` byte-identical; tracker state unchanged.
- Pre-existing accounting patch untouched by this work; W1 changes are additive.
- New artifacts: `tests/test_submission_quote_capture.py`, this report.

## 9. Recommended next step (not authorized)

Validate on **new paper fills** once they exist: confirm the columns populate,
`fee_source` distinguishes estimated from observed, and `mid_at_submit` vs
`filled_avg_price` yields a real slippage number. Then — and only then — W2 (P3).
