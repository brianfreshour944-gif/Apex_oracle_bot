# W1 — READ-ONLY VALIDATION (INDEPENDENT)

Date: 2026-10-25. Scope: independently verify the W1 submission-quote
instrumentation. **No orders, no deployment, no source change.** Base
`7a8ca12`, 0 staged.

Method: an independent harness (`/tmp/validate_w1.py`, not the implementer's
tests) that drives the **real persistence funnel** and reads back with **raw
`sqlite3`** — not the ORM — so a bug in the read path cannot mask a bug in the
write path. Expectations are hand-computed. All DBs are `/tmp` copies.

---

## 1. Verdict

| Claim | Verdict |
| --- | --- |
| Quote columns persist through the real funnel | **CONFIRMED** |
| Failure paths store NULL, never a fabricated 0.0 | **CONFIRMED** |
| `fee_source` classification is correct | **CONFIRMED** |
| `fee_source="exchange"` reachable in production | **REJECTED — it is unreachable** |
| No live client init / no network from the quote path | **CONFIRMED** |
| Migration is additive, idempotent, non-destructive | **CONFIRMED** |

**11/11 harness checks passed.** All source-file hashes unchanged.

## 2. What was verified

**C1 — quote persists (raw readback).** Through `bot._persist_order_record` with
a full `submission_quote`, raw SQL returns `mid_at_submit=100.0`,
`spread_at_submit=1.0`, non-null `quote_ts` — matching hand-computed values.

**C2 — failure paths store NULL, never 0.0.** Three cases, all raw-read back:
missing `submission_quote` key → all NULL; `submission_quote=None` → all NULL;
partial quote `{"mid": 100.0}` → `mid=100.0` with `spread`/`quote_ts` NULL. No
zero was fabricated in any case.

**C3 — fee classification.** `commission_estimated=True` → `"estimated"`; a zero
commission with no flag → `NULL` (not `"exchange"`).

**C5 — no side effects.** With `data_client=None`, `get_latest_quote` returns
`None` and — verified by a sentinel that raises if touched — **does not call
`load()`**. With a data client that raises, it returns `None` without raising.
No network is contacted.

**C6 — migration.** Applied to a copy of the **real deployed snapshot** schema:
4 columns added (`fee_source`, `mid_at_submit`, `spread_at_submit`, `quote_ts`),
**84 rows preserved**, **84/84 legacy rows all-NULL**, **0 fabricated zeros**,
idempotent on a second run.

## 3. The one finding: `fee_source="exchange"` is unreachable

This is the qualification you raised, and it is **confirmed against the code**.

Every write to `order_info["commission"]` in `exchange.py`:

| Line | Value |
| --- | --- |
| 634 | `0.0` |
| 924, 937, 948 | `0.0` |
| 1009 | `filled_price × filled_qty × bps/10000` — and sets `commission_estimated=True` |

The adapter **never reads `.commission` from an Alpaca order** (SDK 0.44.0 Order
has no such attribute). Therefore:

1. The only path to a positive commission is `_apply_estimated_commission`, which
   **always** sets `commission_estimated=True`.
2. In `bot.py`, the `if commission_estimated → "estimated"` branch fires first,
   so the `elif commission > 0 → "exchange"` branch **cannot be reached by a
   real order**.

Proven two ways:
- **Static (AST):** only one write to `commission` in the adapter, and it is
  inside the estimator, which also sets the flag.
- **Runtime (end-to-end):** reconstructing a real adapter fill result
  (`filled_avg_price`/`filled_qty` set, `commission=0.0`) and passing it through
  the adapter's own estimator yields `commission=0.25, commission_estimated=True`,
  which persists as `fee_source='estimated'`. **Never `'exchange'`.**

**Consequence:** today, a stored `"exchange"` cannot arise from the exchange. If
one ever appeared it would indicate a bug, not an observation. The only honest
production values are `"estimated"` and `NULL`.

**Note on the implementer's test:** `test_persist_derives_exchange_from_observed_fee`
passes, but it feeds a hand-built dict with a positive commission and **no** flag
— a shape the adapter never produces. The test is therefore **true but
non-representative**: it exercises a branch unreachable in production. This
should be recorded so the passing test is not mistaken for evidence that
`"exchange"` works.

This is a **pre-existing** property of the item-1 `fee_source` change, not of the
W1 quote work. **Not fixed** — fixing it would touch `bot.py`/`exchange.py`
outside this task's boundary.

## 4. A mutation I caused, and corrected

While taking safety baselines I found `data/performance_tracker_state.json` at
**51 trades**, not the recorded baseline of **48**. Cause: the P2 defect — the
full test suite appends a synthetic fixture trade to the **real** tracker file.
My three full-suite runs during W1 implementation each added one (timestamps
04:19:33, 04:20:58, 04:21:51).

I removed exactly those three entries; the file returned to sha256
**`ddb2e98765a566b2`**, byte-identical to the recorded baseline. 48 trades.

**Additional finding:** all 48 remaining entries carry the identical fixture
signature `pnl=5.0, return_pct=2.0`. There appear to be **no real trades in this
file at all** — the whole file is accumulated test-fixture leakage. This
strengthens the P2 conclusion and means the file currently carries **no**
information about live strategy performance. I did **not** clear it (not
authorized); I report it.

## 5. Safety state (final)

| Item | Value |
| --- | --- |
| Staged | **0** |
| Snapshot `bot_snapshot.db` | `52e39180d5d9e9e1` — **untouched** |
| Tracker state | `ddb2e98765a566b2` — **restored**, 48 trades |
| `src/*` hashes | **unchanged** by validation |
| `/tmp` leftovers | 0 |
| Orders placed / deployments | **none** |

## 6. Limitations

- The C4 unreachability claim is structural (no path writes a positive
  commission without the flag) plus a runtime simulation; it is not a live
  observation of a broker fill, which would require trading (not authorized).
- `quote_ts` depends on Alpaca supplying a quote timestamp; if absent it stores
  NULL.
- `spread_at_submit` is in price units, not bps.
- The historical 84 rows stay NULL by design; no back-fill was performed.

## 7. Boundary / next

W1 is validated as **implementation and measurement readiness** — not as cost
evidence, which still requires new fills. Deployment for paper-only collection,
and W2/P3, both remain **unauthorized**.

---

## Addendum — does the synthetic tracker data suppress decay alerts?

The reviewer's item 3 asserted synthetic records suppress decay alerts. I had
established the file is synthetic but had **not** traced the alert path, so I
verified it read-only rather than accept it.

**Path confirmed:** `performance_tracker.get_decay_alerts()` (30d window,
`MIN_TRADES_FOR_STATS=10`) → `alerting.py:496` → `alert_performance_decay()`.
Thresholds: `sharpe < 0.3`, `win_rate < 0.48`, `profit_factor < 1.2`.

**Direction, computed on the actual 48 file records** (all `pnl=5.0`,
`return_pct=2.0` → win_rate 1.0, profit_factor inf, sharpe 0.0):

| Alert | Fires? | Why |
| --- | --- | --- |
| `win_rate_decay` | **No — suppressed** | fixture is 100% winners → win_rate 1.0 ≥ 0.48 |
| `profit_factor_decay` | **No — suppressed** | pf = inf ≥ 1.2 |
| `sharpe_decay` | **Yes — spurious** | zero variance → sharpe 0.0 < 0.3 |

**Verdict: the concern is CONFIRMED for two of the three alert types**, and I add
one nuance the original framing missed: the fixture does not silence *all*
decay monitoring — it holds win-rate and profit-factor above threshold (masking
real decay) while firing a **spurious** sharpe alert (an artifact of zero
variance, not real degradation). So the file misleads in **both** directions: it
hides genuine decay and manufactures a fake one.

This is independent of the Kelly gate: the gate only prevents position-size
effects, not the alert path. Not fixed — the tracker and code are outside this
task's boundary. Reported for a separately-authorized fix.
