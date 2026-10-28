# Deployed Trade Data Audit — 2026-10-24

Status: **COMPLETE — analyzed against the real deployed snapshot.**

Artifact: `/workspace/project/bot_snapshot.db`
- sha256 `52e39180d5d9e9e10eb0d689277cae0cb0c3c480843d321a0732886223cfca1b`
- 2,105,344 bytes, `PRAGMA integrity_check` = `ok`, `quick_check` = `ok`
- Opened read-only with `mode=ro&immutable=1`; never modified or locked.

Baseline: `7a8ca12d1b4cac00372053e59f97440ff2d48f13` (branch `main`) — the commit
the deployment runs. All source citations below are from `git show HEAD:`.

Reproduce everything with:

```
python scripts/analyze_deployed_trades.py /workspace/project/bot_snapshot.db \
    --json reports/deployed_trade_audit.json
```

## 1. Recorded results (verbatim from the snapshot)

Tables: `decision_snapshots` 24, `orders` 84, `shadow_trades` 8180, `experiments` 6.

| Metric | Value |
|---|---|
| Closed trades | 24 (0 open) |
| Win rate | 33.3% (8 win / 16 lose) |
| Gross profit | $1.4213 |
| Gross loss | $-3.5266 |
| Net | **$-2.1053** |
| Profit factor | **0.403** |
| Max drawdown | **Not reported — no valid equity curve exists** |

**Note on the removed drawdown figure (correction, 2026-10-25).** An earlier draft
of this report listed a max drawdown of `-282.04%`. That number is **not a valid
drawdown** and has been removed as a risk metric. It is a percentage derived from
a cumulative *dollar* P&L curve (the running sum of `realized_pnl`: peak $1.1565 →
trough −$2.1053). A percentage of a dollar sum is dimensionally meaningless, and
no equity-curve drawdown can be computed from this snapshot because the account's
starting equity is not recorded. Do **not** quote `-282.04%` as an account
drawdown. (Retained here only for traceability, explicitly labelled invalid.)

By symbol:

| Symbol | Trades | Win rate | Net | PF |
|---|---|---|---|---|
| BTC/USD | 10 | 40.0% | $-0.1812 | 0.261 |
| ETH/USD | 6 | 16.7% | $-2.2081 | 0.014 |
| SOL/USD | 8 | 37.5% | $+0.2841 | 1.273 |

By exit reason: `max_hold_time_exceeded` 12 / -$1.6708;
`[MOMENTUM] Momentum: Loss of bullish momentum` 10 / -$0.1812;
`[TREND_FOLLOWING] Trend Following: Macro trend flipped bearish` 2 / -$0.2533.

By regime: `sideways` 11 / -$2.8173; `low_volatility` 10 / -$0.1812;
`neutral` 2 / -$0.2533; `bull` 1 / +$1.1465.

`SUM(realized_pnl)` over the 24 rows = **-2.1053**, matching the reported net.

## 2. Confirmed defect — `return_pct` uses a dust quantity as its denominator

Two rows are absurd on their face:

| decision_id | symbol | realized_pnl | recorded return_pct |
|---|---|---|---|
| `f110556591b7407e81d524da4ddc2caa` | ETH/USD | -$0.037459 | **-804.76%** |
| `35e92c4ad143451bad087674fd9bde9e` | SOL/USD | -$0.037615 | **-70218.31%** |

Root cause, proved by arithmetic against the ledger (not inference):

`return_pct` is computed as `realized_pnl / (entry_price * qty) * 100`
(`src/bot.py:212`). The **numerator is the full round-trip P&L** — it subtracts
the entry fee summed across *all* buys carrying the decision_id
(`src/bot.py:185-197` → `get_entry_fee_estimate`, `src/db.py:273`) — while the
**denominator uses the qty of a single exit leg**. When a position is scaled out
in pieces, the final leg can be dust:

- SOL: dust sell leg `qty=5.02e-07` @ 107.31; `pnl/(entry*dust_qty)*100` = **-70,570%** (recorded -70,218%).
- ETH: dust sell leg `qty=1.877e-06` @ 2508.71; `pnl/(entry*dust_qty)*100` = **-799.5%** (recorded -804.76%).

Using the snapshot's own position qty instead gives -0.1258% and -0.1242%,
which are consistent with a ~$30 notional and a ~-0.12% net loss after fees.

Evidence the leg was scaled out in pieces (SOL decision, five sells):
`0.140999 buy` then sells of `0.050101`, `0.050124`, `0.040381`, `0.00004`,
`0.000000502`.

Impact: `return_pct` feeds the validation gate's return history
(`src/db.py:1055-1059`), `performance_tracker` Sharpe/win-rate/decay alerts,
`strategy_selector` adaptive reward, and the foundation-freeze positive-expectancy
gate. A -70,218% sample entering any of those would dominate a mean, blow up a
variance, or trip a decay alarm. **The dollar P&L is unaffected and remains the
trustworthy figure.** The bug is confined to the percentage field, and only on
rows whose final exit leg is dust (2 of 24 here).

**Fixed 2026-10-25** — see §10. `realized_pnl`/`return_pct` are now rebuilt from
the whole round trip's fills and the denominator is the summed entry notional,
not one closing leg.

## 3. Confirmed defect — MFE/MAE are structurally zero on the live close path

All 24 rows record `max_favorable_pct = 0.0` and `max_adverse_pct = 0.0`.

Cause is ordering, verified in deployed source: `_record_committee_outcome`
reads `_trailing_peaks` / `_trailing_troughs` to compute the excursions
(`src/bot.py:224-238`), but the standard close path **pops both dicts first**:

```
src/bot.py:2855   _state.strategy._trailing_peaks.pop(symbol, None)
src/bot.py:2857   _state.strategy._trailing_troughs.pop(symbol, None)
src/bot.py:2860   await _record_committee_outcome(...)
```

So the `symbol in _trailing_peaks` guard is always false and both values stay 0.0.
This path covers all three exit reasons present in the snapshot
(`max_hold_time_exceeded`, momentum, trend-flip), which is exactly why 24/24 are
zero. Only the trailing-stop path (`src/bot.py:1774`) calls the recorder before
popping, and no trade in this snapshot used it.

**Fixed 2026-10-25** — see §10. The pre-emptive pop is removed; the recorder now
reads the peak/trough and clears the state itself (idempotently) on every exit
path.

Consequence: the excursion fields carry **no information** in this dataset. Any
analysis of "did trades move favorably before reversing" is impossible from these
columns. They are not evidence that trades never moved; they are evidence the
measurement never ran.

## 4. Confirmed defect — `shadow_trades` are notional, not capital

8,180 rows, aggregate `realized_pnl` **-48,521,700.31**, and **every row has
`qty = 1.0`** (distinct qty values = `[1.0]`).

`process_shadow_signal(..., qty: float = 1.0)` (`src/shadow_arena.py:252`) is the
default, and P&L is `(price_delta) * qty` (`src/shadow_arena.py:242-244`,
`300-302`). With qty pinned at 1.0 unit, a $1 ETH move books $1 and a $100 BTC
move books $100 — position sizing is absent, so the aggregate is dimensionally
meaningless as account performance.

By candidate: `Bot_A_Light` 1,375 rows / -$33,576,621;
`Bot_B_Standard` 2,906 / -$18,519,787; `Bot_C_Heavy` 3,899 / +$3,574,708.

This table is a **candidate-ranking signal for the evolution tournament**
(`candidate_name` is the model, not a strategy), and it is fit for that
*relative* purpose. It is not evidence about the deployed account and must be
excluded from any profitability claim. It was excluded from §1.

## 5. Order↔decision linkage

84 filled orders, 42 distinct `decision_id` values:

- 11 orders have a **null/empty** `decision_id`.
- 18 orders carry a `decision_id` **not present** in `decision_snapshots`
  (e.g. `a08597aaa623415a85106be1dd1a3429`).
- 0 closed snapshots lack orders — every one of the 24 links to at least one order.

So the join is *complete in the direction that matters* (every trade has orders)
but has 29 orphaned order rows. Orphans are consistent with rejected/unrecorded
decisions and scale-in legs; they do not by themselves invalidate the 24 trades.

The orphans are not cosmetic, though: they are the reason 13 of the 24 closed
rows cannot be certified (11 reconciled / 13 unreconciled — see §10). The
reconstruction in §10 links an exit only through
`orders.decision_id`; when the exit sells carry a null decision_id (11 orders)
or a decision_id absent from `decision_snapshots` (18 orders), the round trip
cannot be assembled and the row is labelled `unreconciled` rather than reported
with a guessed P&L. Fixing the linkage is therefore a prerequisite for
certifying those 13 trades.

## 6. Commission / fee accounting

`orders.commission`: 84 filled, 42 with commission > 0, total **$1.4886**.

| Symbol | Orders | Nonzero | Total |
|---|---|---|---|
| BTC/USD | 30 | **0** | $0.00 |
| ETH/USD | 27 | 22 | $0.8167 |
| SOL/USD | 27 | 20 | $0.6719 |

Every BTC order records zero commission while ETH/SOL record nonzero. The column
has no provenance flag, so we cannot tell actual exchange charge from modelled
estimate. Two consequences:

1. BTC's net P&L of -$0.1812 is **not demonstrably net of fees** — if BTC fees
   were charged but not recorded, the true figure is worse.
2. Fees already subtracted inside `realized_pnl` (the entry-fee sum) and the
   fees stored on `orders` cannot be reconciled without a provenance marker.

## 7. Does the bot have an edge?

| Question | Answer |
|---|---|
| Recorded performance profitable overall? | No — net -$2.1053, PF 0.403 |
| Evidence of a repeatable SOL edge? | No — 8 trades, PF 1.273 is a lead, not proof |
| Is ETH a concern? | Yes — 6 trades, -$2.2081, PF 0.014 |
| Are the percentage returns usable? | No — 2 of 24 corrupted by §2 (now fixed); MFE/MAE void by §3 (now fixed) |
| Are the dollar P&L figures usable? | Partly — 13 of 24 are `unreconciled` (11 reconciled; §5 linkage); the rest carry the §6 BTC fee caveat |
| Is there enough evidence to resume live trading? | No |

24 closed decisions over 2026-10-02 → 2026-10-24 is a very small sample; the
sign of the result is not established by this data either way.

## 8. Recommended next steps

1. ~~Fix §2 so `return_pct` uses position notional~~ — **done 2026-10-25 (§10)**.
2. ~~Fix §3 by moving the pop after the recorder call~~ — **done 2026-10-25 (§10)**.
3. Add a provenance flag to `orders.commission` and reconcile it against
   `realized_pnl`; explain the BTC zeros. (§6; the accounting fix nets the entry
   fee and the exit fee per the verified convention, but BTC's all-zero
   commission still means its P&L is not demonstrably fee-inclusive.)
4. Fix or formally scope §4 — record `qty` from sizing, and label the table
   clearly as a ranking signal.
5. Fix the order→decision linkage (§5) so the 13 currently-`unreconciled` rows
   can be certified; the exit sells are orphaned or carry a null decision_id.
6. **Decide the learner question explicitly.** The adaptive learner still consumes
   `realized_pnl`/`return_pct` regardless of `outcome_status`, so it can train on
   corrupted legacy rows and on new `unreconciled` ones. Honouring the label is a
   strategy-behavior change and needs its own approval — do not fold it into the
   accounting fix. Also decide whether a `decision_id` may be reused across two
   round trips (§10, `c6a6d0423cc2`).
7. Only after 3–6: run a fresh, untouched paper evaluation with every decision,
   rejection, order, fill, fee, and exit recorded.

## 9. Verification performed

- Analyzer: `scripts/analyze_deployed_trades.py` run against the real file; JSON
  written to `reports/deployed_trade_audit.json`.
- Cross-checks run directly against the snapshot with hand-verified SQL
  (sums, per-symbol counts, join coverage, commission totals, dust-leg
  arithmetic) — all figures above are reproduced by those queries.
- Test suite: `pytest tests/` → 527 passed.
- Lint: `ruff check .` → All checks passed.

## 10. Fix applied 2026-10-25 (accounting correctness only)

Fixes §2 and §3 at the source, without touching strategy, risk thresholds,
deployment, or any protected path. Three files changed, plus tests.

- **One shared reconstruction.** `src/db.py:reconstruct_position_outcome()` rebuilds
  a closed position's `realized_pnl`/`return_pct` from **every** fill linked to the
  `decision_id` (scale-ins and scale-outs included), not from a single closing leg.
  The denominator is the summed entry notional, so a dust final leg can no longer
  produce a -804%/-70,218% value.
- **Fee convention, established from the fills (corrected after review).** The
  schema records no fee *currency*, so the convention was read off the recorded
  quantities, which are the primary evidence:
  - The **entry** fee is withheld in the **base asset**: on every fully-closed
    row `sell_qty == buy_qty × (1 − 0.0025)` **exactly** (and `filled_qty == qty`
    on every leg, so this is not a partial-fill artifact). The fee is therefore
    already inside the reduced `proceeds` and must **not** be subtracted again.
  - The **exit** fee is charged in **quote**: the whole net position is sold and
    the fee is deducted from the cash received, so it **is** subtracted in full.
  - The general form subtracts only the entry commission the reduced quantity
    does not already cover; a residual matching *neither* convention fails closed
    as `unexplained_residual` rather than guessing.
  - An earlier revision of this patch had entry/exit reversed and subtracted the
    entry fee a second time. The reviewer's demand for evidence beyond the
    `(buy_qty − sell_qty) × price ≈ fee` identity is what exposed it: that
    identity held for **both** conventions (entry and exit prices are close), so
    it could not discriminate. The quantity identity above can.
  - With the corrected convention the reconstruction reproduces **9 of the 11**
    reconciled historical values to the cent and **corrects the 2 impossible
    rows** (the third previously-listed row is now `unreconciled` — see below).
- **Fail-closed, machine-readable label.** `decision_snapshots.outcome_status`
  (`"reconciled"` / `"unreconciled"` / `NULL`=unknown) is added via an idempotent
  additive migration (`src/db.py:_ensure_additive_columns`, inspector-guarded
  ALTER that is dialect-agnostic so it works on both SQLite and PostgreSQL, run
  from both `_ensure_tables()` and `init_db()`). A round trip that cannot be
  assembled from the ledger, whose quantities do not reconcile within 50 bps, or
  whose residual fits no fee convention, is labelled `unreconciled` and excluded
  from certified statistics. Legacy rows keep `NULL` ("unknown") and are never
  rewritten as `0.0`.
- **All three close paths unified**: standard exit (`src/bot.py`), protective-stop
  flatten (`src/risk.py:_close_open_snapshot`), and restart reconciliation
  (`src/bot.py:reconcile_open_snapshots`) now call the same helper and pass their
  own observed fill only as a fallback leg.
- **MFE/MAE restored.** The pre-emptive `_trailing_peaks`/`_trailing_troughs` pop
  before the recorder call is removed; the recorder reads the excursion values and
  then clears the state itself via `_clear_symbol_excursion_state()` on every exit
  path (idempotent, so a caller's own clear is harmless).

On the deployed snapshot the reconstruction reports **11 reconciled / 13
unreconciled** rows. The unreconciled are the rows whose exit sells are orphaned
or null-linked (11 `no_linked_exit_fills`), the 3.93x over-closed outlier (1
`quantity_mismatch`), and the reused-decision_id double round trip (1
`multiple_position_lifecycles`) — all linkage/shape problems, now visible instead
of silently guessed. For the 11 reconciled rows the reconstructed aggregate is
**-$2.9290**. Against the stored values, 9 rows are reproduced to the cent and
2 change materially (the third previously-listed row is now `unreconciled`):

| symbol | stored | reconstructed | note |
|---|---|---|---|
| ETH/USD `f110556591…` | -$0.0375 | **+$0.1259** | the -804.76% row; the whole-position net was actually a small gain |
| SOL/USD `35e92c4ad1…` | -$0.0376 | -$0.2163 | the -70,218.31% row |
| ETH/USD `c6a6d0423c…` | -$0.1895 | **unreconciled** | two round trips share one decision_id (§12) |

`c6a6d0423cc2` is worth recording: it holds **two** separate buy→sell round trips
under one `decision_id` (buy @10:33 → 3 sells @11:45-11:47; buy @13:32 → sell
@18:34). Summing them would mix two position lifecycles, so the reconstruction
fails closed as `multiple_position_lifecycles` rather than reporting a merged
number (2026-10-25 review, requirement 2).

One notable consequence: the single profitable `bull`-regime trade in §1
(SOL/USD `1fd274986506`, +$1.1465) is itself **`unreconciled`** — its buy
(0.151743) and sell (0.747381) quantities disagree by 393%. So the one
"encouraging" regime datapoint the report leaned on is not certifiable either.

Scope note: `outcome_status` is an **accounting label only**. It is deliberately
**not** wired into the learner, the replay buffer, `record_strategy_outcome`, or
any gate — gating training on it would change strategy behavior, which is out of
scope for an accounting fix. Consumers that build *statistics* should exclude
non-`reconciled` rows; consumers that build *training data* were left untouched.
**Known risk (accepted, separate decision):** the adaptive learner still consumes
`realized_pnl`/`return_pct` regardless of `outcome_status`, so it may train on the
legacy corrupted rows and on new `unreconciled` ones. Making it honour the label
is a strategy-behavior change and needs its own approval (see §8).

Tests added: `tests/test_position_outcome_reconstruction.py` (35 test functions
as of this correction, 2026-10-25 — partial
exit, dust close, scale-in, fee-sized residual passes, real imbalance fails
closed, no-linked-exit, caller-fill substitution, standard-vs-flatten agreement,
the fee-inference matrix (zero-fee / missing-fee / ambiguous / scale-free),
base-vs-quote entry-fee convention, unexplained-residual fail-closed,
multiple-lifecycle rejection, single-lifecycle scale-in still reconciled,
the additive migration against a legacy schema, and MFE/MAE capture).

## 11. Downstream consumer audit (2026-10-25 review, requirement 4)

Every reader of `DecisionSnapshot.realized_pnl` / `return_pct` in the repo, with
whether it can currently see a non-`reconciled` row. **None of these were changed**
— wiring `outcome_status` into any of them is a strategy/risk-behavior change and
is explicitly out of scope. The list is the honest statement of the residual risk.

| consumer | reads | can it see corrupted rows? | impact |
|---|---|---|---|
| `bot._bootstrap_learner_from_history` (bot.py:3569) | `realized_pnl`, `return_pct` of all closed rows (limit 10000) | **YES** — no filter | adaptive meta-learner weights are seeded from history that includes the 2 impossible rows and any `unreconciled` row |
| `committee/adaptive_meta.rebuild_from_history` (adaptive_meta.py:366) | the same dicts | **YES** (via the caller) | the learner's per-brain stats and validation-gate return history inherit the same rows |
| `get_recent_realized_pnl` (db.py:1552) → rolling loss-limit gate (bot.py:2153) | `SUM(realized_pnl)` over a time window | **YES** — no filter | a corrupted negative `realized_pnl` can *trip* the entry block early; a corrupted positive one can *mask* a real drawdown. Fail-open on error. |
| `scripts/evolution_cull.evaluate_and_cull` | `SUM(realized_pnl)` last 30d as the production score | **YES** — no filter | candidate strategies could be culled/promoted against a corrupted production baseline |
| `attribution.analyze_closed_trade` (attribution.py:22) | one row's `realized_pnl`/`return_pct` | **YES** | the LLM explanation is fed a possibly-wrong number (narrative only; no gate) |
| `performance_tracker.record_trade` | `return_pct` passed by the caller | **NO** — the close paths pass the *reconstructed* value | OK once the close paths are fixed |
| `strategy_selector.record_strategy_outcome` | `return_pct` passed by the caller | **NO** — same | OK once the close paths are fixed |
| `scripts/analyze_deployed_trades.py` | raw columns | **YES, deliberately** — it flags them (`return_pct_anomalies`) | read-only diagnostic |

So the fix corrects what is *written* from now on, but three live consumers
(`_bootstrap_learner_from_history`, `get_recent_realized_pnl`'s loss gate,
`evolution_cull`) still read the *stored* column, including the legacy corrupted
rows. That is the residual risk the reviewer asked to make explicit. **Recommend
as a separate, approved change:** have `get_recent_realized_pnl` and the learner
bootstrap exclude `outcome_status != 'reconciled'`; both are behavior-affecting and
were left untouched here on purpose.

## 12. Ledger comparison of the corrected rows (2026-10-25 review, requirement 3)

The reviewer's caution is right: matching the old stored value is a cross-check,
**not** independent proof the old value was correct. Here is the per-row ledger
evidence for the three rows that changed, so the correction rests on the fills.

**ETH/USD `f110556591…`** — stored `-$0.037459`, reconstructed `+$0.125881`.
- Ledger: buy `0.006049 @ 2479.85` (0.037502); then **four** sells —
  `0.002068 @ 2514.51`, `0.002068 @ 2513.75`, `0.001896 @ 2511.21`, and a
  `0.00000188 @ 2508.71` dust leg (0.000012).
- `sell_qty = 0.00603388 = buy_qty × 0.997500` exactly → fully closed, entry fee
  withheld in base. `proceeds - basis - exit_fees = +0.125881`.
- Stored `-$0.0375 ≈ -entry_fee (0.037502)`: the old code took the **dust leg's**
  price as the exit, so the gross was ~0 and the full entry fee dominated. The
  stored `return_pct = -804.76%` divided that by the dust quantity. Both are the
  single-leg bug. The ledger says the position gained ~`+$0.126` on price.

**SOL/USD `35e92c4ad143…`** — stored `-$0.037615`, reconstructed `-$0.216326`.
- Ledger: buy `0.140999 @ 106.71` (0.037620); five sells totalling
  `0.14064650 = buy_qty × 0.997500` exactly.
- Same shape: stored `-$0.0376 ≈ -entry_fee`; the real net is `-$0.216326`.

**ETH/USD `c6a6d0423c…`** — stored `-$0.189452`, reconstructed **`unreconciled`**.
- Ledger holds **two** round trips under one `decision_id` (buy@10:33 → 3 sells
  @11:45-11:47; buy@13:32 → sell@18:34). Each leg's own `sell_qty = buy_qty ×
  0.9975`, but they are two lifecycles. RT1 net `-$0.0653`, RT2 net `-$0.1470`.
- The stored value is neither RT1 nor RT2 nor their sum — it is a third number
  from the single-leg formula.
- **Resolved in this revision:** the reconstruction returns
  `unreconciled / multiple_position_lifecycles` for this shape rather than
  summing the two lifecycles into one outcome (requirement 2).

All three ledger-confirmed corrections share one cause: the old close path used
the **final (often dust) exit leg's** quantity and price, then subtracted the full
round-trip fees. The reconstruction uses every leg and the verified fee
convention. The 9 rows that still match stored exactly are the rows where the
final leg *was* the whole position, so the old single-leg arithmetic happened to
be right.

An earlier draft of this section claimed `f110556591…` was under-closed and should
be `unreconciled`; that was wrong — it has five legs and closes at exactly
`0.9975`. Corrected here after re-reading all legs. (The residual guard does
already fail closed on a genuine under-close: a residual above the 0.5% band is
`quantity_mismatch`, as the 3.93x outlier shows.)

## 13. Scope and diff review (2026-10-25 review, requirement 5)

The tracked diff touches exactly four files — `src/bot.py`, `src/db.py`,
`src/risk.py`, `tests/test_reconciliation_and_db_recovery.py` — plus three new
files (`scripts/analyze_deployed_trades.py`, two test modules) and the report.
**0 staged. Protected paths untouched**: `src/exchange.py`, `strategies.py`,
`src/committee/*`, and `docker-compose.yml` are byte-identical to HEAD.

The migration is additive and nullable-only: `_ensure_additive_columns()` adds
one column, `decision_snapshots.outcome_status VARCHAR(16)`, via a
dialect-agnostic `ALTER TABLE ... ADD COLUMN` guarded by SQLAlchemy `inspect`.
No existing row is rewritten, no column is dropped, and `create_all()` never
ALTERs so the guard is a no-op on a fresh DB. Verified by
`test_additive_migration_adds_outcome_status_to_legacy_db`.

**Excursion-state handling (revised after the 2026-10-25 review — see below).**
The reviewer correctly flagged that an earlier revision of this patch *moved*
`_clear_symbol_excursion_state` into the recorder and added clears on the
recorder's early-return and `except` paths. That state (`risk_manager.peak_prices`,
`strategy._trailing_peaks`) feeds the trailing stop (`risk.py:1219`,
`strategies.py:906`), so retiming its clear is a **trading-behavior change** and is
outside the approved accounting scope.

The corrected implementation does exactly what the reviewer prescribed — capture
the values for the snapshot *before* cleanup, and leave the trailing-stop state's
clear untouched:

- The recorder's clear block is **byte-identical to HEAD** (same statements, same
  timing, happy path only). `git diff` shows **zero** changed lines in that block.
- The recorder takes two new optional kwargs, `excursion_peak`/`excursion_trough`.
  When supplied it uses them; when omitted it falls back to reading the strategy
  dicts (so the trailing-stop close path, which never pre-cleared, still records
  real MFE/MAE with no change to its behavior).
- The **standard close path** reads the two values into locals immediately *before*
  its existing HEAD clear, and passes them in. The clear itself is unchanged.
- No clear was added to any early-return or `except` path. The recorder now returns
  early **without** clearing, exactly as HEAD did.

Net effect: MFE/MAE are populated for the standard close path (HEAD wrote 0.0 for
both because the caller pre-cleared), and trailing-stop state timing is provably
unchanged. Tests `test_recorder_clear_timing_matches_head` and
`test_recorder_early_return_does_not_clear` pin this.

Boundaries observed: no order was placed, cancelled, or sent to any exchange
endpoint; the snapshot was analysed from a copy, read-only. Strategy files,
risk thresholds, and deployment config were not changed — the edits are confined
to the accounting path (`src/db.py`, `src/bot.py`, `src/risk.py`) and tests.
Nothing was staged or committed.

## 14. Fee-provenance gate (2026-10-25 review, primary blocker)

**Reviewer finding, reproduced.** Exact buy/sell quantities with `commission=0`
on both legs were certified `reconciled` with zero fees deducted:

```
buy 1.0 @100 commission=0 ; sell 1.0 @110 commission=0
-> outcome_status = "reconciled", realized_pnl = +10.0, entry_fees = 0.0, exit_fees = 0.0
```

**Root cause.** `OrderRecord.commission` is a plain `Float` defaulting to `0.0`;
there is no provenance flag. `AlpacaExchange._apply_estimated_commission`
computes a fee and sets `commission_estimated=True` **in the order dict**, but
`save_order_record()` has no such parameter and never persists it. The schema
therefore cannot distinguish a *verified* zero fee from a fee that was simply
never recorded. The residual classifier then treated `residual_frac == fee_rate
== 0` as the quote-charged convention and certified a fee-free P&L.

**Fix.** A new gate runs before the convention classifier. For a round trip
assembled entirely from the ledger (`from_ledger`), if either the entry or the
exit fee is `<= 0.0` the row is returned `unreconciled` with reason
`unverified_fee_provenance` and `realized_pnl`/`return_pct` = None. A missing
fee is unverified cost, so no fee-inclusive P&L may be *certified* from it.

The gate deliberately does **not** apply to caller-substituted legs: those are
already labelled `unreconciled`, and their fee is the caller's observed value
(a weaker evidence class). Only ledger rows can be certified, so only ledger
rows are gated. This keeps the approved best-effort numbers on the flatten and
ghost-close fallbacks.

**Impact on the deployed snapshot.** No currently-certified row is lost. The 11
reconciled round trips all carry nonzero commissions on both legs; the single
all-zero-fee round trip (`1fd27498650643ec…`) was already failing closed on a
3.93× quantity mismatch. The change bites future rows where a fee is missing --
e.g. a restored ledger with all-zero BTC commissions.

**Known config coupling (must be documented, not hidden).** If
`ESTIMATED_TAKER_FEE_BPS` is set to `0` -- which the config documents as
"disables fee estimation" -- then every live fill records `commission=0.0` and
this gate would mark *every* row `unreconciled`. Default is `25.0`, so the
deployed configuration is unaffected. Operators who disable fee estimation
should expect all reconstructed P&L to read `unreconciled`; that is the honest
consequence of having no fee evidence.

## 15. Lifecycle counter — direction of the tolerance (2026-10-25 review, blocker 2)

**Reviewer concern.** A partially-exited position re-entered could be merged
into one lifecycle if the merged quantities still match a fee convention,
because `_count_position_lifecycles()` closes a cycle at
`sold >= bought * (1 - 0.005)`.

**Result (narrowed 2026-10-25 review).** The threshold is a heuristic, not a
proof of position identity. Because it closes at `(1 - 0.005)` of buys --
*below* true flat (1.0) -- it errs toward **splitting**, which is fail-closed
(a spurious split yields `multiple_position_lifecycles` -> unreconciled).

An interior exit leaving a residual *larger* than the tolerance used to keep the
cycle open, fold a later buy into it, and -- if the aggregate returned to flat --
certify the merged row `reconciled` as a single lifecycle. **This is now fixed
(2026-10-25, approved two-hunk change):** a new `_has_ambiguous_reentry()` guard
fails such histories closed with `reason = "ambiguous_lifecycle_history"`,
distinct from `multiple_position_lifecycles`. Pinned by
`test_interior_residual_above_tolerance_is_failed_closed` (fails without the
guard) and four companions covering legitimate scale-ins, multi-leg sells,
multiple completed lifecycles, and genuinely-flat re-entries.

The trigger is a residual above the tolerance, which is a function of
**cumulative quantities and their ordering**, not of the configured fee rate. A
genuine partial exit (partial fill or scale-out) at the deployed 25 bps fee
reaches it, so this is **reachable on the deployed configuration** -- it is not
confined to a >0.5% fee regime. This limitation is therefore part of the
**acceptance criteria** for this patch, not a hypothetical to be dismissed.

The earlier enumeration (1,176,512 sequences, length 3-7, quantities
{0.25,0.5,0.75,1.0}) found zero under-splits **for those sequences and
quantities under the assumed fee convention** -- evidence, not a universal
guarantee. The reviewer's example (`buy 1.0, sell 0.996, buy 1.0, sell 0.996`)
closes the first cycle at the 0.996 sell and reports 2 lifecycles ->
`multiple_position_lifecycles`, unreconciled. Hardening the counter is
behavior-affecting and was **approved and incorporated** in this patch after
review (see `reports/LIFECYCLE_FIX_DESIGN_2026-10-25.md` for the design record).

**Test.** `test_partial_exit_then_reentry_is_not_silently_merged` pins the safe
outcome. No tolerance change was made: the reviewer's instruction not to alter
it without testing exchange precision and fee conventions is respected, and the
enumeration shows no unsafe merge exists to fix.

## 16. What "reconciled" does and does not claim (2026-10-25 review, design question)

The reviewer asked whether `reconciled` means "mathematically reconstructed" or
"verified against actual exchange costs". It means the former, and the code now
says so in three places so the label cannot be over-read:

- the `DecisionSnapshot.outcome_status` column comment (the authoritative scope
  note, next to the schema),
- the `reconstruct_position_outcome` return site, and
- the analyzer's human-readable output.

Precise definition:

- **reconciled** -- the P&L was rebuilt from the order ledger and the arithmetic
  balanced: quantities tied out within tolerance and both legs carried a
  **nonzero** commission.
- **not fee-verified** -- `commission` is an *estimated* value on this
  deployment (Alpaca reports none; the adapter derives it from
  `ESTIMATED_TAKER_FEE_BPS`) and the `commission_estimated` flag is not
  persisted, so a nonzero value proves a cost was *recorded*, not that it is
  *accurate*.

A nonzero commission is therefore necessary but not sufficient for fee
verification. Separating accounting reconciliation from fee verification needs a
fee-provenance column (`commission_source` / `commission_verified`) and a
persistence path for `commission_estimated`; that is a deliberate **follow-up**,
out of scope for this patch, and is **not** introduced here.

### `ESTIMATED_TAKER_FEE_BPS=0` is now a tested operating condition

`test_fee_estimation_disabled_makes_ledger_rows_uncertifiable` pins the
consequence: with estimation disabled the adapter yields `commission=0.0`, so
the fee-provenance gate returns every ledger-assembled round trip as
`unreconciled` / `unverified_fee_provenance`. This is conservative and intended
-- no fee evidence, no certified fee-inclusive P&L. Default is `25.0`; the
deployed configuration is unaffected.

### Out of scope, flagged for separate review

The rolling loss-limit consumer's handling of previously stored, potentially
incorrect P&L is unchanged by this patch and needs its own review and
authorization before any adjustment.
## 17. Where reconstructed P&L flows, and the limit of what this patch proves (2026-10-25 review)

The reviewer's qualification is correct and is now recorded in the code, not
just here: **nonzero estimated fees do not make the resulting P&L accurate.**
This patch makes the system *identify* cases where its recorded data is
insufficient; it does **not** make the surviving numbers audited. Any claim that
"the system now knows when its numbers are trustworthy" is too strong and is
withdrawn.

Reconstructed P&L does not merely sit in a table. It informs trading:

| Consumer | Site | Uses reconstructed P&L how |
| --- | --- | --- |
| Rolling soft loss limit | `bot.py` -> `get_recent_realized_pnl` | sums ALL closed rows (no `outcome_status` filter) to decide whether to block a new entry |
| Meta-learner adaptive weights | `bot.py` `learner.update(...)` | `net_pnl`/`return_pct` drive regime weighting |
| Strategy selector | `bot.py` `record_strategy_outcome(...)` | per-strategy reward signal |
| Live training buffer | `bot.py` tensor label | `1.0 if realized_pnl > 0 else 0.0` |
| OOD retraining | `bot.py` -> `get_closed_decision_snapshots` | training history |
| Attribution | `attribution.py` | per-trade P&L |

Two consequences, both deliberate and both **out of scope** for this patch:

1. The rolling loss limit sums every closed row regardless of trust label. That
   is fail-safe by design -- under-counting losses to avoid a block is the more
   dangerous error -- but it means a reconstructed, fee-estimated number gates a
   trading decision. Filtering on `outcome_status` would change trading behavior
   and needs separate review and authorization.
2. The learner/selector/buffer all consume the reconstructed values as signals.
   They are working signals, not audited performance.

Each of these sites now carries an inline NOTE saying exactly this, so the
limitation travels with the code rather than living only in a report.

### What this patch does and does not establish

- Does: quantities and arithmetic reconcile for a round trip assembled from the
  ledger; a missing fee is not silently treated as zero; a merged/split lifecycle
  or an unexplained residual fails closed.
- Does not: establish that the recorded (estimated) fees match what the exchange
  charged, or that the reconstructed P&L is audited. Closing that gap needs a
  persisted fee-provenance field and is a documented follow-up.
