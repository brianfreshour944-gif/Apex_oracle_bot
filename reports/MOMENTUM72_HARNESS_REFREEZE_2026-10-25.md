# MOMENTUM72 — HARNESS RE-FREEZE (instrument corrected, hypothesis unchanged)

Date: 2026-10-25 (re-freeze)
Author: OpenHands. Authorized scope: **harness fix only, no specification
change.** No validation data fetched. Strategy not run on validation.

Base commit: 7a8ca12 (branch main), 0 staged.
Frozen spec sha256 **unchanged**: `08316c38e11f8e20e6fcb795c5fa4daa220c919364209ad808b2a865a21af854`.
Accounting patch sha256 **unchanged**: `d1131e48b4c58e4f21db4f30aeca196c116b2a7cff2e37ac5cd8907cdfbfd4a9`.

---

## 1. What was wrong (recap)

`scripts/momentum72_study.py` did not implement the frozen spec §3:

- **Return timing off by one.** `run(delay=0)` earned `w_{i-1} × logret_i`; the
  spec's primary is `w_i × logret_{i+1}`. The harness's headline was actually the
  spec's *one-bar-delayed sensitivity*.
- **Per-trade net was the closing bar only** (line 116), not the full round trip —
  corrupting criterion 2 (mean) and criterion 3 (bootstrap p).
- **Final open trade recorded as `0.0`** (line 122), not its real net.
- **Exposure/turnover used `prev`** rather than the bar's own target `tgt`.
- **IC and backtest used different timing conventions** (IC was already
  spec-correct; the backtest was not).

## 2. What changed (harness only)

`scripts/momentum72_study.py`:

| Before | After | Spec basis |
| --- | --- | --- |
| `r_gross = prev * lr[i]` | `r_gross = tgt * lr[i]` | §3 `w_i * logret_{i+1}` |
| `exposure_sum += abs(prev)` | `exposure_sum += abs(tgt)` | §3 (exposure of bar *i+1* is `w_i`) |
| trade net = closing-bar `r_net` | trade net = Σ `r_net` over the run | §5.2/§5.3 (per-trade net) |
| final open trade `trades.append(0.0)` | `trades.append(cur_net)` | §3 (see §4 below) |
| run detection at `prev != 0` | run detection at `tgt != 0` | §5.1 (maximal run of `w > 0`) |

`delay=0` now reproduces the spec's primary; `delay=1` reproduces the spec's
one-bar-delayed sensitivity. No parameter, lookback, cost, or criterion changed.

## 3. Regression tests added (`tests/test_momentum72_study.py`)

Six new tests, each pinning an **independently hand-computed** value (not a mere
property). Plus an independent `_reference_engine()` transcription of spec §3 used
as a cross-check.

| Test | What it pins | Expected value |
| --- | --- | --- |
| `test_primary_return_timing_matches_spec_jump_case` | primary timing | `[100]×72+[101,110]` → `log(110/101) = 0.08535984895115685`; delayed → `0.0` |
| `test_delay_one_equals_spec_one_bar_delayed_convention` | delay=1 == `w_{i-1}·lr_i` | primary `0.047856…`, delayed `0.038100…`, and they differ |
| `test_run_matches_literal_spec_transcription_on_a_mixed_series` | whole engine vs literal §3 | gross/net/per-trade all match to 1e-12 |
| `test_full_round_trip_net_charges_entry_and_exit` | entry **and** exit cost | `[100,100,100,200,100,100,100]` → net = gross − 2×`w3`×28bps |
| `test_final_open_trade_is_not_liquidated` | final open run | `[100,101,102,103,104]` → `log(104/102) − 0.0028`; guard `!= 0` |
| `test_information_coefficient_uses_the_primary_convention` | IC timing | hand-computed correlation |

## 4. The one point flagged for your ratification: end-of-period convention

**The spec does not state an explicit liquidation rule.** I searched the spec and
the repo; the only relevant statements are:

- §3: "cost charged per bar = `|w_i − w_{i-1}| × COST_ONE_WAY`" — an exhaustive
  per-bar rule with **no exit event**.
- §3: "A full entry then exit therefore costs 56 bps" — describes a closed trade.
- §4: buy-and-hold is charged one round trip — a separate, explicit statement
  about the benchmark.
- §5.3: the bootstrap uses "the same … method as the existing study" — but that
  clause names the *statistical method*, not the liquidation rule.

**Interpretation I implemented (derived, not invented):** because §3 defines the
return exhaustively per bar and names no liquidation, a run still open at the
final bar is **not force-closed**: it accrues its per-bar returns and pays only
the entry/resize costs it actually incurred. This is a literal reading of §3.

**Alternative reading:** the sibling study (`scripts/edge_study.py`) force-closes
its final open position at `px[-1]`, charging a full round trip. If you prefer
that convention, it is a **one-line change** and would be a *spec clarification*,
not a retune. I did not adopt it, because adopting it would be choosing a
liquidation rule the spec does not state — which your instructions prohibit.

**RATIFIED (2026-10-25):** the operator accepted the §3-derived reading above —
a run open at the final bar is **not force-liquidated** and pays no exit cost.
No code change was required (the corrected harness already implements it). The
force-close alternative is rejected for this study.

**Impact is negligible for this study:** on the already-inspected development
period, the final bar's target is flat (`w[-1] = 0.0`) for all three symbols, so
the final open run does not occur there. The choice matters only if a validation
period happens to end while long.

## 5. Verification performed (all synthetic; no validation data)

- Prerequisites: spec sha256 `08316c38…` ✓; HEAD `7a8ca12`; 0 staged; patch
  `d1131e48…` ✓; protected paths UNCHANGED.
- Harness tests: **18 passed** (12 original + 6 new).
- Related tests: `test_momentum72_study` + `test_edge_study` +
  `test_analyze_deployed_trades` → **49 passed**.
- Full suite: **570 passed** (564 baseline + 6 new).
- Ruff: **All checks passed!**
- Exact diff vs the pinned rev-2 bundle harness (`649b2343…`) reviewed: only
  return timing, per-trade net, final-open handling, exposure accounting, and
  docstrings changed. Nothing else.

## 6. Re-freeze record

| Artifact | sha256 |
| --- | --- |
| `scripts/momentum72_study.py` (corrected) | `24f39a9d7da70a02b5f8c7c63a0e5aa0efa0dfe2df51e0722fd5e12e1c35f752` |
| `tests/test_momentum72_study.py` (corrected) | `5dec4b927e8f283efee9bf28b1d201f8fdcc7e20fa2f10c1aa90354b9db19634` |
| `reports/MOMENTUM72_SPEC_2026-10-25.md` (UNCHANGED) | `08316c38e11f8e20e6fcb795c5fa4daa220c919364209ad808b2a865a21af854` |
| Accounting patch (UNCHANGED) | `d1131e48b4c58e4f21db4f30aeca196c116b2a7cff2e37ac5cd8907cdfbfd4a9` |

The original spec and its original hash remain the historical record. The
hypothesis is not re-frozen; only the instrument is.

## 7. Safety boundary (confirmed)

- **No validation data fetched.** `data/ohlcv/` unchanged; no
  `data/ohlcv_dev2021_2025/`. No bars downloaded.
- Strategy **not** run on 2021-01-01..2025-09-30 or any validation period.
- `MOMENTUM72_RESULTS_2026-10-25.md` **not** created.
- Accounting/lifecycle patch, corrected reports, recovery bundles, protected
  files, and production behavior all UNCHANGED.
- 0 staged; nothing committed, deployed, or restarted.

## 8. Status of your two decisions

1. **End-of-period reading — RATIFIED (2026-10-25).** The §3-derived reading in §4
   stands: a run open at the final bar is not force-liquidated and pays no exit
   cost. No code change was needed.
2. **Fetch authorization — NOT YET GIVEN.** Step 3 (fetch 2021-01-01T06:00Z..
   2025-09-30 and the single frozen run) remains on hold. The harness is frozen
   and ready; nothing further will be fetched or run until you authorize it.

No parameter, period, or criterion was chosen or tuned in producing this report.
