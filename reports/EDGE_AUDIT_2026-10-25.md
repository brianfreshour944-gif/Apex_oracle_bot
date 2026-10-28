# Audit of the Edge-Investigation Negative Result — independent, read-only

Date: 2026-10-25
Auditor: OpenHands (read-only). No repository file was modified, staged, committed,
reset, restored, or deleted. The uncommitted accounting/lifecycle patch is intact.
Base commit: 7a8ca12 (branch main).

Artifacts inspected:
- reports/EDGE_INVESTIGATION_2026-10-25.md
- reports/UNCOMMITTED_PATCH_2026-10-25.md
- tests/test_edge_study.py
- scripts/edge_study.py, scripts/fetch_ohlcv.py
- data/ohlcv/{BTC,ETH,SOL}_USD_1Hour.json (gitignored cache)

All numbers below were produced this session with the commands shown, unless
explicitly labelled "reported". A distinction is kept throughout between
**verified** (I ran it) and **reported** (from the study/report).

---

## Verdict up front

The negative result is **robust**. It is not an artifact of the cost model, the
split, the p-value method, look-ahead, or an implementation defect. The tested
signal family carries **no detectable predictive information** (gross returns and
information coefficients are indistinguishable from zero), so it fails for the
most fundamental reason: weak signal, not merely turnover or fees. The specific
reported p-values are sound and in fact conservative.

Two genuine methodological caveats are recorded below (multiple-testing framing,
statistical power), but neither rescues a positive result, because **no test
came close to significance** (minimum p across all 42 in-sample cells = 0.8898).

**Recommendation: STOP this strategy family.** Do not repair the experiment; it
is not broken. Do not proceed to a new hypothesis within the same signal family.
See "Next experiment" for the one bounded test worth running before concluding.

---

## 1. Data integrity — PASS

Verified with `scripts`-level reads of the cached bars.

| Check | BTC | ETH | SOL |
| --- | --- | --- | --- |
| Bars | 9335 | 9334 | 9329 |
| Duplicate timestamps | 0 | 0 | 0 |
| Monotonic increasing | yes | yes | yes |
| Non-1h gaps | 1 | 2 | 5 |
| Invalid OHLC (h/l/o/c) | 0 | 0 | 0 |
| In-sample / holdout split | 6551/2784 | 6550/2784 | 6545/2784 |

- Timestamps are UTC (`...Z`), strictly increasing, no duplicates.
- Missing bars are genuine exchange gaps (e.g. BTC 2026-03-12 17:00 absent; SOL
  has a cluster on 2025-10-10/11). They are sparse (≤5 per symbol over a year)
  and are handled as a single longer interval, not as a zero-return bar. This is
  correct and immaterial to the conclusion.
- The split boundary is exact: `t < "2026-07-01T00:00:00Z"` for in-sample,
  `>= ` for holdout. First holdout bar is 2026-07-01T00:00:00Z on all three.
  No bar is in both halves; no bar is dropped.
- OHLC relationships hold on every bar (l ≤ o,c ≤ h).

No bar or feature crosses the boundary. Signals use only closes at or before bar
*i*; there is no future information (see §2).

## 2. Signal and execution realism — PASS (with one conservative convention)

Reconstructed each hypothesis from `scripts/edge_study.py` (`h1_momentum`,
`h2_meanrev`, `h3_ma_cross`, `h4_vol_breakout`, `backtest`).

- **Timing / look-ahead.** Signal at bar *i* uses closes ≤ *i*. `h1_momentum`
  and friends return flat until enough history (`i < lookback`) — no look-ahead.
  Execution occurs at the close of the *same* bar *i*. This is the standard
  "signal and fill on the same close" convention; it is mildly optimistic only
  if one believes the close is not tradable at the close.
- **Robustness to that convention.** I re-ran with execution lagged one bar
  (signal at *i*, fill at *i+1*). Every hypothesis remains deeply negative
  (e.g. BTC H1_momentum_24h: −59.55 → −69.94 bps; SOL H2_meanrev_72h:
  −52.57 → −50.28 bps). The conclusion does not depend on the fill convention.
- **Cost charging.** A flat `ROUND_TRIP_BPS = 56` (25 fee + 3 slippage per side)
  is subtracted once per round trip, in `backtest` and `buy_hold` alike. It is
  applied consistently. It is a **modelled** cost, not a measured fill cost —
  stated as a limitation, not a defect.
- **Shorts / cash / sizing / overlap.** The harness holds one position at a time
  (long, flat, or short), no leverage, no compounding of concurrent positions.
  `allow_short` maps the long/flat signal's 0 to −1 ("always in market"), which
  is a clean way to strip out beta. There is no position sizing to audit beyond
  unit exposure. No overlapping positions are modelled; this is appropriate for
  a single-asset, one-position test.
- **Fill realism.** Fills at bar closes are the only assumption. Given the
  conclusion is strongly negative and unchanged under a one-bar lag, no
  unrealistically favourable fill is propping up a result.

**Not a defect.** The same-close fill and flat 56 bps are simplifications, not
errors, and both are tested for sensitivity.

## 3. Statistical validity — SOUND (conservative); two framing caveats

- **p-value construction.** `block_bootstrap_p` resamples contiguous blocks of
  trade returns (block = 12) and returns the fraction of resamples with mean ≤ 0.
  This is a **moving-block bootstrap**, which is the right tool for
  autocorrelated/overlapping trade returns and is *more* conservative than an
  i.i.d. t-test. Verified: for BTC H1_momentum_24h, the naive t is −8.84
  (two-sided p ≈ 0) while the block bootstrap gives p = 1.0000 — they agree the
  mean is not positive; the bootstrap simply does not overstate significance.
- **Serial correlation / overlapping trades.** Handled by the block bootstrap.
  The report does not overclaim from it.
- **Multiple testing (CAVEAT, methodological).** The study effectively runs
  7 hypotheses × 3 symbols × 2 modes = **42 tests**. The report does not state a
  family-wise correction. Verified: the **minimum in-sample p across all 42
  cells is 0.8898**, and **zero cells have p < 0.05**. So no correction —
  Bonferroni (0.05/42 = 0.00119) or Benjamini–Hochberg — would change anything;
  the family-wise conclusion is "nothing is significant" either way. The
  uncorrected framing is a presentational gap, not a result-changing one.
- **Holdout purity.** Verified from code and the report: the holdout
  (2026-07-01..2026-10-24) was split by timestamp only and the hypotheses are
  fixed in code before any result is seen; no parameter was tuned on holdout.
  The two positive holdout cells are SOL trend-following and both are
  insignificant (see §4).
- **Power (CAVEAT, methodological).** With observed per-trade σ ≈ 118 bps,
  detecting a 10 bps/trade edge at 80% power needs ≈ **1097 trades**; even a
  30 bps edge needs ≈ **122**. Observed counts are 30–310 per cell. The study is
  therefore **underpowered to rule out small edges**. This is a real limitation:
  "no convincing edge found" is correct; "no edge exists" is not claimed and
  would not be supported.

## 4. Benchmark fairness — PASS

- **Comparable periods / capital / return definition.** `buy_hold` uses the same
  bar series, the same 56 bps round trip, and the same simple-return definition
  as the strategies. Verified: SOL holdout buy-and-hold net = **+38.52%**.
- **Long/flat vs long/short.** Reported side by side, so beta is visible. In the
  bear in-sample window, long-only strategies (−48% to −91%) were *worse* than
  buy-and-hold (−49% to −65%); in the bull holdout they failed to capture the
  recovery. This is a fair presentation.
- **Reported holdout cells reproduce exactly (verified):**
  - SOL H3_ma_12_48: mean +4.62 bps, total −1.93%, p = 0.3612 (reported 4.6/0.36 ✓)
  - SOL H3_ma_24_168: mean +73.75 bps, total +8.13%, p = 0.2834 (reported 73.8/0.28 ✓)
  Both underperform buy-and-hold (+38.5%) and are insignificant.
- **Risk / exposure (verified, in-sample, long/flat):**
  - Max drawdown: −48% to −91% (worse than or equal to buy-and-hold's −49% to −65%).
  - Exposure (time in market): ~0.41–0.55 for most; **H4_breakout_48 ≈ 0.06**
    (it is almost always flat, so its −68% total comes from few, badly-timed trades).
  - Turnover: 30–310 round trips over ~6550 bars; cost drag 1,680–17,360 bps.

## 5. Negative-result diagnosis — WEAK SIGNAL (primary), costs secondary

This is the central question: do the signals have predictive value that costs
consume, or none at all? **Verified evidence says: essentially none.**

- **Information coefficient (corr of signal with next-bar return):** every
  hypothesis on every symbol gives |IC| ≤ 0.025, most ≤ 0.01. That is noise.
- **Gross (zero-cost) mean per trade:** BTC −3.6 to −113.5 bps; ETH −13.7 to
  −170.7; SOL −6.8 to −195.5. Only three cells are gross-positive and all are
  tiny: BTC H4_breakout +1.16 bps, ETH H4_breakout +3.66, SOL H2_meanrev_72h
  +3.43 — i.e. far below the 56 bps cost and well inside noise.
- **Null model (verified).** Random long/flat at the same exposure yields gross
  mean −1.65 / −2.49 / −2.46 bps (sd ≈ 1.1–2.0). The actual hypotheses' gross
  means sit **inside the null distribution**. The rules are statistically
  indistinguishable from random with respect to next-bar direction.
- **So:** losses are dominated by (a) no directional information and (b) the
  residual cost of repeatedly trading a zero-edge signal. It is *not* primarily
  an implementation defect, and *not* primarily an over-tight cost model — the
  gross signal is already ≈ 0 before any cost is charged.

I will not over-attribute beyond this. The evidence distinguishes "no signal"
from "signal eaten by cost" clearly (the former), but cannot say whether some
*other* untested signal family would work.

---

## Findings, ranked by severity

| # | Severity | Finding | Evidence |
| --- | --- | --- | --- |
| F1 | HIGH (scientific) | Tested signal family has no detectable predictive edge | IC ≤ 0.025; gross mean ≈ 0; null model matches; scripts/edge_study.py:HYPOTHESES |
| F2 | MEDIUM (method) | Underpowered to exclude small (<30 bps) edges | σ≈118 bps → n≈1097 needed for 10 bps; 30–310 observed |
| F3 | LOW (presentation) | No stated family-wise correction for 42 tests | min p = 0.8898, so no correction changes the conclusion |
| F4 | LOW (limitation) | Cost is modelled (56 bps), not measured from fills | edge_study.py:TAKER_FEE_BPS/SLIPPAGE_BPS |
| F5 | LOW (limitation) | Single year, single bear→bull cycle | data span 2025-10-01..2026-10-24 |

## Confirmed defects / methodological concerns / limitations

| Category | Item | Status |
| --- | --- | --- |
| Confirmed defect | **None found** | — |
| Methodological concern | Underpowered for small edges (F2) | Real; does not create a false negative conclusion beyond "no edge demonstrated" |
| Methodological concern | Multiple-testing correction not stated (F3) | Cosmetic; min p = 0.89 |
| Limitation | Modelled, not measured, costs (F4) | Bounds interpretation; gross ≈ 0 makes it moot |
| Limitation | One regime cycle (F5) | Bounds external validity |
| Limitation | Same-close fill convention | Tested with 1-bar lag; conclusion unchanged |
| Limitation | Only 7 simple hypotheses | A different signal family is untested |

## Verified vs reported

- **Verified this session:** bar counts, dupes, gaps, monotonicity, OHLC validity,
  split counts and boundary; gross-vs-net decomposition; ICs; null model;
  1-bar-lag sensitivity; turnover, exposure, max drawdown; the two SOL holdout
  cells; min in-sample p across 42 cells; block bootstrap vs naive t;
  `pytest tests/` = **552 passed**; `pytest tests/test_edge_study.py` = **6 passed**.
- **Reported by the study (reproduced, not merely trusted):** the full in-sample
  and holdout tables in reports/EDGE_INVESTIGATION_2026-10-25.md; the headline
  "no convincing edge" conclusion — reproduced and endorsed.
- I could not retrieve the recovery bundle as an archive in the browser, so I did
  not independently verify the *bundle*; I verified the *working tree* directly,
  which is stronger for this purpose.

## Recommendation

**STOP.** Do not repair the experiment (it is sound) and do not pursue another
hypothesis inside this signal family (it has no measurable edge). The correct
next move is one bounded, pre-registered test that could *falsify* the "no
signal" conclusion on genuinely new data — not more complexity.

## Proposed next experiment (only if you want to spend the effort)

**Hypothesis (single, pre-registered):** a *volatility-scaled* time-series
momentum rule — long when the trailing 72h return divided by trailing 72h
realized volatility is positive, flat otherwise — has positive gross mean return
per trade over hourly bars.

Rationale: it is the one structural variation (vol-normalised trend) not covered
by the seven tested rules, and it is cheap to test.

Design:
- **Data:** new symbols and/or a period not used here. Reserve an **untouched
  validation window** (e.g. the most recent 3 months) that is not inspected until
  the final step.
- **Costs:** 56 bps round trip, as here.
- **Multiple testing:** declare this as **1** test; report family-wise status.
- **Success criteria (all must hold on the validation window):**
  1. gross mean ≥ 56 bps/trade (i.e. it can clear cost), and
  2. block-bootstrap p < 0.05 (one-sided), and
  3. net total return > buy-and-hold net total return over the same window, and
  4. ≥ 100 trades in the window (power).
- **Failure criterion:** if gross mean < cost on the validation window, abandon
  the family entirely. Do not tune and re-test on the same window.

If that single test fails, the evidence-based conclusion is that this project's
signal family is not worth pursuing, and effort should go to execution
economics or to a different, explicitly-motivated hypothesis — not to a larger
committee.
