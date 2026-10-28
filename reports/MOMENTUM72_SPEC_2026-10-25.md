# FROZEN SPECIFICATION — 72-Hour Volatility-Scaled Momentum

Status: **FROZEN — written before any validation evaluation.**
Frozen at (UTC): 2026-10-25T22:26Z
Base commit: 7a8ca12 (branch main). Research-only; production untouched.

This document is written and committed to disk *before* downloading or
evaluating validation data. No parameter below may change after this point.

---

## 0. Data partition (fixed before looking at anything)

| Period | Role | Status |
| --- | --- | --- |
| 2025-10-01 .. 2026-10-24 | **Development** | ALREADY INSPECTED (prior study + audit) |
| 2021-01-01 .. 2025-09-30 | **Validation** | **GENUINELY UNTOUCHED** |

Justification for "untouched": the prior study downloaded bars only from
2025-10-01 (`scripts/fetch_ohlcv.py --start 2025-10-01`); the cache contains
nothing earlier. Alpaca serves hourly bars back to 2021-01-01T06:00Z (verified by
metadata query only, `limit=1`). Therefore 2021-01-01..2025-09-30 has never been
downloaded, viewed, or used. The July–Oct 2026 holdout is **NOT** reused; it is
part of the already-inspected development period and is excluded from validation.

Validation is used **once**. If the strategy fails, the hypothesis is rejected; no
retuning on validation.

## 1. Universe

BTC/USD, ETH/USD, SOL/USD — all three have comparable hourly history from
2021-01-01. If any symbol's validation coverage is materially worse than the
others, it is documented (before outcomes) and excluded; the primary result is
reported per symbol and pooled.

## 2. Hypothesis (one primary configuration)

The sign of the trailing 72-hour price change predicts the sign of the next
holding-period return; position exposure is scaled inversely to trailing realized
volatility. **Primary configuration: LONG/FLAT** (the predeclared primary). The
long/short variant is reported as secondary context only and is not a pass
criterion.

## 3. Exact definitions (frozen)

- **Bars:** completed 1-hour bars, close-to-close. No same-bar execution.
- **Momentum signal at bar *i*:** `mom_i = log(close_i / close_{i-72})`.
  Long candidate if `mom_i > 0`, else flat.
- **Volatility estimator:** sample stdev (ddof=1) of the 72 hourly log returns
  ending at *i*: `sd_i = stdev(logret_{i-71..i})`. Uses only bars ≤ *i*.
- **Volatility lookback:** 72 hours.
- **Position sizing (exposure), capped:**
  `w_i = min(1.0, VOL_TARGET / sd_i)` when long, else `0.0`.
  - `VOL_TARGET = 0.005497` (0.550%/hr) — the **pooled median trailing-72h hourly
    vol over the DEVELOPMENT period only** (BTC 0.4059%, ETH 0.5497%, SOL 0.6281%).
    This single constant is the only quantity taken from development data.
  - **Exposure cap:** 1.0 (no leverage). Floor: 0.0.
- **Entry/exit schedule:** at each hourly bar close *i*, compute `w_i`. If the
  target exposure changes from `w_{i-1}`, trade the delta at the close of bar *i*
  (the first instant the information is available). Hold until the target changes.
  No intrabar stops or targets.
- **Missing bars:** gaps are not forward-filled. A return spanning a gap is a
  single longer-period return; the vol window uses the returns that exist. If the
  number of returns in the trailing window is < 72, no position is taken (warmup).
- **Turnover-based cost accounting (resizing supported):**
  cost charged per bar = `|w_i - w_{i-1}| * COST_ONE_WAY`, where
  `COST_ONE_WAY = 28 bps` (= half of the study's 56 bps round trip). A full entry
  then exit therefore costs 56 bps, consistent with the existing study. Costs are
  subtracted from the bar's return each time exposure changes.
- **Return accounting:** strategy return for bar *i+1* =
  `w_i * logret_{i+1} - |w_i - w_{i-1}| * COST_ONE_WAY`.
  Gross = same without the cost term. Compounded multiplicatively (exp of sum of
  log returns) for total return.
- **Capital:** unit notional, no leverage, single asset at a time, no compounding
  of concurrent positions. Identical capital assumption for the benchmark.

## 4. Predeclared benchmark and baselines

- **Buy-and-hold (primary benchmark):** long exposure 1.0 from the first
  validation bar to the last, charged the same 56 bps round trip once.
- **Flat/cash baseline:** 0% return, no cost.
- Buy-and-hold uses the same bars, the same period, and the same 56 bps cost
  convention.

## 5. Pre-registered success criteria (ALL must hold on validation)

1. **≥ 100 completed trades.** Definition: a *trade* is one entry-to-flat (or
   flat-to-entry) exposure episode of non-zero size — i.e. each maximal run of
   bars with `w > 0`. Resizes within a run are not new trades.
2. **Mean gross return ≥ 56 bps per trade** (must clear modeled cost).
3. **p < 0.05** via a **moving-block bootstrap** of per-trade net returns
   (block = 12), one-sided, H0: mean ≤ 0 — the same serial-dependence-aware method
   as the existing study. (A naive t-test is also reported for transparency.)
4. **Net total return > buy-and-hold net total return** over the same period.
5. **No look-ahead or accounting defect** that could explain the result.

Also reported (not pass criteria): max drawdown, total exposure, turnover,
estimated transaction costs, information coefficient (signal vs next-bar return),
and sensitivity to execution delayed by one bar.

## 6. Verdict rule

- If **gross mean < 56 bps/trade on validation → FAIL (reject hypothesis).**
- If any other criterion fails → **FAIL** (or **INCONCLUSIVE** where the data
  cannot distinguish, e.g. < 100 trades).
- All criteria pass → **PASS**.
- No retuning on validation under any outcome.

## 7. Isolation

New artifacts only, no production files touched:
- `scripts/momentum72_study.py` (harness)
- `tests/test_momentum72_study.py` (deterministic unit tests)
- `reports/MOMENTUM72_SPEC_2026-10-25.md` (this file)
- `reports/MOMENTUM72_RESULTS_2026-10-25.md` (written after evaluation)
- `data/ohlcv_dev2021_2025/` (validation cache, gitignored)

The existing uncommitted accounting/lifecycle patch, all existing reports, and
all production code remain unchanged.
