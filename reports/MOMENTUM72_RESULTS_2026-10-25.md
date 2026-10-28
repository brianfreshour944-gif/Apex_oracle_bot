# MOMENTUM72 — RESULTS (frozen experiment, validation run once)

Date: 2026-10-25
Author: OpenHands. Single frozen run on the genuinely untouched window. No
retuning, no alternative variants, no parameter selection after seeing results.

- Base commit: 7a8ca12 (branch main), 0 staged.
- **Spec sha256 `08316c38…a21af854` — unchanged** (hypothesis untouched).
- **Harness sha256 `24f39a9d…e1c35f752`** — the re-frozen instrument.
- Accounting patch `d1131e48…` — unchanged. Protected files unchanged.

Window: **2021-01-01T06:00:00Z .. 2025-09-30T23:00:00Z**. Data manifest and the
pre-outcome SOL-coverage decision: `MOMENTUM72_VALIDATION_MANIFEST_2026-10-25.md`.

## VERDICT: **FAIL** — the hypothesis is rejected under the frozen rules.

---

## 1. Headline

| Symbol | Bars | Trades | Gross bps/trade | Net bps/trade | Gross total | Net total | Buy&hold | Max DD | IC | p (boot) | p (t) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC/USD | 41,597 | 1,133 | 12.59 | −39.28 | +316.2% | **−98.8%** | +288.0% | −99.1% | 0.006 | 1.000 | <1e-6 |
| ETH/USD | 41,596 | 1,108 | 12.58 | −35.61 | +303.1% | **−98.1%** | +456.5% | −98.8% | 0.005 | 1.000 | <1e-6 |
| SOL/USD | 31,521 | 844 | 33.17 | −1.88 | +1543.2% | **−14.7%** | +12,660.1% | −79.9% | 0.007 | 0.583 | 0.54 |
| **Pooled BTC+ETH** (primary) | — | 2,241 | 12.58 | −37.47 | — | — | — | — | — | 1.0000 | <1e-6 |
| Pooled all three (secondary) | — | 3,085 | 18.21 | −27.73 | — | — | — | — | — | 0.9998 | <1e-6 |

The gross signal is slightly positive (12–33 bps/trade) but **far below the 56 bps
cost bar**, and net returns are negative everywhere. The bootstrap p is ≈1.0
(the opposite of the required p<0.05): under this test the net per-trade mean is
not distinguishable from ≤0 in the required direction. (See §5 for the precise
interpretation of a p of 1.000.)

## 2. Criterion-by-criterion verdict (spec §5; ALL must hold)

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | ≥ 100 completed trades | **PASS** | 1,133 / 1,108 / 844 per symbol; 2,241 pooled (BTC+ETH) |
| 2 | Mean **gross** ≥ 56 bps/trade | **FAIL** | BTC 12.59, ETH 12.58, SOL 33.17 — all < 56 |
| 3 | p < 0.05 (block bootstrap, one-sided) | **FAIL** | p = 1.000 / 1.000 / 0.583; pooled 1.0000 |
| 4 | Net total > buy-and-hold net total | **FAIL** | BTC −98.8% vs +288.0%; ETH −98.1% vs +456.5%; SOL −14.7% vs +12,660.1% |
| 5 | No look-ahead or accounting defect | **PASS** | instrument re-frozen and cross-checked; see §4 |

Spec §6: "If gross mean < 56 bps/trade on validation → FAIL (reject hypothesis)."
That single decisive condition already fails on every symbol. Criteria 3 and 4
also fail. **Overall: FAIL.**

## 3. What the numbers say

- **The cost bar is decisive.** Gross per trade is ~13 bps (BTC/ETH) and ~33 bps
  (SOL) against a modeled 56 bps round trip. Even a *perfectly* cost-free version
  would be marginal; with costs the strategy loses.
- **Turnover is enormous.** BTC 11,336 exposure changes, ETH 16,111, SOL 15,840 —
  the vol-scaling resizes almost every bar. Total modeled cost: BTC 58,767 bps,
  ETH 53,396 bps, SOL 29,581 bps of log-return drag over the window. The signal is
  far too small to pay for its own turnover.
- **Gross total is positive but irrelevant.** +316% / +303% / +1543% gross become
  −99% / −98% / −15% net. The gross figures merely show the underlying long-bias
  in a rising market; buy-and-hold captured that beta far better (up to +12,660%
  for SOL).
- **The signal has almost no information.** IC ≈ 0.005–0.007 (essentially zero).
  Win rate 15–23%. The negative naive t-stats (−5.5, −4.8, −0.1) confirm the net
  mean is negative, not merely insignificant.
- **SOL is not an exception.** Its gross is higher (33 bps) but still below cost,
  and its net is negative. Its p=0.583 is a *failure to reject* H0, not support.

## 4. Criterion 5 — no look-ahead or accounting defect

- The instrument was corrected and independently re-verified before the run
  (`MOMENTUM72_HARNESS_REFREEZE_2026-10-25.md`): primary timing `w_i·logret_{i+1}`,
  full round-trip per-trade net, final-open convention ratified, IC consistent
  with the backtest. 18 hand-pinned tests pass.
- Weights use only bars ≤ *i* (causality test included); the 72-bar warmup is
  enforced; the vol window is trailing.
- Cost is charged on every `|Δw|` including resizes, so turnover is not hidden.
- **Limitations stated, not glossed:** the end-of-period convention for a final
  *open* run is a spec-derived reading (ratified); it does not affect this result
  because all three symbols end flat at 2025-09-30T23:00Z (`w[-1]=0`).

## 5. Statistical caveat (as pre-registered) — precise interpretation

**What a p of 1.000 does and does not mean.** The reported value is the
pre-registered moving-block bootstrap one-sided p for H0: mean net ≤ 0. A value
near 1.0 says the resampled mean is essentially never ≤0 — i.e. *under this
particular null, bootstrap construction (block = 12, 5,000 resamples, seed
12345), and test statistic*, there is no support for a positive net mean. It does
**not** assert that the strategy has literally zero probability of an edge; the
number is a property of the specified test, not a universal statement about the
hypothesis.

**Sample size ≠ independence.** ~1,100 trades per symbol is a large count, but
count does not by itself guarantee independent evidence: trades overlap in time,
the signal is autocorrelated (trailing 72 h), and market returns are serially
correlated. That is exactly why the pre-registered test is a block bootstrap
rather than an i.i.d. t-test. The dependence is **mild but non-zero** — hourly
|acf₁| ≤ 0.024 in development and trade-level lag-1 correlation small — so the
effective sample is smaller than the raw count, which weakens (not strengthens)
any significance claim. It does not rescue this result: the decisive failure is
criterion 2 (gross < 56 bps on every symbol), which is a point estimate and does
not depend on the bootstrap at all.

**Precision of the gross estimate.** Per-trade sd is large (BTC 241 bps, ETH 248,
SOL 542). With ~1,100 trades the MDE is ≈8 bps (per the earlier power note) — well
below the observed ~13–33 bps gross, so the *gross* mean is measured precisely; it
simply sits far below the 56 bps bar.

**The naive t-test** is reported for transparency only; it agrees in direction and
significance but is not the pre-registered instrument.

**Conclusion (worded carefully):** this strategy **failed under the specified
rules on this validation window**. That is not a claim that every possible
momentum strategy is ineffective, and not a claim that the true edge is exactly
zero — only that this hypothesis, as frozen, did not clear its registered bar.

## 6. Regime caveat (pre-registered, restated)

The 2021-2025 window is a different market history from the later bear→recovery
period. A FAIL here rejects *this* hypothesis under *these* rules on *this*
window; it is not a claim that no momentum variant could ever work, and not a
claim that the strategy would fail identically in every regime. Per the frozen
rule, the hypothesis is rejected and **no retuning on validation** is performed.

## 7. Honest limitations

- **SOL coverage hole** (2023-07-06 → 2024-08-26, 10,013 h) is a genuine Alpaca
  data gap. SOL is reported per symbol with the gap disclosed; it is excluded from
  the primary pooled figure (pre-outcome decision in the manifest). This does not
  change the verdict — BTC and ETH alone fail decisively.
- **`data/` is gitignored**, so the validation cache is not in version control;
  its per-file sha256s are recorded in the manifest for reproducibility.
- Single run, no Monte-Carlo of the harness; the deterministic tests plus the
  independent spec transcription cover the arithmetic.

## 8. Safety boundary (confirmed)

- Single frozen run; **no retuning, no alternative variants** after seeing results.
- **No production code, tests, accounting patch, risk limits, or deployment
  config changed.** Protected files byte-identical to HEAD.
- 0 staged; **nothing committed, deployed, restarted, or ordered.** Live trading
  remains frozen.
- New artifacts only: this report, the manifest, and the gitignored validation
  cache.

## 9. Exact commands used

```bash
# fetch (data-only, into a SEPARATE dir; development cache untouched)
python3 - <<'PY'
import sys, json; sys.path.insert(0,"scripts")
from pathlib import Path
from fetch_ohlcv import fetch_symbol
OUT = Path("data/ohlcv_dev2021_2025"); OUT.mkdir(parents=True, exist_ok=True)
for sym in ("BTC/USD","ETH/USD","SOL/USD"):
    bars = fetch_symbol(sym, "2021-01-01", "2025-09-30", "1Hour")
    (OUT / f"{sym.replace('/','_')}_1Hour.json").write_text(
        json.dumps({"symbol": sym, "timeframe": "1Hour", "bars": bars}))
PY

# the single frozen run
.venv/bin/python scripts/momentum72_study.py \
    --cache data/ohlcv_dev2021_2025 \
    --start 2021-01-01T06:00:00Z --end 2025-09-30T23:00:00Z --pool
```

## 10. Bottom line

**FAIL.** The 72-hour volatility-scaled momentum hypothesis, exactly as frozen,
does not survive independent validation: gross per-trade return (~13–33 bps) is
well under the 56 bps cost bar, the bootstrap p-value is ≈1.0, and net returns are
negative on every symbol and in both poolings. Per the pre-registered rule, the
hypothesis is **rejected**, and no retuning on this validation data is performed.
