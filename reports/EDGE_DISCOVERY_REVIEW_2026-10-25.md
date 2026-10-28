# EDGE-DISCOVERY REVIEW — Momentum72 + edge-study results (READ-ONLY)

Date: 2026-10-25. **READ-ONLY. No source/test/config change, no DB write, no
tuning, no parameter change, nothing staged.** Authorized: read-only
edge-discovery review. Base: HEAD `7a8ca12` (branch `main`), 0 staged.

Method: I **re-ran** the existing harnesses on the cached data rather than
trusting the reports. Every number below is labelled **REPRODUCED** (I ran it
this session) or **REPORTED** (from a report, not re-derived). Baseline artifacts
verified by hash first (§1).

---

## 1. Artifact integrity — all hashes match the recorded manifests

| Artifact | sha256 | Status |
| --- | --- | --- |
| `scripts/momentum72_study.py` | `24f39a9d…e1c35f752` | **MATCH** (re-frozen instrument) |
| `tests/test_momentum72_study.py` | `5dec4b92…9db19634` | MATCH |
| `reports/MOMENTUM72_SPEC_2026-10-25.md` | `08316c38…a21af854` | MATCH (hypothesis untouched) |
| `data/ohlcv_dev2021_2025/BTC_USD_1Hour.json` | `71a838b7…5ac789a7` | MATCH (41,597 bars) |
| `…/ETH_USD_1Hour.json` | `073fd25a…c4f85ddc` | MATCH (41,596 bars) |
| `…/SOL_USD_1Hour.json` | `b415e525…21e2ddf7` | MATCH (31,521 bars) |
| `data/ohlcv/{BTC,ETH,SOL}` (development) | `00f8a605…`, `476bfe65…`, `f4877008…` | MATCH |

The validation window (2021-01-01..2025-09-30) is a **different period** from the
edge-study development window (2025-10-01..2026-10-24). So the two studies are
genuinely independent, and Momentum72's validation was not contaminated by the
earlier study.

---

## 2. Question 1 — which hypotheses were tested, rejected, or inconclusive?

### 2a. Edge study (`scripts/edge_study.py`) — 7 pre-registered rules
**REPRODUCED** (all 42 in-sample cells re-run this session):

| Hypothesis | Symbols | In-sample | Holdout | Verdict |
| --- | --- | --- | --- | --- |
| H1 momentum 24h / 72h | BTC/ETH/SOL | all negative, p=1.0 | all negative | **REJECTED** |
| H2 mean-reversion 24h / 72h | BTC/ETH/SOL | all negative, p≥0.90 | all negative | **REJECTED** |
| H3 MA-cross 12/48 | BTC/ETH/SOL | all negative, p=1.0 | SOL **+4.62** (p=0.36) | **REJECTED** |
| H3 MA-cross 24/168 | BTC/ETH/SOL | all negative, p=1.0 | SOL **+73.75** (p=0.28) | **REJECTED** |
| H4 breakout 48 | BTC/ETH/SOL | all negative, p=1.0 | all negative | **REJECTED** |

- **REPRODUCED:** the two positive holdout cells — SOL `H3_ma_12_48` mean
  **+4.62 bps**, p=**0.3612**, total −1.93%; SOL `H3_ma_24_168` mean **+73.75
  bps**, p=**0.2834**, total +8.13%. Both underperform SOL buy-and-hold
  (**+38.52%**, reproduced).
- **REPRODUCED:** min in-sample p across all **42** cells = **0.8898**; cells with
  p<0.05 = **0**. Matches the audit exactly.

### 2b. Momentum72 (`scripts/momentum72_study.py`) — 1 frozen hypothesis
**REPRODUCED** — I re-ran the frozen harness on the validation cache:

| Symbol | Trades | Gross bps/trade | Net bps/trade | Net total | Buy&hold | p (boot) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC/USD | 1,133 | **+12.59** | −39.28 | −98.8% | +288.0% | 1.000 |
| ETH/USD | 1,108 | **+12.58** | −35.61 | −98.1% | +456.5% | 1.000 |
| SOL/USD | 844 | **+33.17** | −1.88 | −14.7% | +12,660.1% | 0.583 |
| Pooled (all 3) | 3,085 | +18.21 | −27.73 | — | — | 0.9998 |

Reproduces the report's headline **to the digit**. Decisive failure: gross
per-trade (~13–33 bps) is **below the 56 bps cost bar on every symbol**.
Verdict **FAIL / REJECTED**, correctly.

**Answer to Q1:** Every tested hypothesis is **REJECTED**. None is merely
inconclusive. The 7 edge-study rules fail in-sample *and* out-of-sample; the one
frozen Momentum72 hypothesis fails its pre-registered bar.

---

## 3. Question 2 — does any candidate show credible positive **net** expectancy?

**No. REPRODUCED.**

- **Gross signal is weakly positive but below cost.** Momentum72 gross is +12.6 to
  +33.2 bps/trade against a 56 bps round trip. The edge-study rules are
  **gross-negative** before costs even apply.
- **Net is negative everywhere** in both studies, on every symbol.
- **Information coefficient ≈ 0.** REPRODUCED: edge-study |IC| ≤ **0.011** on the
  cells I checked; Momentum72 IC = 0.0045–0.0074. Momentum72's own report claims
  IC ≤ 0.025 across all cells — consistent.
- **The signal is indistinguishable from random.** REPRODUCED null model: random
  long/flat at comparable exposure yields gross mean **−0.85 / −1.32 / −1.29 bps**
  (sd 0.65–1.11). The hypotheses' gross means sit **inside the null
  distribution**. There is no detectable directional information.

**The one honest nuance:** the reported Spearman ρ ≈ 0 I recalled from a prior
session could **not** be located — there is no `edge_estimator_study.py`,
`condition_study.py`, or any `Spearman`/`4260` reference anywhere in `reports/`,
`scripts/`, or `tests/` (**REPORTED-ABSENT**, verified by search). So that
specific number is **unverifiable in the current tree** and should not be cited as
evidence. It is not needed: the ICs and null model above make the same point with
reproducible evidence.

**Answer to Q2:** No candidate shows credible positive net expectancy. The
weakly-positive *gross* Momentum72 signal does not clear cost; every *net* figure
is negative.

---

## 4. Question 3 — robustness to look-ahead, multiple testing, holdout

**Look-ahead — CLEAN (read + reproduced).**
- `edge_study.py`: every signal function uses `px[i - lookback : i + 1]` — bars
  ≤ *i* only; the position is held over bar *i+1*. Verified in source.
- `momentum72_study.py` `weights()`: `window = lr[i - VOL_LOOKBACK : i]` — returns
  **strictly up to bar *i***; `w[i]` is the target held over *i→i+1*. Verified in
  source.
- **1-bar-lag robustness REPRODUCED:** delaying execution one bar leaves every
  result deeply negative (BTC H1_momentum_24h −59.55 → **−69.94** bps; SOL
  H2_meanrev_72h −52.57 → **−50.28** bps). The conclusion does not depend on the
  same-close fill convention.

**Multiple testing — framed, but immaterial.**
- Edge study = 7 hypotheses × 3 symbols × 2 modes = **42 cells** (REPRODUCED).
  Min p = **0.8898**, zero cells p<0.05 — no correction (Bonferroni 0.00119 or
  BH) could change the conclusion.
- Momentum72 = **1** pre-registered test; no multiple-comparison burden.

**Holdout integrity — sound.**
- Edge study holdout (2026-07-01..2026-10-24) was split by timestamp only, rules
  fixed in code beforehand. The two positive cells are insignificant and
  underperform buy-and-hold.
- Momentum72 used a **genuinely untouched** window (2021-2025) — verified by
  hash-distinct data and the spec frozen *before* evaluation. Run **once**, no
  retuning. **REPORTED + consistent with the artifacts.**
- **One process note (from the reports, not a defect):** the harness had a
  return-timing off-by-one (`w_{i-1}·lr_i` instead of `w_i·lr_{i+1}`) that was
  caught and fixed **before** validation ran, with 6 hand-pinned regression tests
  added. I confirmed the corrected timing in source (`r_gross = tgt * lr[i]`) and
  the 18 harness tests pass. This was handled correctly — the instrument was
  fixed and re-frozen *before* touching validation data.

**Answer to Q3:** No look-ahead defect. The negative result is robust to the fill
convention, needs no multiple-testing correction, and rests on genuinely held-out
data in both studies.

---

## 5. Question 4 — the single most informative next experiment

The two studies agree: **simple price-pattern signals on hourly crypto majors have
no detectable edge net of a 56 bps round trip.** Both failures share a structural
cause worth isolating: Momentum72's gross signal was *weakly positive* (+13–33
bps) but its vol-scaling resized almost every bar (11k–16k exposure changes),
generating 29k–59k bps of cost drag. **Cost, not signal direction, was decisive.**

**Recommended single experiment (pre-registered, one test):**

> **Does a LOW-TURNOVER trend signal have positive net expectancy?**
> Hold exposure constant between entries (no per-bar vol resizing); rebalance
> only on a signal flip or a fixed weekly schedule. Same universe, same 56 bps
> cost, same block bootstrap, same success bar (gross ≥ 56 bps, p<0.05, net >
> buy-and-hold, ≥100 trades) on a **fresh untouched window**.

Rationale: it isolates the one variable that killed Momentum72 (turnover) while
holding signal, cost model, and statistics fixed. If a low-turnover variant still
fails, the honest conclusion is that **this signal family has no edge** and effort
should move to execution economics or a different, explicitly-motivated
hypothesis — not to more committee brains.

**A cheaper, more informative alternative worth considering first:** before any
new backtest, measure the **real** fill costs from the (already-post-fix) paper
account. Both studies assume 56 bps; if real paper costs are materially different,
the cost bar — and every conclusion above — moves. That is a *measurement*, not a
strategy test, and it bounds all future work.

---

## 6. Findings

| ID | Severity | Claim | Evidence |
| --- | --- | --- | --- |
| EDGE-1 | HIGH (scientific) | Tested signal family has no detectable net edge | REPRODUCED: IC ≤ 0.011; null model −0.85…−1.32 bps matches hypotheses; all net negative |
| EDGE-2 | MEDIUM (method) | Both studies underpowered for very small (<10 bps) edges | REPORTED power note (n≈1097 for 10 bps); count ≠ independence |
| EDGE-3 | LOW (process) | Harness had a return-timing off-by-one | REPORTED; fixed + re-frozen *before* validation; source verified |
| EDGE-4 | LOW (evidence) | The recalled Spearman ρ≈0 study is **absent** from the tree | Verified by search — do not cite it as evidence |
| EDGE-5 | LOW (context) | `data/performance_tracker_state.json` holds 48 **synthetic** trades (all pnl=5.0, ret=2.0) | REPRODUCED — not real performance; excluded from any edge claim |

**EDGE-5 note:** this state file (`_STATE_PATH = data/performance_tracker_state.json`,
`performance_tracker.py:21`) contains 48 identical synthetic trades. It is **not
real trading evidence** and must not be read as a performance signal. Flagged
because it sits on the sizing path (`performance_tracker` feeds Kelly sizing).

---

## 7. Unresolved questions

1. **Real fill costs.** Both studies model 56 bps; neither measures actual paper
   fills. The true cost bar is unknown.
2. **Is the weakly-positive Momentum72 gross signal real?** +13–33 bps with
   IC≈0.006 is within noise; not established either way.
3. **Does the absent `edge_estimator_study` exist elsewhere?** It is referenced in
   prior-session memory but not in the tree. If it produced the ρ≈0 result, that
   artifact is **lost** (consistent with the earlier sandbox-wipe).
4. **SOL coverage hole** (10,013 h, 2023-07→2024-08) — genuine Alpaca gap;
   SOL excluded from the primary pooled figure pre-outcome. Handled correctly.

---

## 8. EDGE SUMMARY

```
Strategy/scope: 7 pre-registered hourly rules (edge_study.py) + 1 frozen
                volatility-scaled 72h momentum (momentum72_study.py)
In-sample result:   all 42 edge-study cells negative; min p = 0.8898 (REPRODUCED)
Out-of-sample result: Momentum72 FAIL (gross 12.6-33.2 bps < 56 bps cost; net
                negative all symbols) (REPRODUCED); edge-study holdout: 2 positive
                cells, both p>0.28 and below buy-and-hold (REPRODUCED)
Effective trials: 42 (edge study) + 1 (Momentum72)
Look-ahead findings: NONE
Overfitting findings: NONE (pre-registered; holdout untouched; harness fixed
                before validation)
Commands executed: sha256sum; momentum72_study.py --pool; edge_study.py
                (in-sample + --holdout); pytest (49 passed); null-model and
                lag-sensitivity recomputation
Not covered: real fill costs; the absent Spearman/condition study; SOL gap region
EDGE VERDICT: INCONCLUSIVE (no edge demonstrated)
```

**Why INCONCLUSIVE rather than LIKELY_FITTED:** there is no overfitting — the
hypotheses were pre-registered, the holdout was untouched, and nothing was tuned.
There is also no *support*: no cell clears cost with significance. The honest
reading is **"no edge demonstrated,"** which is exactly what both studies
concluded. A profitable edge remains **unproven**.

---

## 9. Recommendation

1. **Do not loosen gates, add committee brains, or add strategy complexity** on
   this evidence. More votes on a signal with IC≈0 produce a confident zero.
2. **Run the low-turnover experiment** (§5) — one pre-registered test isolating
   turnover, on a fresh untouched window.
3. **Or, preferably first: measure real paper fill costs.** It is cheaper, bounds
   every conclusion above, and is a measurement rather than a strategy test.
4. **Do not cite the absent Spearman study** as evidence (EDGE-4).
5. **Treat `performance_tracker_state.json` as synthetic** (EDGE-5).
6. Live trading stays frozen; no implementation, deployment, or retuning is
   authorized or performed.

## 10. Safety boundary

- 0 staged; no source/test/config change; snapshot `52e39180…` byte-identical.
- No tuning, no parameter change, no alternative variants run.
- No production code, accounting patch, risk limit, or deployment config touched.
- Nothing committed, deployed, restarted, or ordered.
- Only new artifact: this report.
