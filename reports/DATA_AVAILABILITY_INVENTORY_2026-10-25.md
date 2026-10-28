# Data-Availability Inventory (read-only) — for the frozen 72h momentum experiment

Date: 2026-10-25
Author: OpenHands. **Read-only**: no strategy run on any proposed validation
period, no file written except this report, no network data downloaded, no git
operation, no DB write.
Base commit: 7a8ca12 (branch main), 0 staged. Code patch sha256 unchanged
(`d1131e48…`). OHLCV caches unmodified (mtimes still 2026-10-25 20:38–20:39).

---

## 1. Inventory of existing OHLCV files and exact timestamp ranges

Only one cache exists: `data/ohlcv/` (gitignored). Three hourly files:

| File | Bars | First bar (UTC) | Last bar (UTC) | Span | Dups | Missing h | Max gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTC_USD_1Hour.json | 9,335 | 2025-10-01T00:00:00Z | 2026-10-24T23:00:00Z | 388.96 d | 0 | 1 | 2 h |
| ETH_USD_1Hour.json | 9,334 | 2025-10-01T00:00:00Z | 2026-10-24T23:00:00Z | 388.96 d | 0 | 2 | 2 h |
| SOL_USD_1Hour.json | 9,329 | 2025-10-01T00:00:00Z | 2026-10-24T23:00:00Z | 388.96 d | 0 | 7 | 3 h |

- Schema: `{"symbol","timeframe","bars":[{t,o,h,l,c,v,n,vw}]}` — Alpaca crypto bars.
- Monotonic, zero duplicates, gaps sparse (≤5 per symbol over a year) and small
  (2–3 h). These are the gaps the EDGE_AUDIT already documented.
- **The entire cache is 2025-10-01 → 2026-10-24.** Nothing earlier, nothing later.

No other market-data source exists in the repo. The only other data files are
`data/historical_experiences.jsonl` (852 rows of *bot feature tensors*, dated
2026-04-07..2026-09-29 — bot records, not OHLCV) and `data/live_experiences.jsonl`
(empty). `data/bot.db` is the (empty local) operational DB. No CSV/parquet. No
OHLCV is tracked in git history (`git ls-files` and `git log -- data/ohlcv` empty).

## 2. Is any data after the previously examined period available?

No. The cache ends **2026-10-24T23:00Z**. A read-only metadata probe of Alpaca
(`limit=1`) confirms the live endpoint's most recent completed hourly bar is
**2026-10-23T00:00Z** (the endpoint's present). So there is **no already-downloaded
data beyond the examined period**, and no *new* future data exists yet — it would
have to accrue in real time.

## 3. Has any additional historical data already been used in development/selection?

No pre-2025-10-01 data has been touched. Evidence:

- The fetcher defaults and the only recorded invocation are
  `scripts/fetch_ohlcv.py --start 2025-10-01 --end 2026-10-24`; the cache
  contains nothing earlier.
- The EDGE_AUDIT states the span as 2025-10-01..2026-10-24 with the split at
  2026-07-01; EDGE_INVESTIGATION used the same file and the same split.
- The frozen spec's `DEV_START/DEV_END = 2025-10-01 .. 2026-10-24` and
  `VAL_START/VAL_END = 2021-01-01 .. 2025-09-30` match: the validation window is
  outside the cache entirely.

**Alpaca availability (read-only probe, `limit=1`, no bars analysed):** the
earliest hourly bar is **2021-01-01T06:00:00Z** for all three symbols; probes for
2020-06-01 and 2020-12-01 returned empty. This confirms the spec's claim that
2021-01-01..2025-09-30 is fetchable and has never been downloaded.

## 4. If no independent validation data exists: earliest future window + sample size

Two distinct situations — they are **not** the same:

- **Historical independent data DOES exist and is fetchable:** 2021-01-01T06:00Z
  → 2025-09-30 (≈4.74 yr), never downloaded or viewed. This is the spec's
  validation period and it is genuinely untouched. (Alpaca cannot serve anything
  earlier than 2021-01-01, so there is no *older* alternative.)
- **Earliest FUTURE (real-time-accruing) window** — only if you decline the
  historical window: there is no stored future data; a fresh period would begin
  at the next unseen bar (2026-10-25 onward) and must accrue in real time.

**Sample size.** Characterising the **already-inspected development period only**
(counts, not validation outcomes):

| Symbol | Bars (1.07 yr) | Entry episodes | Episodes/yr |
| --- | --- | --- | --- |
| BTC/USD | 9,335 | 224 | 210 |
| ETH/USD | 9,334 | 238 | 223 |
| SOL/USD | 9,329 | 243 | 228 |
| **Pooled** | 27,998 | **705** | **221** |

Extrapolating the *rate* (≈221 entry episodes/yr pooled) to the 4.74-yr window:
**≈1,045 pooled episodes ≈ 348 per symbol**, far above the spec's ≥100 floor.
Median hold ≈4 h, mean ≈19 h. So the historical window comfortably clears the
trade-count screen — but see §5.

## 5. Effective sample size and dependence (power caution)

The ≥100-trade threshold is a **screening floor, not a power guarantee**. From the
development period (dispersion only; no validation outcome used):

- Per-trade net-return sd ≈ **60 bps** (BTC 47, ETH 70, SOL 64).
- Minimum detectable effect (one-sided α=0.05, power=0.80, z≈2.485):

| n trades | MDE |
| --- | --- |
| 100 | ≈ **15.0 bps**/trade |
| 348 (per-symbol window) | ≈ **8.0 bps**/trade |
| 1,045 (pooled window) | ≈ **4.6 bps**/trade |

Since the spec's pass bar is **≥56 bps gross**/trade, even 100 trades can detect
an effect that large with power. But the *net* question ("does it clear cost")
lives near 0–56 bps, where 100 trades is underpowered and the pooled window is
only marginally powered for small true edges.

Dependence: hourly returns are near-uncorrelated (|acf₁| ≤ 0.024; n_eff deflation
only ~1.35–1.48×), and consecutive-trade returns are weakly correlated
(lag-1 ≈ −0.11..0.00). Dependence is therefore **mild, not severe** — but it is
non-zero, which is why the spec already pre-registers a moving-block bootstrap
(block=12) rather than a naive t-test. The effective sample size is closer to
n/1.4 than to n.

**Caveat that matters more than power:** the spec's window (2021→2025) spans a
different market regime than development (2025-10→2026-10, a bear→bull cycle).
That is a feature (out-of-regime test) but it means a PASS/FAIL is a statement
about that regime, not a general edge claim.

## 6. Did not do (by instruction)

No strategy was run on 2021-01-01..2025-09-30 or any proposed validation period.
`reports/MOMENTUM72_RESULTS_2026-10-25.md` was **not** created. No bars were
downloaded into the repo (the Alpaca probe requested `limit=1` metadata only).

## 7. Bottom line

- **An untouched, independently fetchable validation period exists:**
  2021-01-01T06:00Z → 2025-09-30 (≈4.74 yr). It is not in the cache and has never
  been viewed. The July–Oct 2026 holdout is **not** usable — it is inside the
  already-inspected development period.
- Expected **≈1,045 pooled entry episodes** (≈348/symbol), well above the ≥100
  floor; MDE ≈4.6 bps/trade pooled.
- **No future-accruing data is stored**; if the historical window is declined, the
  earliest independent window is real-time (2026-10-25 onward) and must accrue.
- Recommendation: the historical window is a legitimate, genuinely independent
  validation set. Whether to run the frozen experiment on it is a decision for
  the operator; this inventory does not run it.

## Reproduce (all read-only)

```bash
# ranges / gaps / dups
python3 - <<'PY'
import json,datetime as dt
for s in ("BTC_USD","ETH_USD","SOL_USD"):
    b=json.load(open(f"data/ohlcv/{s}_1Hour.json"))["bars"]
    print(s,len(b),b[0]["t"],b[-1]["t"])
PY
# Alpaca earliest-available probe (metadata only, limit=1)
# GET https://data.alpaca.markets/v1beta3/crypto/us/bars
#     ?symbols=BTC/USD&timeframe=1Hour&start=2021-01-01T00:00:00Z&end=2021-01-01T23:59:59Z&limit=1
```

No file was modified by this inventory except this report.
