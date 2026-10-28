# P1 — PAPER-TRADING EXECUTION COST MEASUREMENT (READ-ONLY)

Date: 2026-10-25. **READ-ONLY.** Authorized scope: measure actual paper-trading
execution costs. No orders, deployments, restarts, commits, or source changes. No
DB repair, no config change.

Base: HEAD `7a8ca12` (branch `main`), 0 staged.

---

## VERDICT

**The available data CANNOT produce a trustworthy measured cost distribution.**

- **Fees: NOT measurable.** Every nonzero commission is **exactly 25.00 bps** —
  which is the value of the code's own estimator (`ESTIMATED_TAKER_FEE_BPS=25`),
  not an exchange-reported fee. The measurement is **circular**: it recovers the
  model's number. Alpaca's Order model surfaces no commission (the code says so
  itself), and the snapshot carries no `commission_estimated` flag column, so the
  two are indistinguishable in the stored data.
- **Spread: NOT measurable.** The snapshot has **no** quote columns
  (`mid_at_submit`, `spread_at_submit`, `quote_ts` do not exist at HEAD — they are
  uncommitted working-tree additions). No bid/ask anywhere, including in the JSON
  blobs (scanned).
- **Slippage: NOT cleanly measurable.** There is no contemporaneous reference
  price. The only reference available is an hourly bar close that is up to
  **60 minutes stale** at fill time, so a fill-vs-bar-close number conflates
  **market drift with slippage**. It is an upper bound on |drift+slip|, not a
  slippage estimate.

**Therefore: the 56 bps modeled round trip cannot be validated or refuted from
this snapshot.** Both of its components are unmeasurable here — the fee leg
because it *is* the model, the slippage leg because the reference is too coarse.

The fix is **instrumentation, not analysis**: the columns needed to measure this
(`mid_at_submit`, `spread_at_submit`, `quote_ts`) already exist in the uncommitted
working tree but were **not present** in the deployed bot that wrote this
snapshot.

---

## 1. Exact database and window examined

| Property | Value |
| --- | --- |
| Database | `/workspace/project/bot_snapshot.db` |
| sha256 (prefix) | `52e39180d5d9e9e1…` — **unchanged** |
| Size | 2,105,344 bytes |
| Integrity | `PRAGMA integrity_check` = **ok** |
| Access mode | `file:…?mode=ro&immutable=1` (read-only, immutable) |
| Orders | **84**, all `status='filled'` |
| Order window | **2026-10-02 03:16:43 → 2026-10-24 04:21:08** |
| Decision snapshots | 24 (window 2026-10-02 03:16:43 → 2026-10-24 03:51:06) |
| Schema at HEAD | `orders` has `commission`; **no** `mid_at_submit`/`spread_at_submit`/`quote_ts` |

The local `data/bot.db` (57,344 bytes) is **empty** (0 orders, 0 snapshots) and was
not used. The snapshot is the only source of real paper records.

**Reference data:** `data/ohlcv/{BTC,ETH,SOL}_USD_1Hour.json` (9,329–9,335 bars,
2025-10-01 → 2026-10-24) — hourly bars covering the trade window. Used **only** as
the staleness-limited slippage reference (§4); it is not a quote source.

## 2. Fees — MEASURED, but circular (not exchange-reported)

| Metric | Value |
| --- | --- |
| Total recorded commission | **$1.4886** |
| Fills with nonzero fee | 42 / 84 |
| Fills with zero fee | 42 / 84 |
| Fee when nonzero | **exactly 25.00 bps** (min = median = max = 25.00) |

By symbol:

| Symbol | Fills | Nonzero fee | Zero fee |
| --- | ---: | ---: | ---: |
| BTC/USD | 30 | **0** | 30 |
| ETH/USD | 27 | 22 | 5 |
| SOL/USD | 27 | 20 | 7 |

**The 25.00 bps is the estimator's own output.** `src/exchange.py:925-952`
(`_fill_estimated_commission`): when the exchange reports no commission, it sets
`commission = filled_avg_price × filled_qty × (ESTIMATED_TAKER_FEE_BPS / 10000)`
and flags `commission_estimated = True`. With `ESTIMATED_TAKER_FEE_BPS = 25`
(`src/config.py:414`), every estimated fee is *by construction* exactly 25.00 bps.
Verified: all 42 nonzero rows reproduce `qty × px × 0.0025` to the cent.

**Why BTC has zero fees.** The fee-vs-zero split is **temporal, not by asset**:

| Period | Fills | With fee | Total fee |
| --- | ---: | ---: | ---: |
| before 2026-10-13 | 42 | **0** | $0.0000 |
| 2026-10-13 onward | 42 | 42 | $1.4886 |

All 30 BTC fills (and the first 12 ETH/SOL fills) predate the estimator's
deployment. So "BTC has no fees" is a **recording gap**, not a fee-free asset —
exactly the trap you warned against. Treating those zeros as zero cost would
understate BTC costs by 25 bps/leg.

**Provenance conclusion:** the stored `commission` field is a **modelled estimate
for every fill that has one**, and **missing for every fill before 2026-10-13**.
There is **no exchange-reported fee** in this snapshot, and no column to
distinguish observed from estimated.

## 3. Spread — NOT MEASURABLE

- `orders` columns: `order_id, decision_id, symbol, side, qty, filled_qty,
  filled_avg_price, commission, status, type, time_in_force, client_order_id,
  submitted_at, filled_at` — **no** spread/quote field.
- Scanned **all** `decision_snapshots` JSON blobs (`feature_snapshot_json`,
  `votes_json`, `tensor_state_json`) for any key containing
  `spread|bid|ask|mid|quote`: **NONE**.
- The capture mechanism exists only in the **uncommitted working tree**
  (`mid_at_submit`, `spread_at_submit`, `quote_ts` on `OrderRecord`) — it was
  **not in the deployed bot** that produced this snapshot.

**Spread cost cannot be measured from this data at all.** Not missing-and-assumed-zero:
genuinely absent. Per your instruction, it is reported as **missing**, never as 0.

## 4. Slippage — measurable only as a stale-reference bound

No contemporaneous quote exists, so I used the **last completed hourly bar close**
before submission (no look-ahead). Signed convention: **adverse positive** (buy
fill above reference, or sell fill below, is positive).

| Symbol | n | median | mean | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| BTC/USD | 30 | +3.65 | +6.98 | +27.91 | +43.41 |
| ETH/USD | 27 | −1.07 | −0.95 | +32.46 | +74.24 |
| SOL/USD | 27 | −13.09 | −10.45 | +39.22 | +98.09 |
| **Pooled** | **84** | **+1.85** | **−1.17** | — | — |

Median **magnitude** |fill − reference| = **12.77 bps**.

**This is NOT a slippage estimate, and I will not present it as one.** The
reference bar closes at the top of the hour; fills occur within the hour, so the
reference is **0–60 minutes stale**. A ±12.77 bps median magnitude is consistent
with ordinary hourly crypto drift, not with execution impact. Two facts sharpen
this:

- **Fills are near-instant:** submit→fill delta is ~**0.56 s** (49 orders; see §6
  for the sign anomaly). Real slippage over 0.56 s should be a fraction of a bps —
  so the observed spread of ±13 bps is **drift, not slippage**.
- The hourly bar cannot resolve sub-second execution cost. **A 1-hour reference is
  the wrong instrument.**

So slippage is **indeterminate**: the measured quantity is dominated by a
confound, and the true value is not recoverable from this data.

## 5. Execution sample quality

| Measure | Value |
| --- | --- |
| Total fills | 84 (100% `filled`; no partials, no rejects, no cancels) |
| Usable for fee analysis | 42 (those with a recorded fee) |
| Usable for slippage (vs bar) | 84 (all have `submitted_at` + price > 0) |
| Usable for **spread** | **0** (no quote data) |
| Usable for **true slippage** | **0** (no contemporaneous reference) |
| Orders with `submitted_at` **and** `filled_at` | 49 / 84 |
| Orders with `filled_at` NULL | **35** |
| Decisions with **both** legs linked by `decision_id` | **1 / 24** |

Coverage by symbol: BTC 30, ETH 27, SOL 27 fills — balanced. But **BTC has zero
recorded fees** (§2) and the **round-trip linkage is essentially absent** (1 of 24
decisions has both a buy and a sell joinable by `decision_id` — the Item 3/4
linkage defect already documented). Cost-per-round-trip therefore **cannot be
aggregated per decision** from this snapshot; only per-leg statistics are possible.

## 6. Data-integrity anomalies found

- **`filled_at` precedes `submitted_at` by ~0.56 s** in all 49 orders that have
  both (median −0.56 s, range −0.58 to −0.55 s). A consistent negative delta is a
  **clock/assignment artifact**, not physics. Any latency or slippage computed
  from these two fields would be wrong-signed. (Flagged, not fixed.)
- **35 orders have NULL `filled_at`** despite `status='filled'`.
- **3 dust fills** with notional < $1 (e.g. $0.000054), each carrying a
  proportionally tiny commission — the same dust partials behind the
  `return_pct` blowups in the earlier audit. They are real rows, not errors.

## 7. Cost distribution vs the 56 bps model

| Component | Modelled | Measured | Verdict |
| --- | --- | --- | --- |
| Fee per leg | 25 bps | 25.00 bps — **but it IS the model** | **Circular; not evidence** |
| Slippage per leg | 3 bps | ±12.77 bps vs stale bar | **Confounded by drift; not evidence** |
| Round trip | **56 bps** | **cannot be computed** | **Unvalidated** |

**No distribution can be honestly reported**, because neither component is
independently measured. The single most important consequence: **we do not know
whether the true round-trip cost is above or below 56 bps.** That matters for the
edge work — Momentum72 failed with gross ~13–33 bps/trade *against* 56 bps; if
real costs were materially lower, the margin changes (though SOL at −1.88 bps net
would still not clear zero).

## 8. Direct answers to your six points

1. **Fees:** recorded commissions are **estimates** (exactly 25.00 bps = the
   model), with **no exchange-reported fees** present and **no flag** to separate
   them. Pre-2026-10-13 fills have **no fee at all** (all 30 BTC fills included).
2. **Spread:** **not available** — no quote columns, no quote keys in any blob.
3. **Slippage:** computable only against a **1-hour-stale** bar close; the result
   is dominated by drift (~13 bps median magnitude vs ~0.56 s execution), so it is
   **not a usable slippage estimate**. Buy/sell direction was applied correctly.
4. **Sample quality:** 84 fills, balanced across symbols; but **0 usable for
   spread**, **0 usable for true slippage**, 42 usable for fee (all circular), and
   only **1 of 24** decisions round-trip-linkable.
5. **Distribution:** **not reportable** — both components are unmeasurable here;
   56 bps is therefore **unvalidated**, not confirmed.
6. **Limitations:** paper fills are recorded, but the snapshot lacks the quote and
   latency instrumentation needed to make any cost estimate meaningful. **Hourly
   bars and a 1-second timestamp pair are not sufficient.**

## 9. Findings

| ID | Severity | Claim | Evidence |
| --- | --- | --- | --- |
| P1-1 | **HIGH** | Stored fees are the code's own estimate, not observed; the snapshot cannot validate the 25 bps input | REPRODUCED: all 42 nonzero = 25.00 bps = `exchange.py:944` formula with `ESTIMATED_TAKER_FEE_BPS=25` |
| P1-2 | **HIGH** | 42/84 fills (incl. all 30 BTC) record **zero** fee — a recording gap, not zero cost | REPRODUCED: pre-2026-10-13 fills = 0 fee; post = 42/42 with fee |
| P1-3 | **MEDIUM** | Spread is entirely unmeasurable — no quote data at HEAD or in blobs | REPRODUCED: `PRAGMA table_info`; blob key scan |
| P1-4 | **MEDIUM** | Slippage cannot be separated from drift with an hourly reference | REPRODUCED: ~0.56 s execution vs ±12.77 bps median vs stale bar |
| P1-5 | **LOW** | `filled_at` precedes `submitted_at` by ~0.56 s (artifact) | REPRODUCED: 49/49 orders |
| P1-6 | **LOW** | Only 1/24 decisions round-trip-linkable by `decision_id` | REPRODUCED (Item 3/4 defect) |

## 10. Recommendation (no implementation authorized)

1. **Do not treat the 56 bps as validated.** It is currently a *modelled
   assumption*; this snapshot neither confirms nor refutes it.
2. **The remedy is instrumentation, and it partly exists.** `mid_at_submit`,
   `spread_at_submit`, and `quote_ts` are already written by the uncommitted
   working tree. Deploying that instrumentation (a **separate, authorized**
   change) would make the next snapshot measurable.
3. **Persist an explicit `commission_estimated` flag** on the order row so
   observed and estimated fees are never conflated again.
4. **Do not back-fill or repair** the historical zeros — they are evidence of the
   gap, not an error to erase.
5. **Until instrumentation lands, no cost claim should be made** in either
   direction (neither "costs are 56 bps" nor "costs are lower").

## 11. Safety boundary

- 0 staged; no source/test/config change; snapshot `52e39180…` byte-identical;
  integrity ok.
- Opened read-only (`mode=ro&immutable=1`); no write, no repair, no back-fill.
- No orders, deployments, restarts, or commits.
- Only new artifact: this report.
