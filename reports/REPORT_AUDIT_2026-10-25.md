# Independent Audit of Three Reports — read-only

Date: 2026-10-25
Auditor: OpenHands (read-only). No tracked file was modified, staged, or committed.
Base commit: 7a8ca12 (branch `main`), **0 staged**. The uncommitted accounting/
lifecycle patch is intact (4 tracked files modified, unchanged by this audit).

Artifacts audited:
1. `reports/DEPLOYED_TRADE_DATA_AUDIT_2026-10-24.md` (real money — highest stakes)
2. `reports/EDGE_INVESTIGATION_2026-10-25.md`
3. `reports/EDGE_AUDIT_2026-10-25.md`

Supporting: `reports/LIFECYCLE_FIX_DESIGN_2026-10-25.md`,
`reports/UNCOMMITTED_PATCH_2026-10-25.md`, the live snapshot
`/workspace/project/bot_snapshot.db`, and the source at HEAD.

Method: every claim re-derived from the snapshot or HEAD with the commands shown.
Labels: **VERIFIED** (I reproduced it), **PARTLY VERIFIED** (true in substance,
imprecise in a detail), **UNSUPPORTED** (cannot reproduce), **WRONG** (contradicted).

---

## Verdict up front

The **substantive findings in all three reports are correct**. The accounting
defects (dust-leg `return_pct`, structurally-zero MFE/MAE, notional
`shadow_trades`, order-linkage orphans, unverifiable BTC commission) are real and
reproduce exactly. The edge conclusion (no detectable edge; weak signal, not
cost) reproduces exactly. The lifecycle fix does what it claims and is correctly
fail-closed.

The weaknesses are **documentation defects, not analysis defects** — but two of
them matter because they could mislead a decision. The most serious is an
**internal contradiction in the real-money report**: §5/§8 say "12 of 24
unreconciled" while §10 (computed later) says "11 reconciled / 13 unreconciled".
I verified the correct figure is **11/13**. The other is a **drawdown figure
(−282.04%) that is dimensionally impossible** and should never be quoted as a
risk metric.

| Report | Substance | Documentation |
| --- | --- | --- |
| DEPLOYED_TRADE_DATA_AUDIT | **CORRECT** | **has 2 material defects** (12-vs-13; −282% DD) + 3 minor |
| EDGE_INVESTIGATION | **CORRECT** | clean |
| EDGE_AUDIT | **CORRECT** | clean (1 minor: test count is a snapshot) |

---

## 1. DEPLOYED_TRADE_DATA_AUDIT_2026-10-24.md

### 1a. Identity and headline figures — VERIFIED

- Snapshot sha256 `52e39180…cfca1b`, 2,105,344 bytes — **matches** the report.
- `PRAGMA integrity_check` / `quick_check` = `ok`.
- Table counts: `decision_snapshots` 24, `orders` 84, `shadow_trades` 8180,
  `experiments` 6 — **all match**.
- Closed 24, 8 win / 16 lose, net **−$2.1053**, gross profit $1.4213, gross loss
  −$3.5266, win rate 33.3%, PF **0.403** — **all match to the cent**.
- Per symbol: BTC 10/−$0.1812, ETH 6/−$2.2081, SOL 8/+$0.2841 — **match**.
- Per exit reason and per regime — **match** (`regime` is the column name, not
  `market_regime`; the report used the right values).
- `scripts/analyze_deployed_trades.py` re-run: every printed figure reproduces.

### 1b. §2 dust-leg `return_pct` defect — VERIFIED

- ETH `f110556591…`: pnl −$0.037459, `return_pct` −804.76%. Ledger: buy
  0.006049 @ 2479.85, then four sells ending in a `1.877e-06` dust leg.
  `sell_qty = 0.00603388 = buy_qty × 0.997500` **exactly**.
- SOL `35e92c4ad1…`: pnl −$0.037615, `return_pct` −70218.31%. Ledger: buy
  0.140999 @ 106.71, five sells ending in a `5.02e-07` dust leg; ratio 0.997500.
- Root cause confirmed at HEAD `src/bot.py:212`:
  `return_pct = realized_pnl / (entry_price * qty) * 100` — numerator is the full
  round-trip P&L, denominator is one leg's qty. Using the snapshot's own position
  qty gives −0.1242% / −0.1258%, matching the report's §2.

### 1c. §3 MFE/MAE structurally zero — VERIFIED

- All 24 rows: `max_favorable_pct = 0.0`, `max_adverse_pct = 0.0` (24/24, 24/24).
- Cause confirmed at HEAD: the recorder reads `_trailing_peaks`/`_trailing_troughs`
  (`bot.py:224-238`) but the standard close path **pops them first**
  (`bot.py:2855`, `bot.py:2857`) before `_record_committee_outcome` (`bot.py:2860`).
  Only the trailing-stop path (`bot.py:~1774`) records before popping, and no
  snapshot trade used it. Exactly as reported.

### 1d. §4 `shadow_trades` notional — VERIFIED

- 8,180 rows, distinct `qty` = `[1.0]`, aggregate `realized_pnl` −48,521,700.31.
- By candidate: Bot_A_Light 1,375/−33,576,621; Bot_B_Standard 2,906/−18,519,787;
  Bot_C_Heavy 3,899/+3,574,708 — **all match**. Correctly excluded from §1.

### 1e. §5 order↔decision linkage — VERIFIED, but see 1i

- 84 filled orders; 11 null/empty `decision_id`; 18 orphans not in
  `decision_snapshots`; 42 distinct linked decision_ids — **all match**.
- Every one of the 24 closed snapshots links to ≥1 order — **verified**.

### 1f. §6 commission provenance — VERIFIED

- `orders.commission` total **$1.4886**; 42 rows nonzero.
  BTC 30 orders/**0 nonzero**/$0.00; ETH 27/22/$0.8167; SOL 27/20/$0.6719 —
  **match**. The report correctly declines to certify BTC P&L as fee-inclusive.

### 1g. §10 reconstruction — VERIFIED (11 reconciled / 13 unreconciled, agg −$2.9290)

Replicating the guard-aware classification over the ledger:
`reconciled=11, no_linked_exit_fills=11, quantity_mismatch=1,
multiple_position_lifecycles=1` → **11/13**, aggregate **−$2.9290** — matches §10
exactly. Reproduced-to-the-cent **9** rows; changed **2** material rows
(ETH `f110556591…` −0.0375→+0.1259; SOL `35e92c4ad1…` −0.0376→−0.2163); the third
(`c6a6d0423cc2`) is correctly `unreconciled` (two lifecycles). The 3.93× outlier
(`1fd274986506`) is the `quantity_mismatch`; it is also the lone "bull" regime
trade — the report's own §10 note is correct and important.

### 1h. §13 scope / protected paths — VERIFIED

`git diff --stat` = 4 files (bot.py, db.py, risk.py, one test), **0 staged**.
`src/exchange.py`, `strategies.py`, `src/committee/committee.py` are **byte-identical
to HEAD**. `src/risk.py`'s change replaces an inline P&L formula in
`_close_open_snapshot` with the shared reconstruction — it is an accounting change,
not a risk-limit change (the method's guards are untouched).

### 1i. Documentation defects found (the part to act on)

| # | Severity | Defect | Evidence |
| --- | --- | --- | --- |
| D1 | **MEDIUM** | **Internal contradiction: "12 of 24 unreconciled" vs §10 "11 reconciled / 13 unreconciled".** The correct number is **13 unreconciled / 11 reconciled**. | §5 line 152, §7 table, §8 item 5 say 12; §10 lines 271/276 say 13. My replication: 13. |
| D2 | **MEDIUM** | **Max drawdown −282.04% is dimensionally impossible.** It is computed on a *cumulative dollar sum* (running += pnl) with peak 1.1565, trough −2.1053 → the formula yields −282%, which no equity curve can produce. Should not be quoted as a risk figure. | analyzer `_max_drawdown_pct` over `equity=[running sum]`; reproduced: peak 1.1565, trough −2.1053. |
| D3 | LOW | §10 cites `reports/PROPOSED_LIFECYCLE_FIX.diff` — **file does not exist**. Same stale reference in LIFECYCLE_FIX_DESIGN and UNCOMMITTED_PATCH. | `ls reports/PROPOSED_LIFECYCLE_FIX.diff` → No such file. |
| D4 | LOW | §10 says the test file has "22 cases"; it has **35**. §10 also names a test `test_additive_migration_against_legacy_schema`; the actual name is `test_additive_migration_adds_outcome_status_to_legacy_db`. | `grep -c "def test_"` = 35; name mismatch. |
| D5 | LOW | §9 "pytest → 527 passed" is a point-in-time snapshot; the same file's §10 implies 542. Not a contradiction (tests were added), but the report never states the counts are time-ordered. Current suite: **564**. | grep of reports; live run. |

**D1 and D2 are the ones worth fixing.** D1 changes a stated number about real
money. D2 is a metric that is numerically impossible and could be quoted in a
risk discussion.

---

## 2. EDGE_INVESTIGATION_2026-10-25.md — VERIFIED

Re-ran `scripts/edge_study.py` (in-sample and `--holdout`):

- In-sample representative cells reproduce (BTC H1_24h −59.55 bps; H3_ma_24_168
  −169.5; SOL H2_meanrev −52.57 → etc.).
- Holdout table reproduces; the two positive cells reproduce **exactly**:
  SOL H3_ma_12_48 = +4.62 bps, p = 0.3612, total −1.93%;
  SOL H3_ma_24_168 = +73.75 bps, p = 0.2834, total +8.13%.
- Buy-and-hold SOL holdout = **+38.52%**, matching the report's +38.5%.
- Cost model 56 bps (25 fee + 3 slip/side) confirmed in code and applied to
  `buy_hold` as well.

The report's honest framing (positive cells are insignificant AND underperform
buy-and-hold; "no edge" ≠ "no edge ever") is accurate. **No defects found.**

---

## 3. EDGE_AUDIT_2026-10-25.md — VERIFIED

- **Data integrity:** 0 duplicates, monotonic, valid OHLC, exact split boundary
  (first holdout bar 2026-07-01T00:00:00Z) — **verified**.
- **IC ≤ 0.025:** I recomputed the max |IC| in-sample = **0.0251** (the audit
  rounds to 0.025; true). Every hypothesis on every symbol is at or below that.
- **Min in-sample p = 0.8898:** reproduced **exactly**; 0 of 42 cells p < 0.05.
- **Null model:** random long/flat at matched exposure → gross −1.65 / −2.49 /
  −2.47 bps (sd 1.1–2.0) — **matches**; the rules sit inside the null.
- **1-bar lag:** all hypotheses remain deeply negative — **verified**.
- **Exposure:** H4_breakout_48 ≈ 0.056–0.059 — **matches** the "≈0.06" claim.
- **Diagnosis (weak signal, not cost):** endorsed. Gross means are ≈0/negative
  before any cost; ICs are noise.

One minor imprecision: §"Verified this session" states `pytest tests/` = 552.
That was true at the time; the suite is now **564** because tests were added
after. Time-ordered, not wrong. **No material defects.**

---

## 4. LIFECYCLE_FIX_DESIGN_2026-10-25.md — VERIFIED

- The bug is real and I reproduced it: for `buy 1.0, sell 0.5, buy 0.5, sell
  0.9975`, `_count_position_lifecycles` returns **1** and `_has_ambiguous_reentry`
  returns **True** → without the guard the row is certified `reconciled` (the
  false certification), with the guard it is `unreconciled`.
- Reachable at the deployed 25 bps fee: `buy 1.0, sell 0.994, buy 1.0, sell
  1.0035` is flagged `ambiguous_lifecycle_history` (residual 0.6% > 0.5% tol).
- No scale-in false positive: `buy 1.0, buy 0.5, sell 1.49625` → not flagged.
- "0 of 42 decisions flagged on the deployed snapshot" — **verified** (0 flagged).
- All 5 named regression tests exist with the exact names claimed.
- **Defect:** the referenced `reports/PROPOSED_LIFECYCLE_FIX.diff` does not exist
  (same as D3).

---

## 5. Grades

| Claim area | Grade |
| --- | --- |
| Snapshot identity, headline P&L, per-symbol/exit/regime | **VERIFIED** |
| §2 dust-leg `return_pct` defect | **VERIFIED** |
| §3 MFE/MAE structurally zero | **VERIFIED** |
| §4 shadow_trades notional | **VERIFIED** |
| §5 order linkage (11 null, 18 orphan) | **VERIFIED** |
| §6 commission provenance | **VERIFIED** |
| §10 reconstruction 11/13, agg −$2.9290, 9 reproduced | **VERIFIED** |
| §13 protected paths untouched, 0 staged | **VERIFIED** |
| "12 of 24 unreconciled" (§5/§7/§8) | **WRONG** — correct is 13 |
| Max drawdown −282.04% | **WRONG** (impossible as an equity drawdown) |
| §10 diff-file reference | **UNSUPPORTED** — file absent |
| §10 "22 cases" / test name | **PARTLY VERIFIED** (35 cases; name differs) |
| Edge investigation tables + holdout cells | **VERIFIED** |
| Edge audit IC / min-p / null model / exposure | **VERIFIED** |
| Lifecycle bug reachability + no false positive | **VERIFIED** |

---

## 6. Risk assessment

- **No live-trading risk was introduced.** The patch is uncommitted, 0 staged,
  and the audit changed no tracked file. Protected paths
  (`exchange.py`, `strategies.py`, `committee/`) are byte-identical to HEAD.
- **Residual risk the reports themselves flag, correctly:** the adaptive learner
  and the rolling loss-limit gate (`get_recent_realized_pnl`) still read the
  *stored* `realized_pnl` regardless of `outcome_status`, so they can consume the
  legacy corrupted rows. The reports call this out and deliberately leave it as a
  separate, approval-required change. I agree with that scoping.
- **Documentation risk:** D1 (wrong unreconciled count) and D2 (impossible
  drawdown) could be quoted downstream. Fix before either number is reused.

## 7. Internal contradictions found

1. **12 vs 13 unreconciled** in the real-money report (D1) — the highest-value
   fix; a reader of §5/§8 gets a different number than a reader of §10.
2. **−282.04% drawdown** reported alongside a long-only, ~$100-notional account
   (D2) — mathematically impossible as an equity figure.
3. Test-count snapshots (527 → 542 → 552 → 564) presented without a date qualifier
   (D5) — benign but easy to misread as disagreement.

## 8. Recommended next action

1. **Fix D1**: change "12 of 24" to "13 of 24" in §5, §7, §8 of
   DEPLOYED_TRADE_DATA_AUDIT (the §10 figure is correct).
2. **Fix D2**: either drop the max-drawdown line or relabel it as
   "cumulative dollar peak-to-trough on a sum-of-P&L curve (not an equity
   drawdown; not comparable to a % drawdown)".
3. **Fix D3/D4**: remove or create `PROPOSED_LIFECYCLE_FIX.diff`; correct "22
   cases" → 35 and the migration test name.
4. **Do not** treat the corrected reconstruction numbers as certifying the bot's
   P&L: the reports are right that `orders.commission` has no provenance and BTC's
   is all-zero, so every figure remains "reconstruction-with-recorded-fees".
5. **Keep live trading frozen** and the patch uncommitted, as the reports require.

## Reproduce

```bash
# snapshot identity + headline
sha256sum /workspace/project/bot_snapshot.db
python3 -c "import sqlite3;c=sqlite3.connect('file:/workspace/project/bot_snapshot.db?mode=ro&immutable=1',uri=True);print(c.execute('PRAGMA integrity_check').fetchone())"
.venv/bin/python scripts/analyze_deployed_trades.py /workspace/project/bot_snapshot.db --json /tmp/out.json
# edge study
.venv/bin/python scripts/edge_study.py --holdout --holdout-start 2026-07-01T00:00:00Z
# suite + lint
.venv/bin/python -m pytest tests/ -q      # 564 passed
uvx ruff check .                          # All checks passed!
```

No file was modified by this audit. All findings above were produced read-only.
