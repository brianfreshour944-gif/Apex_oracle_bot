# READ-ONLY TRACKER RECOVERY ASSESSMENT

Date: 2026-10-25. Authorized scope: **read-only assessment only**. No mutation,
no deletion, no deployment, no commit. Base `7a8ca12`, 0 staged.

Question under study: should the contaminated tracker state be **rebuilt** from
trustworthy records, or **cleanly reset**? This report answers the four
pre-decision questions and makes a recommendation. **It makes no change.**

---

## Q1 — Does a trustworthy source of genuine tracker outcomes exist?

**No. There is no recoverable source of genuine outcomes. A rebuild is not
possible from trustworthy data.**

The tracker's only writer is `strategy_selector.record_strategy_outcome()`
(`src/strategy_selector.py:226`), called from the live closed-trade path
(`src/bot.py:433`) with `realized_pnl` and `return_pct`. The only candidate
source for a rebuild is therefore `decision_snapshots` in the DB. Three
independent problems rule it out:

**1. The tracker has never recorded a real trade.** Every one of the 48 entries
postdates the last real DB trade:

| Source | Range | Count |
| --- | --- | --- |
| Tracker entries | 2026-10-24T22:45 → 2026-10-26T02:25 | 48 |
| Real DB trades (`closed_at`) | 2026-10-02T03:19 → 2026-10-24T04:21 | 24 |

`min(tracker) > max(DB)` is **True**. So the 48 entries cannot be real outcomes —
they were written after the bot's last real trade, by test fixtures.

**2. The rebuild source does not even overlap the contaminated group.** The DB
would produce these `(strategy, regime)` groups:

| (strategy, regime) | trades |
| --- | --- |
| (momentum, low_volatility) | 10 |
| (None, sideways) | 3 |
| (None, bull) | 1 |
| (mean_reversion, sideways) | 8 |
| (None, neutral) | 2 |

The contaminated group is `trend_following|bull` — a pair the DB **never
produces**. A rebuild would not replace the bad group; it would add unrelated
groups alongside it.

**3. The DB's `return_pct` carries the known dust-exit defects.** The DB itself
has 2/24 implausible values (`ETH −804.76`, `SOL −70218.31`), the original audit
defect. `return_pct` is also a **percentage** (stored `−0.127769` = −0.128%),
while the tracker divides by 100 to get a fraction — units are consistent, but
the two blowups would corrupt any rebuild that used them.

Also, 6/24 DB rows lack `selected_strategy` entirely (it lives in
`feature_snapshot_json`, absent on those rows).

**Conclusion:** there is no independent trustworthy source. Any rebuild would
either import the DB's defects or require the same kind of reconstruction the
audit already found unreliable.

## Q2 — Which consumers and alerts are affected?

The contaminated state has **one** live effect and two that are structurally
neutralized:

| Consumer | Reads | Effect of contamination |
| --- | --- | --- |
| `alerting.py:497` → `alert_performance_decay()` | `get_decay_alerts()` | **ACTIVE — spurious `sharpe_decay` alert** |
| `risk.py:1561` → Kelly cap | `get_regime_win_payoff()` | **None — returns `None`** |
| `performance_server.py:13` | `get_performance_report()` | `/metrics` shows a bogus `trend_following\|bull` pair |

**The alert (real, but benign).** Computed on the actual file:

| Alert | Fires? | Reason |
| --- | --- | --- |
| `win_rate_decay` | No | fixture is 100% winners → win_rate 1.0 ≥ 0.48 |
| `profit_factor_decay` | No | pf = inf ≥ 1.2 |
| `sharpe_decay` | **Yes** | zero variance → sharpe 0.0 < 0.3 |

So one spurious WARNING fires every slow-check interval. It is **alert-only** —
the code comment states plainly it "does NOT pause trading or resize positions."

**The Kelly cap (structurally disabled).** `get_regime_win_payoff("bull")`
returns **`None`** — verified by calling it. Why: the fixture's zero variance
makes `payoff_ratio = inf`, and the guard `if payoff_ratio <= 0 or not
np.isfinite(payoff_ratio): return None` rejects it. So Kelly gets nothing.
Independently, `KELLY_SIZING_ENABLED` defaults to **False** and is unset in
`.env`. Two layers, both off.

**`/metrics`** would report the fake pair to any dashboard scraping it.

## Q3 — Backup and verify without touching the original

A safe procedure exists and is non-destructive. **Not executed** (assessment only):

1. `sha256sum` the original; record it.
2. Copy to a timestamped backup **outside** `data/` (e.g.
   `reports/backup/performance_tracker_state.<sha>.json`).
3. Verify the backup's hash matches the original's.
4. Perform any reset/rebuild **on a copy**, not in place.
5. Diff the copy's computed reports against expectations before promoting.
6. Only then replace, and re-verify the hash.

Note: the file is **gitignored** (not tracked), so git offers no recovery —
the manual backup is the only safety net.

## Q4 — How to confirm recovered state behaves correctly

**Reset behaviour verified on an isolated copy** (original untouched):

| Check | Result |
| --- | --- |
| `get_performance_report()["alerts"]` | `[]` — no spurious alert |
| `get_performance_report()["pairs"]` | `{}` |
| `get_regime_win_payoff("bull")` | `None` (correct "not enough data") |
| `log_weekly_report()` | logs "Insufficient data for performance report." — no crash |

An empty state is handled cleanly on every path. Acceptance criteria for a
recovery: alerts empty, `/metrics` shows no pair, Kelly still `None`, weekly
report does not crash, and the file is valid JSON loading without the
`Failed to load` warning.

## Recommendation — clean reset, not rebuild

**Rebuild is not viable** (Q1: no trustworthy source; the source doesn't even
overlap). A "rebuild" would fabricate structure from defective data.

**Recommend a clean reset** to `{}`, with three conditions:

1. **Preserve a hash-verified backup first** (Q3) — the file is untracked, so
   there is no other recovery.
2. **Do not attempt to reconstruct** the lost history. The tracker's own
   `MIN_TRADES_FOR_STATS=10` means it correctly reports "insufficient data" until
   10 real outcomes accumulate — the safe default.
3. **Accept the cost:** decay monitoring is blind until ~10 real closed trades.
   That is strictly better than monitoring driven by fixtures.

This matches your stated fallback: "a clean reset may ultimately be safer than
pretending the existing state represents real trading performance."

## Incidental finding — a second instance of the same defect

While assessing, I found `src/feature_drift_monitor.py:20` has the **identical
hardcoded relative path** (`_STATE_PATH = "data/feature_drift_state.json"`), is
**not** isolated in `conftest.py`, and **is** written by the test suite:
`strategies.py:470` → `record_features_for_drift` → `_save_state()`.

**Verified:** the file's mtime advanced across a full-suite run (04:38 → 04:45).
Its content stayed empty, so no corruption is *currently* visible — but it is the
same latent class as P2 and would leak the same way once tests record features.

This is **out of scope** for the tracker recovery (different file) and was **not
fixed**. Reported for separate authorization, alongside the P2 fix already done.

## Limitations

- Consumers were identified by static search + targeted runtime calls; I did not
  run the live bot.
- "All 48 entries are synthetic" rests on the fixture signature and the timestamp
  bound, not on a recorded provenance for each entry.
- Whether the deployment even runs the alert slow-check depends on runtime
  config I cannot observe read-only.

## Safety state

| Item | Value |
| --- | --- |
| Staged | **0** |
| Tracker | `ddb2e98765a566b2` — 48 records, **unchanged** |
| Snapshot | `52e39180d5d9e9e1` — unchanged |
| Drift state | rewritten by the suite (empty content); noted, not fixed |
| Deletion / deployment / commit | **none** |

## Next (each needs its own authorization)

1. Authorize the reset (with backup) — or choose otherwise.
2. Decide the `feature_drift_state.json` isolation (same class as P2).
3. Then W1 deployment for paper-only measurement.
4. Accumulate real fills; assess execution costs.
5. Only then W2/P3.
