# TRACKER RESET — EXECUTED WITH VERIFIED BACKUP

Date: 2026-10-25. Authorized scope: **reset the tracker with a verified backup**
— nothing else. No feature-drift change, no W1 deployment, no commit. Base
`7a8ca12`, 0 staged.

---

## What was done

The contaminated performance-tracker state was reset to `{}`, after a
hash-verified backup was taken **outside `data/`**.

| Step | Action | Result |
| --- | --- | --- |
| 1 | Record original hash | `ddb2e98765a566b2b3e63dd0608ba781c8aba849ad221115165f4d03c8b1fd24` |
| 2 | Back up outside `data/` | `reports/backup/performance_tracker_state.ddb2e987….json` |
| 3 | Verify backup == original | **MATCH** (full 64-char sha256) |
| 4 | Reset tracker to `{}` | file now `{}` |
| 5 | Verify behaviour | all 5 checks pass (below) |
| 6 | Final safety check | snapshot untouched, 0 staged |

**Only one file was modified:** `data/performance_tracker_state.json`.

## Verification

**Independent hash sanity check.** The reset file's hash is
`44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`, which is
exactly `sha256("{}")` — confirming the write is an empty object and nothing
else. The file content is literally `{}`.

**Consumer behaviour on the reset state:**

| Check | Before reset | After reset |
| --- | --- | --- |
| `get_decay_alerts()` | 1 spurious `sharpe_decay` | **`[]`** |
| `get_performance_report()["pairs"]` | `{trend_following\|bull: …}` | **`{}`** |
| `get_regime_win_payoff("bull")` | `None` (guard) | **`None`** (correct "no data") |
| `/metrics` payload | bogus pair shown | **`{"pairs": {}, "alerts": []}`** |
| `log_weekly_performance()` | — | "Insufficient data for performance report." — **no crash** |

**Reads do not re-write.** The hash was identical before and after all read
calls — the reset is stable, not transient.

**Full suite does not repopulate it.** The decisive check: across a full
`pytest tests/` run (596 passed), the tracker hash stayed
`44136fa3…` (`{}`) — **unchanged**. This confirms the P2 isolation fix holds, so
the reset is durable rather than being immediately overwritten by the next test
run.

**Backup is genuinely restorable.** Restoring the backup to an isolated copy
returned **48 records** and the `sharpe_decay` alert reappeared — proving the
backup is a true pre-reset snapshot, not an empty or corrupt file.

## Safety state

| Item | Value |
| --- | --- |
| Tracker | `44136fa3…` = `{}` |
| Snapshot `bot_snapshot.db` | `52e39180…` — **untouched** |
| Backup | `ddb2e987…` — preserved, verified, restorable |
| `data/feature_drift_state.json` | `82a78c0d…` — **NOT touched by this task** |
| Staged | **0** |
| `/tmp` leftovers | 0 |
| Orders / deployment / commit | **none** |

## Consequence — stated plainly

The decay monitor now reports **"insufficient data"** until ~10 genuine closed
trades accumulate (`MIN_TRADES_FOR_STATS=10`). Monitoring coverage is
**temporarily reduced** — blind, but honest. An alert that cannot evaluate
performance is preferable to one reporting fixture-driven performance. The
spurious `sharpe_decay` warning is gone.

## Scope boundary

This reset affects the **workspace copy only**. The deployed instance is a
separate host I cannot reach; its tracker state is not changed by this action.

Not done here (each remains separately unauthorized):

- `feature_drift_state.json` test isolation (same defect class as P2).
- W1 deployment for paper-only measurement.
- P3 research.

## Restoration procedure (if ever needed)

```
cp reports/backup/performance_tracker_state.ddb2e98765a566b2b3e63dd0608ba781c8aba849ad221115165f4d03c8b1fd24.json \
   data/performance_tracker_state.json
sha256sum data/performance_tracker_state.json   # expect ddb2e987…
```

Note: this restores the **synthetic** records. It is a rollback of the reset, not
a recovery of real data — there is no real data to recover.
