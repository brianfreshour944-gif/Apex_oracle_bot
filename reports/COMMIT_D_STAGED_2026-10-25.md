# COMMIT D — STAGED (INDEX ONLY, NOT COMMITTED)

Date: 2026-10-25. **Authorized action: "Stage D only on
w1-instrumentation-accounting."** Executed. **No commit made.**

---

## Staged contents (6 files, +171 / −8)

```
src/config.py                               | 17 +++++++++++++++++   (D hunks only)
src/feature_drift_monitor.py                |  9 +++++----
src/performance_tracker.py                  |  9 +++++----
tests/conftest.py                           | 12 ++++++++++++
tests/test_feature_drift_isolation.py       | 63 +++++++++++++++++++++
tests/test_performance_tracker_isolation.py | 69 +++++++++++++++++++++
```

## How `src/config.py` was split

Rather than drive interactive `git add -p`, the diff was filtered
deterministically: hunks parsed, the one containing `CAPTURE_SUBMISSION_QUOTE`
excluded, the remaining three written to a patch and applied with
`git apply --cached`. Dry-run (`--check`) passed **before** the index was touched.

| Hunk | Scope | Staged? |
| --- | --- | --- |
| `@@ -1,5 +1,6 @@` `import os` | D | **yes** |
| `@@ -423,6 +424,14 @@` `CAPTURE_SUBMISSION_QUOTE` | A | **excluded** |
| `@@ -685,6 +694,20 @@` the two `*_STATE_PATH` fields | D | **yes** |
| `@@ -948,6 +971,8 @@` the two exports | D | **yes** |

## Contamination check — the required verification

| Marker class | Found in staged diff? |
| --- | --- |
| **A** — `submission_quote`, `mid_at_submit`, `spread_at_submit`, `quote_ts`, `get_latest_quote`, `CAPTURE_SUBMISSION_QUOTE`, `fee_source` | **NONE** |
| **B** — `reconstruct_position_outcome`, `outcome_status`, `_close_open_snapshot`, `ACCOUNTING` | **NONE** |
| **C** — `excursion`, `max_fav`, `max_adv` | **NONE** |

**Unstaged remainder (correctly held back):** `src/bot.py`, `src/config.py`
(the A hunk), `src/db.py`, `src/exchange.py`, `src/risk.py`,
`tests/test_reconciliation_and_db_recovery.py`.

## Index integrity checks

| Check | Result |
| --- | --- |
| Staged `config.py` blob parses as Python | **OK** |
| Staged blob contains the two state-path fields | **yes (4 refs)** |
| Staged blob contains `CAPTURE_SUBMISSION_QUOTE` | **0 — excluded** |
| `git diff --cached --check` (whitespace) | clean |
| D tests against **staged-only** state (`stash --keep-index`) | **8 passed** |

The staged tree was tested in isolation: the working tree was parked with
`git stash push --keep-index`, the D tests run against index-only content, then
the working tree restored with `git stash pop`. This proves the staged changes
stand alone and do not depend on the unstaged A/B/C work.

## Safety state after staging

| Item | Value |
| --- | --- |
| Branch | `w1-instrumentation-accounting` |
| Commits ahead of `main` | **0** |
| `main` SHA | `7a8ca12` — unchanged |
| Staged | **6 files** (as authorized) |
| Unstaged | 6 files (A/B/C — untouched) |
| Stash | empty (restored) |
| tracker / drift / snapshot / backup | `44136fa3…` / `82a78c0d…` / `52e39180…` / `ddb2e987…` — all unchanged |

## Stopped here

Per the authorization: **no commit, no other files staged, no source modified,
no deployment.** Awaiting separate authorization to commit D.

The working tree was temporarily parked to test the staged-only state; it has
been fully restored (verified: 6 staged, 6 unstaged, 11 untracked, stash empty).
