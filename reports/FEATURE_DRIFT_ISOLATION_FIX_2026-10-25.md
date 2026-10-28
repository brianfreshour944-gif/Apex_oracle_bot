# FEATURE-DRIFT TEST ISOLATION — FIX EXECUTED

Date: 2026-10-25. Authorized scope: **fix the `feature_drift_state.json` test
isolation** — nothing else. No tracker change, no W1 deployment, no commit. Base
`7a8ca12`, 0 staged.

---

## Defect

`src/feature_drift_monitor.py` hardcoded a **relative** default path:

```python
_STATE_PATH = os.path.join("data", "feature_drift_state.json")
```

`tests/conftest.py` isolated `DATABASE_URL`, `ADAPTIVE_STATE_PATH`, and
`PERFORMANCE_TRACKER_STATE_PATH` — but **not** the drift state. So a full-suite
run wrote the **real** runtime file via:

```
strategies.py:470  ->  record_features_for_drift()  ->  FeatureDriftMonitor._save_state()
```

**Verified before the fix:** the file's mtime advanced across a full-suite run
(04:38 → 04:45). Same defect class as the performance-tracker leak fixed in P2.

## Fix — mirrors the P2 precedent

| File | Change |
| --- | --- |
| `src/config.py` | Added `FEATURE_DRIFT_STATE_PATH` field (default `data/feature_drift_state.json`) + module-level export |
| `src/feature_drift_monitor.py` | Imports `settings`; `__init__` resolves `state_path or settings.FEATURE_DRIFT_STATE_PATH` at **construction** time; removed dead `_STATE_PATH` |
| `tests/conftest.py` | Redirects `FEATURE_DRIFT_STATE_PATH` into the existing `_TEST_STATE_DIR` temp dir |
| `tests/test_feature_drift_isolation.py` | New regression test (4 cases) |

Resolution moved from import time to construction time so an env-var override is
always honoured — the same reasoning as P2.

## Verification

**Decisive check — the real file is now untouched by the suite:**

| | Before suite | After suite |
| --- | --- | --- |
| Drift hash | `82a78c0d…` | `82a78c0d…` — **identical** |
| Drift mtime | `04:48:57.711` | `04:48:57.711` — **identical** |

Previously the mtime advanced on every run. It no longer does.

**Suite result:** `600 passed` (596 + 4 new), 5 warnings, ~26s.

**Regression test** (`tests/test_feature_drift_isolation.py`, 4 passed):

1. Default path comes from `settings`, not a hardcoded constant.
2. An explicit `state_path` is still honoured (signature stays compatible).
3. The suite path is redirected out of the repo.
4. `record_features_for_drift` writes the redirected path and leaves the real
   file byte-identical — and after that call the real file's hash **and mtime
   were unchanged**.

**Ruff:** `All checks passed!`

**Production default unchanged:** with no env override,
`settings.FEATURE_DRIFT_STATE_PATH == "data/feature_drift_state.json"` — the
deployed bot reads exactly the same path as before.

**No circular import:** `src.config` does not import the drift monitor; import
succeeds.

## Safety state

| Item | Value |
| --- | --- |
| Drift state `data/feature_drift_state.json` | `82a78c0d…` — **unchanged** (hash + mtime) |
| Tracker | `44136fa3…` = `{}` — unchanged |
| Snapshot `bot_snapshot.db` | `52e39180…` — untouched |
| Staged | **0** |
| Deployment / commit / restart | **none** |

## Scope boundary

This is a **test-isolation** fix. It changes no runtime behaviour: the production
default path is identical, and no strategy, risk, or accounting logic is touched.

Not done here (separately unauthorized): W1 deployment, P3 research.

## Note on an unrelated modified file

`git status` also shows `tests/test_reconciliation_and_db_recovery.py` modified
(mtime 2026-10-25 03:44). That change is **pre-existing**, from earlier
accounting work in a prior session — **not** part of this task. This task's edits
are confined to the three files listed above plus the new test.
