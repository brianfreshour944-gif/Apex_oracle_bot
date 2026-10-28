# COMMIT D — PREPARATION & VALIDATION (READ-ONLY)

Date: 2026-10-25. **Nothing staged, nothing committed.** Branch
`w1-instrumentation-accounting`, 0 commits, 0 staged. Read-only preparation for
Commit D, per the recommended sequence.

---

## 1. Exact hunk split for `src/config.py`

The file contains **D hunks and one A hunk**. `git diff -U0` shows:

| Hunk | Content | Scope |
| --- | --- | --- |
| `@@ -2,0 +3 @@` | `+import os` | **D** |
| `@@ -425,0 +427,8 @@` | `+CAPTURE_SUBMISSION_QUOTE` field | **A — EXCLUDE** |
| `@@ -687,0 +697,14 @@` | `+PERFORMANCE_TRACKER_STATE_PATH`, `+FEATURE_DRIFT_STATE_PATH` | **D** |
| `@@ -950,0 +974,2 @@` | `+` the two module exports | **D** |

The A hunk is **cleanly separable** — it is a distinct, contiguous block. Staging
D requires a partial stage of this file (`git add -p`, or a patch-based `git
apply --cached`), selecting the three D hunks and rejecting the A hunk.

## 2. The other three D files are pure D

`src/performance_tracker.py`, `src/feature_drift_monitor.py`, and
`tests/conftest.py` contain **only** D changes — no A/B/C contamination. Verified
by full diff inspection.

## 3. Orphaned-import check (a real risk of this change)

Both modules removed `_STATE_PATH = os.path.join(...)`. If `os` were then unused,
Ruff would flag an orphaned import. **Checked:** both files still use `os.` (2
uses each, in their atomic-write helpers), so `import os` remains needed. No
orphan.

## 4. Validation results (read-only)

| Check | Result |
| --- | --- |
| D-scope isolation tests | **8 passed** in 0.44s |
| Full suite | **600 passed**, 5 warnings, 26.5s |
| Ruff (D files) | `All checks passed!` |

**Decisive check — the suite did not touch real runtime state:**

| File | Before | After |
| --- | --- | --- |
| tracker hash | `44136fa355b3678a…` | `44136fa355b3678a…` **UNCHANGED** |
| tracker mtime | `04:48:19.011` | `04:48:19.011` **UNCHANGED** |
| drift hash | `82a78c0d2d33025d…` | `82a78c0d2d33025d…` **UNCHANGED** |
| drift mtime | `04:48:57.711` | `04:48:57.711` **UNCHANGED** |

The isolation works: a full-suite run leaves both real files byte-identical.

## 5. Proposed Commit D

**Message:** `test: isolate performance-tracker and feature-drift state files`

**Files (6):**

```
src/config.py                              (3 of 4 hunks — D only)
src/performance_tracker.py
src/feature_drift_monitor.py
tests/conftest.py
tests/test_performance_tracker_isolation.py
tests/test_feature_drift_isolation.py
```

**Excluded:** `CAPTURE_SUBMISSION_QUOTE` hunk (A), all of `bot.py`, `db.py`,
`exchange.py`, `risk.py`, all research files, `reports/backup/`.

## 6. Not done (awaiting explicit authorization)

- **Staging** — the index was not modified. `git add -p`/patch stage is a
  mutation and was held.
- **Commit D** — held.
- Quote-executor isolation — held, separate scope.
- Deployment — held.

## 7. Decisions settled by you this round

- **Tracker backup:** preserve outside implementation commits; keep the active
  tracker empty; do not commit synthetic records into production history; verify
  recoverability before any cleanup/migration.
- **Research files:** leave untracked; preserve, don't delete.

## Safety state

Branch `w1-instrumentation-accounting`; **0 commits, 0 staged**, no file
modified. Tracker `44136fa3…` = `{}`, drift `82a78c0d…`, snapshot `52e39180…`,
backup `ddb2e987…` all unchanged.
