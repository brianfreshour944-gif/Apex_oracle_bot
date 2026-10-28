# COMMIT C — PREPARATION (READ-ONLY)

Date: 2026-10-25. **Read-only. Nothing staged, edited, or committed.**
"Authorize Stage C" was framed as a recommendation + conditional template, so
staging is **held** pending a bare authorization.

Baseline: HEAD `ff4ac11` (B), `main` `7a8ca12`, **0 staged**, 2 unstaged.

---

## 1. Unstaged state is exactly C

`git diff --name-only` → `src/bot.py`, `tests/test_position_outcome_reconstruction.py`.
No other tracked modifications. Untracked = `reports/` + 6 research
scripts/tests (Bucket 2, intentionally untracked) — **not** part of C.

## 2. C hunks in `src/bot.py` — all C-only (B count = 0)

Line numbers shifted after B's commit (was `@@-223/@@-2875/@@-2889`, now
`@@-226/@@-2915/@@-2929`):

| Hunk | C markers | B markers |
| --- | --- | --- |
| `@@ -122,6 +122,8` (`excursion_peak/trough` params) | 2 | 0 |
| `@@ -136,6 +138,13` (docstring) | 2 | 0 |
| `@@ -226,14 +235,21` (peak/trough read) | 4 | 0 |
| `@@ -2915,7 +2931,19` (caller captures peak/trough) | 4 | 0 |
| `@@ -2929,6 +2957,8` (pass to recorder) | 4 | 0 |

## 3. Test-file change is a pure block restoration

`git diff --numstat` → **92 added, 0 removed, one hunk** at `@@ -674,6 +674,98`.
Content = the `# ── MFE/MAE capture` block only (677–768 in the worktree file):
`_FakeStrategy`, `_FakeRiskManager`, `excursion_state` fixture, and the three
tests `test_mfe_mae_recorded_from_caller_supplied_excursions`,
`test_recorder_clear_timing_matches_head`, `test_recorder_early_return_does_not_clear`.

No unrelated change in the file.

## 4. Staging plan

```
src/bot.py                                    → all 5 C hunks (plain `git add`)
tests/test_position_outcome_reconstruction.py → whole file (block re-added; B already committed)
```

Simpler than B: no hunk filtering, no in-hunk split, no untracked-blob dance.
Both files are already tracked and their diffs are C-only.

## 5. Post-staging verification to run

1. Staged diff = only those 2 files.
2. `src/bot.py` staged diff: 5 hunks, all C markers > 0, **B markers = 0**.
3. Test file staged diff: single hunk, 92 added / 0 removed.
4. No research files staged (`test_edge_study`, `test_momentum72_study`,
   `test_analyze_deployed_trades`, `scripts/*`).
5. Full suite on the staged tree → expect **600 passed** (551 tracked + 49
   untracked research) with C's 3 tests now committed-tracked.
6. `git diff` (unstaged) afterwards → **empty**.

## 6. Boundary (unchanged)

Executor isolation stays **separate**; submission-quote capture stays
**disabled** until that work lands. Committing C authorizes none of: executor
changes, enabling capture, push, deploy, live trading.

## Current state

HEAD `ff4ac11`, `main` `7a8ca12`, 0 staged, 2 unstaged, stash empty,
worktrees = main only. State files unchanged.
