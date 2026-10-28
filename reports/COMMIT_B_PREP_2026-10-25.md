# COMMIT B — PREPARATION REPORT (READ-ONLY)

Date: 2026-10-25. **Read-only. No staging, edits, commit, push, deploy, or DB change.**
Prepared because "Authorize Commit B" was issued while the same message warned
that B and C still share hunks. Nothing was staged or committed.

Baseline: branch `w1-instrumentation-accounting`, HEAD `be1def0` (A), `main`
`7a8ca12`, **0 staged**, 4 unstaged.

---

## 1. Method

Classified hunks by **added lines only** (context lines produce false B/C
positives). Then proved separability **empirically** by building B-only and
C-only patches and applying each to a clean A worktree (`be1def0`).

## 2. Hunk map — remaining unstaged changes

### `src/bot.py` — 11 hunks, cleanly separable
| Hunk | Scope |
| --- | --- |
| `@@ -122`, `@@ -136` (signature + docstring `excursion_peak/trough`) | **C** |
| `@@ -165,51 +174,54` (ledger reconstruction block) | **B** |
| `@@ -223,14 +235,21` (peak/trough read) | **C** |
| `@@ -245` (`outcome_status=outcome_status` arg) | **B** |
| `@@ -273`, `@@ -383` (reconstructed-P&L NOTE comments) | **B** |
| `@@ -984`, `@@ -1222` (`reconcile_open_snapshots`) | **B** |
| `@@ -2875`, `@@ -2889` (caller captures peak/trough) | **C** |

**Correction to the earlier map:** `@@ -245` is **B-only**, not "B+C". The
earlier tag came from context lines. No bot.py hunk is genuinely interleaved.

### `src/db.py` — 7 hunks, **all B**
`@@ -75` (`outcome_status` column), `@@ -362` (`reconstruct_position_outcome`,
335 lines), `@@ -532` (migration adds `decision_snapshots.outcome_status`),
`@@ -1127`, `@@ -1292`, `@@ -1322`, `@@ -1335`. **No C hunks in db.py.**

### `src/risk.py` — 4 hunks, **all B**
`exit_commission` plumbing through `_close_open_snapshot` and the two flatten
callers.

### `tests/test_reconciliation_and_db_recovery.py` — 1 hunk, **B**

## 3. The real entanglement: `tests/test_position_outcome_reconstruction.py`

This file is **B and C co-located** (796 lines):

| Lines | Content | Scope |
| --- | --- | --- |
| 1–676 | helpers + B tests (`reconstruct_*`, `standard_close_*`, migration) | **B** |
| **677–768** | **`# ── MFE/MAE capture` block**: `_FakeStrategy`, `_FakeRiskManager`, `excursion_state` fixture, 3 tests | **C** |
| 769–796 | section 8 (moved from fee-provenance) | **B** |

The C block also uses B's helpers `_snapshot`, `_buy`, `_sell`, `_closed`.

## 4. Empirical separability (worktrees at `be1def0`)

| Tree | Result |
| --- | --- |
| B-only patches applied | bot/db/risk apply cleanly; imports OK |
| **B-only full suite** | **550 passed, 1 failed** — the failure is `test_mfe_mae_recorded_from_caller_supplied_excursions` (**C's test**) |
| C-only patches applied | applies cleanly; imports OK |
| **C-only full suite** | **518 passed, 33 failed** — all B tests fail (B's code absent) |

**Conclusion:** the *code* is cleanly separable (each patch applies alone).
The *only* coupling is the shared test file. B's suite is green once C's
MFE/MAE block is excluded; C depends on B, as expected for an ordered sequence.

## 5. Exact B staging plan

```
src/bot.py            → B hunks only: @@-165, @@-245, @@-273, @@-383, @@-984, @@-1222
src/db.py             → all 7 hunks (all B)
src/risk.py           → all 4 hunks (all B)
tests/test_reconciliation_and_db_recovery.py → 1 hunk
tests/test_position_outcome_reconstruction.py → file MINUS the MFE/MAE block (677–768)
```

Method: same hunk-parser → filtered patch → `git apply --cached --recount`.
For the test file, build the B version by removing lines 677–768, stage that
blob, and leave the MFE/MAE block to C.

**C then:** bot.py `@@-122/@@-136/@@-223/@@-2875/@@-2889` + re-add the
MFE/MAE block. C builds on B — a normal ordered sequence.

## 6. Tests to run before staging B

1. B-only worktree (`be1def0` + B patches, test file minus MFE/MAE) →
   `pytest tests/` must be **fully green** (expect 550 passed, 0 failed).
2. Confirm the MFE/MAE block is **absent** from the staged B test file.
3. Confirm B/C markers: staged diff must contain B markers but **no C markers**
   (`excursion`, `max_fav`, `max_adv`, `_trailing_peak`, `_trailing_trough`).

## 7. Blockers / notes

- **`bot.py @@-245` and `db.py @@-1292/@@-1322` are NOT interleaved** — the
  earlier "B+C" tags were context-line artifacts. No in-hunk split is needed
  for B (unlike A's `db.py @@-494`).
- **The one real split is the test file** (§3). This is why committing B before
  this check would have pulled C's tests into B.
- A's `db.py` migration dict already holds `orders`; B adds
  `decision_snapshots` — verified non-overlapping.

## Current state (unchanged)

HEAD `be1def0`, `main` `7a8ca12`, **0 staged**, 4 unstaged, stash empty, no
worktrees beyond the main one. State files unchanged.
