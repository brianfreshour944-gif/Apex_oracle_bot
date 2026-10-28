# tests/test_fee_provenance.py — SPLIT PLAN (READ-ONLY, HELD)

Date: 2026-10-25. **Read-only. Nothing edited or staged by this step.**
Authorization for the split/stage was **not issued** (conditional template).

---

## Why the split is required

`tests/test_fee_provenance.py:171` `test_fee_source_does_not_change_reconstruction_output`
calls `db.reconstruct_position_outcome()`, which **does not exist at HEAD** — it is
**B's** function. A test that cannot pass on an A-only checkout cannot live in A.

## Exact boundary

| Lines | Content | Goes to |
| --- | --- | --- |
| 1–15 | module docstring — **remove the line-12 bullet** about reconstruction | **A** |
| 19–25 | imports | **A** |
| 28–44 | `fresh_db` fixture | **A** |
| 47–52 | `_read_order` helper | **A** |
| 55–118 | sections 1–6 (6 `fee_source` tests) | **A** |
| 121–166 | section 7 (migration + idempotency test) | **A** |
| 169–196 | **section 8** — `test_fee_source_does_not_change_reconstruction_output` | **B** |

**Net:** A keeps the file minus section 8 and the docstring bullet. B receives
section 8, appended to `tests/test_position_outcome_reconstruction.py`.

## Fixture compatibility (verified)

`fresh_db` is **byte-identical** in both files (same dispose/reset/`init_db`
sequence). `_read_order` is A-only; section 8 uses `db.save_order_record` +
`OrderRecord`/`get_db_session` directly, so it needs **no** A helper. The move is
clean.

## Preservation rule (your constraint)

Section 8 moves **verbatim** — same body, same two assertions
(`before["outcome_status"] == "reconciled"`, `after == before`). No assertion is
weakened, skipped, or rewritten to ease the split.

## How to prove A's portion is B-independent (method for the authorized step)

The current worktree contains B's code, so A's portion would pass there for the
wrong reason. The check must run against an **A-only** tree:

1. `git worktree add /tmp/apexA HEAD` (clean checkout at `6c46fb1`).
2. Apply A's staged changes into it.
3. Run `pytest tests/test_fee_provenance.py -q` there.
4. Expect: all A tests pass; `reconstruct_position_outcome` is **absent**
   (grep 0) and section 8 is **not** present.

Using a **temporary worktree** also sidesteps the `stash pop --keep-index`
conflict that bit us during Stage A.

## Post-split staging plan

- A: stage `tests/test_fee_provenance.py` **after** section 8 + the docstring
  bullet are removed (file becomes A-clean).
- B: section 8 is staged with B, later.

## Current state (unchanged)

Branch `w1-instrumentation-accounting`, HEAD `6c46fb1`, `main` `7a8ca12`,
**5 staged** (A's five files, untouched), 4 unstaged, stash empty.
`test_fee_provenance.py` is **untracked and unmodified** — the split has not begun.
