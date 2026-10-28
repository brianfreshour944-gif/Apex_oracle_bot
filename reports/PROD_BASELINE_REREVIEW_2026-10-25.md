# RE-REVIEW — scripts/prod_baseline.py after PB-1..PB-4 fixes (READ-ONLY)

Date: 2026-10-25. **Read-only.** No commit authorized, no production access, no
deployment change. `CAPTURE_SUBMISSION_QUOTE` stays `False`.

Reviewer: agent, adversarial pass. Target: the fixed `scripts/prod_baseline.py`
(232 lines) and `tests/test_prod_baseline.py` (18 tests), both **untracked**.

## VERDICT: PB-1..PB-4 are genuinely CLOSED

Each was re-tested by trying to bypass it, not by re-reading the fix. One new
**informational** finding (PB-5) surfaced that is SQLite-inherent, not a defect,
but does carry an operational constraint that should be documented.

| ID | Status | Evidence |
| --- | --- | --- |
| PB-1 (out opened read-write) | **CLOSED** | no sidecars created on the target; atomic temp+replace |
| PB-2 (silent overwrite) | **CLOSED** | existing file refused, exit 2, bytes unchanged |
| PB-3 (no same-path guard) | **CLOSED** | abs/relative/symlink/hard-link all refused |
| PB-4 (tracebacks, debris) | **CLOSED** | one-line errors, exit 2, no partial file |
| PB-5 (WAL source sidecars) | **NEW, informational** | SQLite-inherent; see below |

---

## PB-1 — CLOSED (bypass attempts failed)

- `--out == source`, WAL mode → refused with a clear message; **no `-wal`/`-shm`
  created on the target** (the original repro is gone).
- `--out == source --force` → **still refused**. `--force` does not bypass the
  same-file guard, which is the right precedence.
- The copy is built via `mkstemp` in the destination directory and `os.replace`d;
  the caller's `--out` is never passed to `sqlite3.connect` read-write.

## PB-2 — CLOSED

- Existing **SQLite** file at `--out` → refused, exit 2, bytes byte-identical
  afterwards.
- Existing **non-SQLite** file → refused, contents preserved.
- `--force` correctly allows overwrite and the result reads back correctly.

## PB-3 — CLOSED (four bypass attempts)

| Attempt | Result |
| --- | --- |
| `--out` absolute == source | refused |
| `--out ./bot.db` relative == source | refused |
| `--out` symlink → source | refused (realpath resolves it) |
| source symlink, `--out` real path | refused |

Exit code is **2** for all refusals.

## PB-4 — CLOSED

- Missing source → `error: source database not found: …`, exit 2, no traceback.
- Non-database source → `error: could not copy the database: file is not a
  database`, exit 2, **no file left at `--out`**, no `.prod_baseline_*` temp
  debris.
- `--out` in a missing directory → clean error, exit 2.
- `--out` is a directory → clean error, exit 2, no debris.
- Source `chmod 444` with a writable destination → **succeeds** (no source write
  needed). Correct.

## Correctness checks (beyond the review scope, but worth having)

- **Uncheckpointed WAL data is captured.** Wrote 84 orders / 24 closed, did not
  checkpoint (WAL = 24,752 bytes), ran the tool → copy reports 84 / 24 / −2.1053.
  The backup API reads committed-but-uncheckpointed frames.
- **Concurrent writer.** With a live writer inserting continuously, the tool
  produced a consistent 29-row copy, `integrity_check = ok`, no error. This is
  the realistic host case and it works.
- Source bytes (sha256) unchanged after every run.

---

## PB-5 (NEW, informational) — a WAL-mode source gains `-wal`/`-shm` beside it

Opening a WAL-mode database, even with `mode=ro`, makes SQLite create `-shm` and
`-wal` **next to the source**. Reproduced with the tool, and then with a bare
`sqlite3.connect("file:…?mode=ro")` read that does nothing but `SELECT COUNT(*)`
— so **this is SQLite behavior, not a defect in the tool.**

Two consequences worth knowing:

1. **A read-only source directory makes the tool fail.** With the source
   directory `chmod 555`, the run aborts: `attempt to write a readonly database`
   (exit 2, clean error, no debris). If `/app/data` is read-only in production,
   the tool will not work — the copy must go somewhere writable and the source
   directory must permit sidecar creation.
2. **The tool is read-only for the database file, not strictly for the
   filesystem.** The main DB's bytes are never modified (verified), but sidecar
   files may appear. On the host the bot already runs WAL, so these almost
   certainly already exist — the practical impact is low, but the docstring's
   phrasing ("never writes to the source database") should say *database file*,
   not imply *no filesystem effect*.

**Why not just use `immutable=1` to avoid the sidecars?** Because it is **wrong
for a live database**, and I proved it: with a writer holding committed but
uncheckpointed WAL frames, `immutable=1` returned `no such table: t`, while
`mode=ro` returned the correct 50 rows. `immutable=1` assumes the file cannot
change; on a live WAL DB it reads a stale view and can miss the schema entirely.
So `mode=ro` is the **correct** choice for the source, and the sidecar side
effect is the price. The `immutable=1` use in `_counts` is on the **copy**
(static) and is correct there.

**Recommendation (non-blocking):** one docstring sentence noting that a WAL
source may gain sidecar files and that the source directory must be writable.
No code change required for correctness.

## Residual observations (minor, non-blocking)

- Refusal messages go to stderr; the `source (read-only): …` line goes to stdout,
  so under some shells the error can appear to precede it. Cosmetic.
- `--force` + `--out` pointing at a directory is caught at `os.replace` with a
  clean message rather than up front. Acceptable.
- The tool does not verify the copy's `integrity_check` before reporting counts
  (it prints `integrity` from the copy, so a corrupt copy is visible, but the
  verdict is still computed). Minor.

## Test-suite assessment

18 tests, each pinned to a behaviour. The 12 new ones are true negative controls:
against a reconstructed pre-fix tool, **7 fail**; all 18 pass on the fixed tool.
Coverage now includes out==source (absolute, relative, symlink, hard link), WAL
sidecar absence, existing-out refusal, `--force`, missing/bad source, missing
directory, partial schema, locked source, and the net tolerance.

## Required changes before commit

**None blocking.** PB-1..PB-4 are closed. Optional: the PB-5 docstring note.

## Boundary

Read-only re-review. Nothing changed or committed. No production access used or
sought. `CAPTURE_SUBMISSION_QUOTE` remains `False`.
