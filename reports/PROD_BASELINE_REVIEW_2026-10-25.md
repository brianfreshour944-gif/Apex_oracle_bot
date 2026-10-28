# REVIEW — scripts/prod_baseline.py + tests (READ-ONLY, pre-commit)

Date: 2026-10-25. **Read-only.** No commit authorized, no production access, no
deployment change. `CAPTURE_SUBMISSION_QUOTE` stays `False`.

Reviewer: agent. Reviewed artifact: `scripts/prod_baseline.py` (131 lines) and
`tests/test_prod_baseline.py` (6 tests), both **untracked**.

## VERDICT: NOT READY TO COMMIT — 4 findings, 1 HIGH

The tool's *happy path* is correct and its core safety claim holds: in every
normal invocation the source database was byte-identical before and after. But
three of the four operational edge cases named in the review request are **not**
handled, and one of them can create files next to a live WAL-mode database.

| ID | Severity | Finding |
| --- | --- | --- |
| PB-1 | **HIGH** | `--out` is opened read-write; pointing it at the source (or any live DB) creates `-shm`/`-wal` sidecars on that database |
| PB-2 | MEDIUM | An existing file at `--out` is overwritten with **no warning and no confirmation** |
| PB-3 | MEDIUM | No guard that `--out` differs from the source path |
| PB-4 | LOW | Failure paths raise bare tracebacks (no friendly message) and can leave a 0-byte file at `--out` |

---

## PB-1 (HIGH) — `--out` is opened read-write; sidecars land next to a live DB

`_make_copy` (line 63) opens the destination with a plain
`sqlite3.connect(out_path)` — **no `mode=ro`**. If `out_path` is itself a SQLite
database, SQLite opens it read-write and, for a WAL-mode database, creates
`-shm` and `-wal` sidecar files beside it.

**Reproduced** against a WAL-mode database:
```
before: live.db only
after running:  python scripts/prod_baseline.py /tmp/probe2/live.db --out /tmp/probe2/live.db
live.db  live.db-shm  live.db-wal     <-- two new files created on the live DB
```
The main file's bytes were unchanged, but **the tool still opened the live
database read-write and wrote sidecar files**. On the production host, if anyone
ever passes `--out` equal to (or inside the directory of) `/app/data/bot.db`,
this touches the live database. The tool's docstring claims it "never writes to
the source database"; **that claim is false for this input.**

Note also: even `_counts` opens with `immutable=1` (line 87) which is a *static
snapshot assertion* — correct for a copy, wrong if the path is live.

## PB-2 (MEDIUM) — silent overwrite of an existing file at `--out`

`--out` is overwritten with no existence check and no confirmation.

**Reproduced** with a pre-existing *valid SQLite database* at `--out`:
```
existing: /tmp/p3/other.db  (table 'precious', 1 row)
after:    the file now holds the source's tables; 'precious' is gone
```
A pre-existing *non-SQLite* file at `--out` is **not** clobbered (SQLite refuses
it), which is why the hazard is easy to miss: it only bites when the target is
already a SQLite file — i.e. exactly the case of overwriting a previous baseline
snapshot.

## PB-3 (MEDIUM) — no guard that `--out` differs from the source

There is no `os.path.samefile` / realpath comparison between `args.db` and
`args.out`. Combined with PB-1 and PB-2, `--out == source` is accepted silently.
A one-line guard would turn a dangerous input into a clear error.

## PB-4 (LOW) — bare tracebacks and a stray partial file

- Missing source: `sqlite3.OperationalError: unable to open database file` with a
  full traceback (exit 1). Correct exit code, unfriendly output.
- `--out` in a nonexistent directory: same traceback.
- Non-database source: exit 1, **and a 0-byte file is left at `--out`**
  (reproduced). A failed run should not leave debris that a later run might
  mistake for a snapshot.

---

## What the review request asked, and the answers

**SQLite connection cleanup** — SOUND. `_make_copy` closes `dst` in a `finally`
and `src` in an outer `finally` (lines 62-70); `_counts`' connection is closed in
a `finally` (lines 88-91). Ran under `-W error::ResourceWarning`: no warning.
No leak found.

**Backup failures** — PARTIAL. Failures propagate and exit 1 (good), but with a
bare traceback (PB-4) and a possible 0-byte `--out` (PB-4). A **locked source**
(the realistic host case) is handled well: with another connection holding
`BEGIN EXCLUSIVE`, the backup still produced a readable copy.

**Source read-only enforcement** — SOUND for the source *handle*: it is opened
`mode=ro` (line 61). NOT sound for the source *path*: see PB-1. The source file's
bytes were unchanged in every test, including `--out == source` and WAL mode.

**Accidental overwrite of an existing snapshot** — **NOT PREVENTED (PB-2)**.

## Secondary observations

- **Float equality.** `net` is compared with `==` against `-2.1053`. Tested: the
  rounded SQL `SUM` reproduces `-2.1053` exactly here, so this works for the
  known data — but exact float equality on a money value is fragile. Prefer a
  tolerance (`abs(cur-ref) < 0.005`). Low risk; noted for correctness.
- **Partial-schema DB.** A DB lacking `decision_snapshots` prints
  `ERROR: no such table` and reports "DIFFERS"/"NEW data exists". The error text
  is visible, but "NEW data exists" is a misleading verdict for a schema problem.
- **`order_cols` ERROR branch** (lines 55-56) is untested.
- **Test gaps:** no test covers `--out == source`, an existing `--out`, a locked
  source, a missing source, a partial schema, or a backup failure. The 6 tests
  all exercise the happy path plus the quote-column branches.

## Required changes before commit

1. **PB-1**: refuse to open a destination that is an existing SQLite database
   unless `--force` is given, and/or open the destination in a way that cannot
   touch a live DB. At minimum, reject `--out == source`.
2. **PB-3**: add an explicit `samefile`/realpath guard with a clear error.
3. **PB-2**: refuse to overwrite an existing file without `--force`.
4. **PB-4**: catch `sqlite3.Error` and print a one-line error; do not leave a
   0-byte file behind on failure.
5. Tests for each of the above (these are the negative controls that are
   currently missing).

## Boundary

Read-only review. Nothing changed; nothing committed. No production access used
or sought. `CAPTURE_SUBMISSION_QUOTE` remains `False`.
