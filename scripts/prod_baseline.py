#!/usr/bin/env python3
"""Produce a consistent, read-only production baseline for the deployed bot.

Run this ON THE HOST (or wherever the live DB lives). It never writes to the
source database file: it takes a consistent copy via the SQLite backup API, then
reports counts and compares them against the known 24-trade baseline.

Usage:
    python scripts/prod_baseline.py /app/data/bot.db
    python scripts/prod_baseline.py /app/data/bot.db --out /tmp/bot_baseline.db

Safety properties (each is covered by a test):
  - the source is opened read-only (mode=ro) and is never opened read-write;
  - the copy is built in a temporary file and moved into place only on success,
    so a failed run leaves no partial file at --out;
  - an existing file at --out is never overwritten without --force;
  - --out is refused when it resolves to the source database itself;
  - the copy is analyzed with immutable=1;
  - no VACUUM / DELETE / write ever touches the source database file.

One caveat on "read-only": SQLite creates -wal/-shm sidecar files beside a
WAL-mode database the moment it is opened, even read-only, so a run can add
those two files next to the source. The source database file itself is never
modified (verified byte-for-byte). Two consequences: the source's directory
must be writable enough for SQLite to create those sidecars -- a read-only
directory aborts the run -- and --out must point somewhere writable. Do not
"fix" this by reading the source with immutable=1: that flag tells SQLite the
file cannot change, so on a live WAL database it reads a stale view and can
miss committed, uncheckpointed data (and even the schema).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import sys
import tempfile

# The 2026-10-25 baseline this compares against.
KNOWN = {"closed": 24, "orders": 84, "filled": 84, "net": -2.1053}

QUOTE_COLS = ("mid_at_submit", "spread_at_submit", "quote_ts", "fee_source")

# Money comparison tolerance: SUM() over floats need not reproduce the
# baseline to the last bit.
NET_TOL = 0.005


def _counts(conn: sqlite3.Connection) -> dict:
    def scalar(sql: str):
        try:
            row = conn.execute(sql).fetchone()
            return row[0] if row else None
        except sqlite3.Error as exc:
            return f"ERROR: {exc}"

    out = {
        "integrity": scalar("PRAGMA integrity_check"),
        "orders": scalar("SELECT COUNT(*) FROM orders"),
        "filled": scalar("SELECT COUNT(*) FROM orders WHERE lower(status)='filled'"),
        "snapshots": scalar("SELECT COUNT(*) FROM decision_snapshots"),
        "closed": scalar(
            "SELECT COUNT(*) FROM decision_snapshots WHERE lower(status)='closed'"
        ),
        "net": scalar(
            "SELECT ROUND(SUM(realized_pnl),4) FROM decision_snapshots "
            "WHERE lower(status)='closed'"
        ),
    }
    try:
        out["order_cols"] = [r[1] for r in conn.execute("PRAGMA table_info(orders)")]
    except sqlite3.Error as exc:
        out["order_cols"] = f"ERROR: {exc}"
    return out


class Refused(Exception):
    """A guard rejected the requested paths; nothing was written."""


def _check_paths(src_path: str, out_path: str, force: bool) -> None:
    """Refuse unsafe destinations BEFORE opening anything."""
    src_abs = os.path.realpath(os.path.abspath(src_path))
    out_abs = os.path.realpath(os.path.abspath(out_path))
    if src_abs == out_abs:
        raise Refused(
            f"--out is the same file as the source database ({out_path}); "
            "refusing to open the live database for writing"
        )
    # samefile also catches a hard link to the source.
    if os.path.exists(src_path) and os.path.exists(out_path):
        try:
            if os.path.samefile(src_path, out_path):
                raise Refused(
                    "--out and the source are the same file (hard link); refusing"
                )
        except OSError:
            pass
    if os.path.exists(out_path) and not force:
        raise Refused(
            f"{out_path} already exists; pass --force to overwrite it"
        )


def _make_copy(src_path: str, out_path: str) -> None:
    """Copy src -> out via the SQLite backup API, atomically.

    The destination is a fresh temporary file in the same directory; it is moved
    into place only after a successful backup. The caller's --out path is never
    opened read-write, so the tool cannot write to the destination database.
    (The source may still gain SQLite -wal/-shm sidecars; see the module
    docstring.)
    """
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".prod_baseline_", suffix=".tmp", dir=out_dir)
    os.close(fd)
    try:
        src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
        try:
            dst = sqlite3.connect(tmp)
            try:
                with dst:
                    src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
        os.replace(tmp, out_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def _is_error(value: object) -> bool:
    return isinstance(value, str) and value.startswith("ERROR")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("db", help="path to the live bot database (read-only)")
    ap.add_argument(
        "--out",
        default="/tmp/bot_baseline.db",
        help="where to write the consistent copy (default: /tmp/bot_baseline.db)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="allow overwriting an existing file at --out",
    )
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"error: source database not found: {args.db}", file=sys.stderr)
        return 2

    try:
        _check_paths(args.db, args.out, args.force)
    except Refused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"source (read-only): {args.db}")
    try:
        _make_copy(args.db, args.out)
    except sqlite3.Error as exc:
        print(f"error: could not copy the database: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: could not write the copy: {exc}", file=sys.stderr)
        return 2
    print(f"consistent copy:    {args.out}")

    conn = sqlite3.connect(f"file:{args.out}?mode=ro&immutable=1", uri=True)
    try:
        counts = _counts(conn)
    finally:
        conn.close()

    print("\n--- counts (on the copy) ---")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    cols = counts.get("order_cols")
    if isinstance(cols, list):
        present = [c for c in QUOTE_COLS if c in cols]
        missing = [c for c in QUOTE_COLS if c not in cols]
        print("\n--- quote instrumentation ---")
        print(f"  present: {present or 'NONE'}")
        print(f"  missing: {missing or 'NONE'}")
        if not present:
            print(
                "  NOTE: no quote columns is consistent with capture never having\n"
                "        been enabled. It is NOT evidence of a broken analyzer."
            )

    # A schema/read error is not a data difference; do not report a verdict
    # that would be read as "new trades exist".
    count_errors = {
        k: counts[k]
        for k in ("orders", "filled", "snapshots", "closed", "net")
        if _is_error(counts.get(k))
    }
    if count_errors:
        print("\n--- comparison vs 2026-10-25 baseline ---")
        for k, v in count_errors.items():
            print(f"  {k}: {v}")
        print(
            "\n  => INCONCLUSIVE: the database could not be read as expected.\n"
            "     This is a schema/read problem, NOT evidence of new trades."
        )
        print("\n--- machine-readable ---")
        print(json.dumps({**counts, "matches_baseline": None}, indent=2, default=str))
        return 2

    print("\n--- comparison vs 2026-10-25 baseline ---")
    same = True
    for key in ("closed", "orders", "filled", "net"):
        cur, ref = counts.get(key), KNOWN.get(key)
        if key == "net" and isinstance(cur, (int, float)) and isinstance(ref, (int, float)):
            match = abs(cur - ref) <= NET_TOL
        else:
            match = cur == ref
        same &= match
        print(f"  {key}: current={cur} baseline={ref} {'MATCH' if match else 'DIFFERS'}")
    if same:
        print(
            "\n  => Counts match exactly: NO new trading evidence since 2026-10-25."
        )
    else:
        print("\n  => Counts differ: NEW data exists. Analyze the copy:")
        print(f"     python scripts/analyze_deployed_trades.py {args.out} --json out.json")

    print("\n--- machine-readable ---")
    print(json.dumps({**counts, "matches_baseline": same}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
