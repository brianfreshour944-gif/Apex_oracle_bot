"""Tests for scripts/prod_baseline.py.

The script is a read-only operator tool. These tests pin both the happy path and
the operational hazards found in review (PB-1..PB-4): the tool must never open
the source read-write, must never overwrite an existing file without --force,
must refuse --out == source, and must not leave debris on failure.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prod_baseline.py"


def _load():
    spec = importlib.util.spec_from_file_location("prod_baseline", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pb():
    return _load()


def _seed(path: Path, *, closed: int, orders: int, net: float, quote_cols: bool, wal: bool = False) -> None:
    """Build a database with the real schema shape used by the bot."""
    conn = sqlite3.connect(path)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    if quote_cols:
        conn.execute(
            "CREATE TABLE orders (order_id TEXT, decision_id TEXT, status TEXT,"
            " commission REAL, mid_at_submit REAL, spread_at_submit REAL,"
            " quote_ts DATETIME, fee_source TEXT)"
        )
    else:
        conn.execute(
            "CREATE TABLE orders (order_id TEXT, decision_id TEXT, status TEXT,"
            " commission REAL)"
        )
    conn.execute(
        "CREATE TABLE decision_snapshots (decision_id TEXT, status TEXT,"
        " realized_pnl REAL)"
    )
    for i in range(orders):
        conn.execute(
            "INSERT INTO orders (order_id, decision_id, status, commission)"
            " VALUES (?,?,?,?)",
            (f"o{i}", f"d{i}", "filled", 0.0),
        )
    for i in range(closed):
        conn.execute(
            "INSERT INTO decision_snapshots VALUES (?,?,?)",
            (f"d{i}", "closed", net / max(closed, 1)),
        )
    conn.commit()
    conn.close()


# --- happy path -------------------------------------------------------------


def test_matches_baseline_exactly(pb, tmp_path, capsys):
    db = tmp_path / "bot.db"
    _seed(db, closed=24, orders=84, net=-2.1053, quote_cols=False)
    assert pb.main([str(db), "--out", str(tmp_path / "copy.db")]) == 0
    assert "NO new trading evidence" in capsys.readouterr().out


def test_detects_new_data(pb, tmp_path, capsys):
    db = tmp_path / "bot.db"
    _seed(db, closed=30, orders=100, net=5.0, quote_cols=False)
    pb.main([str(db), "--out", str(tmp_path / "copy.db")])
    text = capsys.readouterr().out
    assert "NEW data exists" in text
    assert "DIFFERS" in text


def test_reports_quote_columns_present(pb, tmp_path, capsys):
    db = tmp_path / "bot.db"
    _seed(db, closed=24, orders=84, net=-2.1053, quote_cols=True)
    pb.main([str(db), "--out", str(tmp_path / "copy.db")])
    text = capsys.readouterr().out
    assert "mid_at_submit" in text
    assert "present: NONE" not in text


def test_no_quote_columns_notes_capture_never_enabled(pb, tmp_path, capsys):
    db = tmp_path / "bot.db"
    _seed(db, closed=24, orders=84, net=-2.1053, quote_cols=False)
    pb.main([str(db), "--out", str(tmp_path / "copy.db")])
    text = capsys.readouterr().out
    assert "capture never having" in text
    assert "NOT evidence of a broken analyzer" in text


def test_source_is_never_modified(pb, tmp_path):
    db = tmp_path / "bot.db"
    _seed(db, closed=24, orders=84, net=-2.1053, quote_cols=False)
    before = db.read_bytes()
    pb.main([str(db), "--out", str(tmp_path / "copy.db")])
    assert db.read_bytes() == before, "source database must not be modified"


def test_copy_is_independent_of_source(pb, tmp_path):
    db = tmp_path / "bot.db"
    _seed(db, closed=24, orders=84, net=-2.1053, quote_cols=False)
    out = tmp_path / "copy.db"
    pb.main([str(db), "--out", str(out)])
    conn = sqlite3.connect(f"file:{out}?mode=ro", uri=True)
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 84
    conn.close()


# --- PB-3 / PB-1: --out must not be the source ------------------------------


def test_refuses_out_equal_to_source(pb, tmp_path, capsys):
    db = tmp_path / "bot.db"
    _seed(db, closed=24, orders=84, net=-2.1053, quote_cols=False)
    before = db.read_bytes()
    rc = pb.main([str(db), "--out", str(db)])
    assert rc == 2
    assert "same file" in capsys.readouterr().err
    assert db.read_bytes() == before


def test_refuses_out_equal_to_source_via_relative_path(pb, tmp_path, capsys, monkeypatch):
    db = tmp_path / "bot.db"
    _seed(db, closed=24, orders=84, net=-2.1053, quote_cols=False)
    monkeypatch.chdir(tmp_path)
    rc = pb.main(["bot.db", "--out", "./bot.db"])
    assert rc == 2
    assert "same file" in capsys.readouterr().err


def test_out_equal_to_source_does_not_create_wal_sidecars(pb, tmp_path):
    """PB-1: a WAL-mode live DB must not gain -wal/-shm from this tool."""
    db = tmp_path / "live.db"
    _seed(db, closed=24, orders=84, net=-2.1053, quote_cols=False, wal=True)
    # settle any pre-existing sidecars
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    for suffix in ("-wal", "-shm"):
        p = Path(str(db) + suffix)
        if p.exists():
            p.unlink()
    pb.main([str(db), "--out", str(db)])
    assert not Path(str(db) + "-wal").exists(), "tool created a -wal on the live DB"
    assert not Path(str(db) + "-shm").exists(), "tool created a -shm on the live DB"


# --- PB-2: no silent overwrite ----------------------------------------------


def test_refuses_existing_out_without_force(pb, tmp_path, capsys):
    db = tmp_path / "bot.db"
    _seed(db, closed=24, orders=84, net=-2.1053, quote_cols=False)
    out = tmp_path / "existing.db"
    _seed(out, closed=1, orders=1, net=9.0, quote_cols=False)
    before = out.read_bytes()
    rc = pb.main([str(db), "--out", str(out)])
    assert rc == 2
    assert "already exists" in capsys.readouterr().err
    assert out.read_bytes() == before, "existing snapshot must be untouched"


def test_force_allows_overwrite(pb, tmp_path):
    db = tmp_path / "bot.db"
    _seed(db, closed=24, orders=84, net=-2.1053, quote_cols=False)
    out = tmp_path / "existing.db"
    _seed(out, closed=1, orders=1, net=9.0, quote_cols=False)
    assert pb.main([str(db), "--out", str(out), "--force"]) == 0
    conn = sqlite3.connect(f"file:{out}?mode=ro", uri=True)
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 84
    conn.close()


def test_refuses_existing_non_sqlite_file_without_force(pb, tmp_path, capsys):
    db = tmp_path / "bot.db"
    _seed(db, closed=24, orders=84, net=-2.1053, quote_cols=False)
    out = tmp_path / "precious.txt"
    out.write_text("IMPORTANT EXISTING DATA")
    rc = pb.main([str(db), "--out", str(out)])
    assert rc == 2
    assert out.read_text() == "IMPORTANT EXISTING DATA"


# --- PB-4: failures are clean, no debris ------------------------------------


def test_missing_source_is_clean_error(pb, tmp_path, capsys):
    rc = pb.main([str(tmp_path / "nope.db"), "--out", str(tmp_path / "x.db")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err
    assert "Traceback" not in err
    assert not (tmp_path / "x.db").exists()


def test_bad_source_leaves_no_partial_file(pb, tmp_path, capsys):
    bad = tmp_path / "notadb.txt"
    bad.write_text("not a database" * 100)
    out = tmp_path / "partial.db"
    rc = pb.main([str(bad), "--out", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert not out.exists(), "a failed run must not leave a file at --out"
    assert not list(tmp_path.glob(".prod_baseline_*")), "temp file left behind"


def test_out_in_missing_dir_is_clean_error(pb, tmp_path, capsys):
    db = tmp_path / "bot.db"
    _seed(db, closed=24, orders=84, net=-2.1053, quote_cols=False)
    rc = pb.main([str(db), "--out", str(tmp_path / "nodir" / "x.db")])
    assert rc == 2
    assert "Traceback" not in capsys.readouterr().err


# --- schema problems are inconclusive, not "new data" -----------------------


def test_partial_schema_is_inconclusive_not_new_data(pb, tmp_path, capsys):
    db = tmp_path / "partial.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE orders (order_id TEXT, status TEXT)")
    conn.commit()
    conn.close()
    rc = pb.main([str(db), "--out", str(tmp_path / "copy.db")])
    assert rc == 2
    text = capsys.readouterr().out
    assert "INCONCLUSIVE" in text
    assert "NEW data exists" not in text


# --- locked source (realistic host case) ------------------------------------


def test_locked_source_still_produces_a_copy(pb, tmp_path):
    db = tmp_path / "live.db"
    _seed(db, closed=24, orders=84, net=-2.1053, quote_cols=False, wal=True)
    holder = sqlite3.connect(db)
    holder.execute("BEGIN EXCLUSIVE")
    holder.execute("INSERT INTO orders (order_id, decision_id, status, commission)"
                   " VALUES ('x','dx','filled',0)")
    try:
        out = tmp_path / "copy.db"
        rc = pb.main([str(db), "--out", str(out)])
        assert rc == 0
        assert out.exists()
    finally:
        holder.rollback()
        holder.close()


# --- money tolerance --------------------------------------------------------


def test_net_uses_tolerance(pb, tmp_path, capsys):
    """A net within NET_TOL of the baseline is a MATCH, not a DIFFERS."""
    db = tmp_path / "bot.db"
    # 24 rows whose rounded sum lands a hair off the baseline.
    _seed(db, closed=24, orders=84, net=-2.1053, quote_cols=False)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE decision_snapshots SET realized_pnl = -2.1054/24")
    conn.commit()
    conn.close()
    pb.main([str(db), "--out", str(tmp_path / "copy.db")])
    text = capsys.readouterr().out
    assert "net: current=-2.1054 baseline=-2.1053 MATCH" in text
