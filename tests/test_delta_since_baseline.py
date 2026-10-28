"""Tests for scripts/delta_since_baseline.py.

Each expectation is pinned to a hand-computed value on a real-schema seeded
database, so a regression in the cutoff split or the per-bucket arithmetic fails
here rather than silently mis-reporting a live snapshot.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "delta_since_baseline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("delta_since_baseline", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


dsb = _load_module()

CUTOFF = dsb.CUTOFF


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE decision_snapshots (
            decision_id TEXT PRIMARY KEY,
            symbol TEXT, regime TEXT, final_action TEXT,
            entry_price REAL, qty REAL, status TEXT, exit_reason TEXT,
            max_favorable_pct REAL, max_adverse_pct REAL,
            realized_pnl REAL, return_pct REAL, holding_period_sec REAL,
            votes_json TEXT, closed_at TEXT, created_at TEXT
        );
        CREATE TABLE orders (
            order_id TEXT, decision_id TEXT, symbol TEXT, side TEXT,
            qty REAL, filled_qty REAL, filled_avg_price REAL,
            commission REAL, status TEXT, type TEXT, time_in_force TEXT,
            client_order_id TEXT, submitted_at TEXT, filled_at TEXT
        );
        """
    )
    return conn


def _trade(conn, did, symbol, pnl, closed_at, *, regime="sideways",
           exit_reason="max_hold_time_exceeded", votes=None, return_pct=None):
    conn.execute(
        "INSERT INTO decision_snapshots (decision_id, symbol, regime, final_action, "
        "entry_price, qty, status, exit_reason, realized_pnl, return_pct, "
        "holding_period_sec, votes_json, closed_at, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (did, symbol, regime, "buy", 100.0, 0.01, "closed", exit_reason, pnl,
         pnl * 100 if return_pct is None else return_pct, 28800.0,
         json.dumps(votes or {"quant": "buy"}), closed_at, closed_at),
    )


def _order(conn, oid, did, symbol, side, commission=0.01, filled_at=None):
    conn.execute(
        "INSERT INTO orders (order_id, decision_id, symbol, side, qty, filled_qty, "
        "filled_avg_price, commission, status, type, time_in_force, submitted_at, filled_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (oid, did, symbol, side, 0.01, 0.01, 100.0, commission, "filled", "market",
         "gtc", filled_at or CUTOFF, filled_at or CUTOFF),
    )


def _seed_old(conn, n=2):
    """n trades at/before the cutoff."""
    for i in range(n):
        did = f"old{i:03d}"
        _trade(conn, did, "BTC/USD", -0.10, f"2026-10-0{i + 1} 00:00:00")
        _order(conn, f"o-old{i}", did, "BTC/USD", "buy")
    conn.commit()


# ---------------------------------------------------------------- cutoff split

def test_cutoff_is_the_local_snapshot_newest_close():
    assert CUTOFF == "2026-10-24 04:21:08.380985"
    assert dsb.EXPECTED_OLD == 24


def test_trade_exactly_at_cutoff_counts_as_old(tmp_path):
    """<= cutoff is old; only strictly newer is new. Exercises the real split."""
    conn = _make_db(tmp_path / "t.db")
    _trade(conn, "boundary", "BTC/USD", -0.10, CUTOFF)
    _trade(conn, "newer", "BTC/USD", -0.20, "2026-10-24 04:21:09.000000")
    conn.commit()
    old, new = dsb.split_at_cutoff(_closed(conn), CUTOFF)
    assert [r["decision_id"] for r in old] == ["boundary"]
    assert [r["decision_id"] for r in new] == ["newer"]
    conn.close()


def test_split_is_stable_when_nothing_is_new(tmp_path):
    conn = _make_db(tmp_path / "t.db")
    _seed_old(conn, 3)
    conn.commit()
    old, new = dsb.split_at_cutoff(_closed(conn), CUTOFF)
    assert len(old) == 3 and new == []
    conn.close()


def _closed(conn):
    cur = conn.execute(
        "SELECT decision_id, symbol, regime, exit_reason, realized_pnl, return_pct, "
        "holding_period_sec, closed_at, votes_json FROM decision_snapshots "
        "WHERE lower(status)='closed' ORDER BY closed_at"
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]


# ------------------------------------------------------------------ arithmetic

def test_net_and_pf_hand_computed(tmp_path):
    conn = _make_db(tmp_path / "t.db")
    # new trades: +0.50, -0.20, -0.30 -> net -0.00 ... choose unambiguous values
    _trade(conn, "n1", "BTC/USD", +0.60, "2026-10-25 00:00:00")
    _trade(conn, "n2", "BTC/USD", -0.20, "2026-10-25 01:00:00")
    _trade(conn, "n3", "BTC/USD", -0.10, "2026-10-25 02:00:00")
    conn.commit()
    s = dsb._summ(_closed(conn))
    assert s["trades"] == 3
    assert s["wins"] == 1 and s["losses"] == 2
    assert s["net"] == pytest.approx(0.30)
    assert s["gross_profit"] == pytest.approx(0.60)
    assert s["gross_loss"] == pytest.approx(-0.30)
    assert s["pf"] == pytest.approx(2.0)          # 0.60 / 0.30
    assert s["win_rate"] == pytest.approx(100 / 3)
    conn.close()


def test_pf_is_none_when_no_losses(tmp_path):
    conn = _make_db(tmp_path / "t.db")
    _trade(conn, "n1", "BTC/USD", +0.10, "2026-10-25 00:00:00")
    conn.commit()
    assert dsb._summ(_closed(conn))["pf"] is None
    conn.close()


def test_pf_is_zero_when_no_wins(tmp_path):
    conn = _make_db(tmp_path / "t.db")
    _trade(conn, "n1", "BTC/USD", -0.10, "2026-10-25 00:00:00")
    conn.commit()
    assert dsb._summ(_closed(conn))["pf"] == 0.0
    conn.close()


def test_max_drawdown_uses_running_peak():
    # peak 12 then trough 6 -> -50%; the -40% figure would use peak 10
    assert dsb._max_dd_pct([10.0, 12.0, 6.0, 8.0]) == pytest.approx(-50.0)


def test_max_drawdown_all_losses_from_zero_is_zero():
    # equity starts at 0 and only falls; a percentage drawdown from 0 is undefined
    assert dsb._max_dd_pct([0.0, -1.79]) == pytest.approx(0.0)


def test_empty_bucket_has_zero_win_rate_not_a_crash():
    s = dsb._summ([])
    assert s["trades"] == 0 and s["win_rate"] == 0.0 and s["pf"] is None


# --------------------------------------------------------------------- buckets

def test_buckets_split_by_symbol_and_regime(tmp_path):
    conn = _make_db(tmp_path / "t.db")
    _trade(conn, "n1", "ETH/USD", -0.40, "2026-10-25 00:00:00", regime="sideways")
    _trade(conn, "n2", "ETH/USD", -0.10, "2026-10-25 01:00:00", regime="neutral")
    _trade(conn, "n3", "SOL/USD", +0.30, "2026-10-25 02:00:00", regime="sideways")
    conn.commit()
    new = _closed(conn)
    bysym = {k: dsb._summ(v) for k, v in dsb._bucket(new, "symbol").items()}
    assert bysym["ETH/USD"]["net"] == pytest.approx(-0.50)
    assert bysym["SOL/USD"]["net"] == pytest.approx(0.30)
    byreg = {k: dsb._summ(v) for k, v in dsb._bucket(new, "regime").items()}
    assert byreg["sideways"]["trades"] == 2
    assert byreg["neutral"]["trades"] == 1
    conn.close()


def test_bucket_handles_none_key():
    out = dsb._bucket([{"regime": None, "realized_pnl": -0.1}], "regime")
    assert "(none)" in out


# ----------------------------------------------------------------------- votes

def test_vote_tally_counts_each_brain_action(tmp_path, capsys):
    conn = _make_db(tmp_path / "t.db")
    votes = {"quant": "buy", "sentinel": "hold"}
    _trade(conn, "n1", "BTC/USD", -0.10, "2026-10-25 00:00:00", votes=votes)
    _trade(conn, "n2", "BTC/USD", -0.10, "2026-10-25 01:00:00", votes=votes)
    conn.commit()
    conn.close()
    rc = dsb.main([str(tmp_path / "t.db")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "quant" in out and "'buy': 2" in out
    assert "'hold': 2" in out


def test_malformed_votes_json_is_skipped(tmp_path, capsys):
    conn = _make_db(tmp_path / "t.db")
    conn.execute(
        "INSERT INTO decision_snapshots (decision_id, symbol, status, realized_pnl, "
        "return_pct, votes_json, closed_at) VALUES (?,?,?,?,?,?,?)",
        ("bad", "BTC/USD", "closed", -0.1, -1.0, "{not json", "2026-10-25 00:00:00"),
    )
    conn.commit()
    conn.close()
    rc = dsb.main([str(tmp_path / "t.db")])
    assert rc == 0  # does not crash on bad JSON


# ---------------------------------------------------------------------- CLI

def test_no_new_trades_returns_zero_and_says_so(tmp_path, capsys):
    conn = _make_db(tmp_path / "t.db")
    _seed_old(conn, 2)
    conn.close()
    rc = dsb.main([str(tmp_path / "t.db"), "--cutoff", CUTOFF])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No trades newer than the cutoff" in out


def test_cutoff_mismatch_warns(tmp_path, capsys):
    """If the old count is not 24 the tool must say so, not silently proceed."""
    conn = _make_db(tmp_path / "t.db")
    _seed_old(conn, 2)
    conn.close()
    dsb.main([str(tmp_path / "t.db")])
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "expected 24" in out


def test_new_trades_are_reported_with_counts(tmp_path, capsys):
    conn = _make_db(tmp_path / "t.db")
    _seed_old(conn, 24)
    _trade(conn, "n1", "ETH/USD", -0.40, "2026-10-25 00:00:00")
    _trade(conn, "n2", "SOL/USD", +0.10, "2026-10-25 01:00:00")
    conn.commit()
    conn.close()
    rc = dsb.main([str(tmp_path / "t.db")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NEW        : 2" in out
    assert "trades=2" in out
    assert "WARNING" not in out


def test_orders_linked_only_to_new_decisions(tmp_path, capsys):
    conn = _make_db(tmp_path / "t.db")
    _seed_old(conn, 24)
    _trade(conn, "n1", "ETH/USD", -0.40, "2026-10-25 00:00:00")
    _order(conn, "oa", "n1", "ETH/USD", "buy")
    _order(conn, "ob", "n1", "ETH/USD", "sell")
    conn.commit()
    conn.close()
    dsb.main([str(tmp_path / "t.db")])
    out = capsys.readouterr().out
    assert "orders linked to new decisions : 2" in out
    assert "orders per new trade           : 2.00" in out


def test_return_pct_anomaly_flagged_for_new_trades(tmp_path, capsys):
    conn = _make_db(tmp_path / "t.db")
    _seed_old(conn, 24)
    _trade(conn, "n1", "SOL/USD", -0.0376, "2026-10-25 00:00:00", return_pct=-70218.31)
    conn.commit()
    conn.close()
    dsb.main([str(tmp_path / "t.db")])
    out = capsys.readouterr().out
    assert "-70218.31%" in out
    assert "bad denominator, not a real loss" in out


def test_json_output_written(tmp_path):
    conn = _make_db(tmp_path / "t.db")
    _seed_old(conn, 24)
    _trade(conn, "n1", "ETH/USD", -0.40, "2026-10-25 00:00:00")
    conn.commit()
    conn.close()
    out_json = tmp_path / "out.json"
    dsb.main([str(tmp_path / "t.db"), "--json", str(out_json)])
    data = json.loads(out_json.read_text())
    assert data["new"]["trades"] == 1
    assert data["old"]["trades"] == 24
    assert data["new_by_symbol"]["ETH/USD"]["net"] == pytest.approx(-0.40)


def test_readonly_open_does_not_create_missing_file(tmp_path):
    missing = tmp_path / "nope.db"
    with pytest.raises(sqlite3.OperationalError):
        conn = dsb._ro(str(missing))
        conn.execute("SELECT 1")
    assert not missing.exists()


def test_script_uses_only_stdlib():
    """It must run inside the container with no third-party imports."""
    src = _SCRIPT.read_text()
    for banned in ("import pandas", "import numpy", "import sqlalchemy", "from src"):
        assert banned not in src
