"""Tests for scripts/analyze_deployed_trades.py.

Each expectation is pinned to a hand-computed value on a real-schema seeded
database (tables built from the bot's own SQLAlchemy models), so a regression in
the arithmetic fails here rather than silently mis-reporting a live snapshot.
"""
from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "analyze_deployed_trades.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_deployed_trades", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


adt = _load_module()


def _make_db(path: Path) -> sqlite3.Connection:
    """Build the snapshot schema exactly as src/db.py declares it (subset)."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE decision_snapshots (
            decision_id TEXT PRIMARY KEY,
            symbol TEXT, regime TEXT, final_action TEXT,
            entry_price REAL, qty REAL, status TEXT, exit_reason TEXT,
            max_favorable_pct REAL, max_adverse_pct REAL,
            realized_pnl REAL, return_pct REAL, holding_period_sec REAL,
            created_at TEXT, closed_at TEXT
        );
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY, decision_id TEXT, symbol TEXT, side TEXT,
            qty REAL, filled_qty REAL, filled_avg_price REAL, commission REAL,
            status TEXT, type TEXT
        );
        CREATE TABLE shadow_trades (
            trade_id TEXT PRIMARY KEY, candidate_name TEXT, symbol TEXT, side TEXT,
            qty REAL, entry_price REAL, exit_price REAL, status TEXT, realized_pnl REAL
        );
        """
    )
    return conn


def _snap(conn, did, symbol, pnl, *, regime="sideways", reason="max_hold_time_exceeded",
          status="closed", mfe=0.0, mae=0.0, closed_at="2026-10-10T00:00:00"):
    conn.execute(
        "INSERT INTO decision_snapshots (decision_id, symbol, regime, final_action, "
        "entry_price, qty, status, exit_reason, max_favorable_pct, max_adverse_pct, "
        "realized_pnl, return_pct, holding_period_sec, created_at, closed_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (did, symbol, regime, "buy", 100.0, 1.0, status, reason, mfe, mae, pnl, 0.0, 60.0,
         "2026-10-10T00:00:00", closed_at),
    )


def test_profit_factor_matches_hand_computation(tmp_path):
    """Gross wins / |gross losses| = 23/42 (hand-computed)."""
    assert adt._profit_factor([10.0, 13.0, -20.0, -22.0]) == pytest.approx(23.0 / 42.0)


def test_profit_factor_none_when_no_losses():
    assert adt._profit_factor([5.0, 7.0]) is None


def test_max_drawdown_uses_running_peak(tmp_path):
    """Peak 12 -> trough 6 is -50%, NOT -40% (that belongs to a peak of 10)."""
    assert adt._max_drawdown_pct([10.0, 12.0, 6.0, 8.0]) == pytest.approx(-50.0)
    assert adt._max_drawdown_pct([10.0, 6.0, 8.0]) == pytest.approx(-40.0)


def test_net_and_pf_end_to_end(tmp_path):
    """Two symbols; overall net -19, PF 23/42, per-symbol buckets correct."""
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    _snap(conn, "d1", "BTC/USD", 10.0, regime="low volatility")
    _snap(conn, "d2", "BTC/USD", 13.0, regime="low volatility")
    _snap(conn, "d3", "ETH/USD", -20.0, regime="sideways")
    _snap(conn, "d4", "ETH/USD", -22.0, regime="sideways")
    conn.commit()
    conn.close()

    conn = adt.connect_readonly(str(db))
    try:
        res = adt.analyze(conn)
    finally:
        conn.close()

    assert res["closed_trades"] == 4
    assert res["overall"]["net_pnl"] == pytest.approx(-19.0)
    assert res["overall"]["profit_factor"] == pytest.approx(23.0 / 42.0)
    assert res["overall"]["win_rate"] == pytest.approx(50.0)
    assert res["by_symbol"]["BTC/USD"]["net_pnl"] == pytest.approx(23.0)
    assert res["by_symbol"]["ETH/USD"]["net_pnl"] == pytest.approx(-42.0)


def test_readonly_connection_cannot_write(tmp_path):
    """The analyzer's connection must refuse writes."""
    db = tmp_path / "snap.db"
    _make_db(db).close()
    conn = adt.connect_readonly(str(db))
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO decision_snapshots (decision_id) VALUES ('x')")
    finally:
        conn.close()


def test_readonly_open_of_missing_file_does_not_create_it(tmp_path):
    missing = tmp_path / "nope.db"
    with pytest.raises(sqlite3.OperationalError):
        adt.connect_readonly(str(missing))
    assert not missing.exists()


def test_open_snapshots_excluded_from_closed_stats(tmp_path):
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    _snap(conn, "d1", "BTC/USD", 5.0)
    _snap(conn, "d2", "BTC/USD", 999.0, status="open")
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        res = adt.analyze(conn)
    finally:
        conn.close()
    assert res["closed_trades"] == 1
    assert res["open_trades"] == 1
    assert res["overall"]["net_pnl"] == pytest.approx(5.0)


def test_excursion_fields_counted_when_zero(tmp_path):
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    _snap(conn, "d1", "BTC/USD", -1.0, mfe=0.0, mae=0.0)
    _snap(conn, "d2", "BTC/USD", -1.0, mfe=2.5, mae=1.0)
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        res = adt.analyze(conn)
    finally:
        conn.close()
    assert res["excursion_fields"]["zero_mfe"] == 1
    assert res["excursion_fields"]["zero_mae"] == 1
    assert res["excursion_fields"]["n"] == 2


def test_order_join_coverage_classifies_gaps(tmp_path):
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    _snap(conn, "d1", "BTC/USD", 1.0)
    conn.executemany(
        "INSERT INTO orders (order_id, decision_id, symbol, side, qty, filled_qty, "
        "filled_avg_price, commission, status, type) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("o1", "d1", "BTC/USD", "buy", 1, 1, 100.0, 0.25, "filled", "market"),   # joinable
            ("o2", None, "BTC/USD", "sell", 1, 1, 101.0, 0.25, "filled", "market"),  # missing id
            ("o3", "ghost", "ETH/USD", "buy", 1, 1, 3000.0, 0.0, "filled", "market"),  # orphan
            ("o4", "d1", "BTC/USD", "sell", 1, 0, 0.0, 0.0, "new", "market"),        # not filled
        ],
    )
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        res = adt.analyze(conn)
    finally:
        conn.close()
    o = res["orders"]
    assert o["filled"] == 3
    assert o["joinable_to_decision"] == 1
    assert o["filled_missing_decision_id"] == 1
    assert o["filled_orphan_decision_id"] == 1
    assert o["recorded_commission_sum"] == pytest.approx(0.5)
    assert o["filled_with_nonzero_commission"] == 2


def test_shadow_trades_summary(tmp_path):
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    conn.executemany(
        "INSERT INTO shadow_trades (trade_id, candidate_name, symbol, side, qty, "
        "entry_price, exit_price, status, realized_pnl) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            ("s1", "Bot_A", "BTC/USD", "buy", 1.0, 80000.0, 81600.0, "closed", 1600.0),
            ("s2", "Bot_B", "ETH/USD", "buy", 1.0, 3000.0, 2940.0, "closed", -60.0),
        ],
    )
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        res = adt.analyze(conn)
    finally:
        conn.close()
    assert res["shadow_trades"]["total"] == 2
    assert res["shadow_trades"]["net_pnl"] == pytest.approx(1540.0)
    assert res["shadow_trades"]["distinct_qty_values"] == [1.0]


def test_missing_tables_are_tolerated(tmp_path):
    """A snapshot with no orders/shadow_trades must not raise."""
    db = tmp_path / "snap.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE decision_snapshots (decision_id TEXT PRIMARY KEY, symbol TEXT, "
        "regime TEXT, final_action TEXT, entry_price REAL, qty REAL, status TEXT, "
        "exit_reason TEXT, max_favorable_pct REAL, max_adverse_pct REAL, "
        "realized_pnl REAL, return_pct REAL, holding_period_sec REAL, created_at TEXT, closed_at TEXT)"
    )
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        res = adt.analyze(conn)
    finally:
        conn.close()
    assert res["orders"]["filled"] == 0
    assert res["shadow_trades"]["total"] == 0


def test_render_contains_key_lines(tmp_path):
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    _snap(conn, "d1", "BTC/USD", -2.0)
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        out = adt.render(adt.analyze(conn))
    finally:
        conn.close()
    assert "OVERALL" in out
    assert "BY SYMBOL" in out
    assert "EXCURSION FIELDS" in out
    assert "SHADOW TRADES" in out
    assert "RETURN_PCT ANOMALIES" in out


# --- Cases the handoff explicitly requires (Step 3 / Step 4 item 11) ---


def test_all_winners_has_no_profit_factor_and_positive_net(tmp_path):
    """All-winning: PF undefined (no losses), net positive, losses=0."""
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    _snap(conn, "d1", "BTC/USD", 4.0)
    _snap(conn, "d2", "BTC/USD", 6.0)
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        res = adt.analyze(conn)
    finally:
        conn.close()
    assert res["overall"]["profit_factor"] is None
    assert res["overall"]["net_pnl"] == pytest.approx(10.0)
    assert res["overall"]["losses"] == 0
    assert res["overall"]["win_rate"] == pytest.approx(100.0)


def test_all_losers_has_zero_profit_factor(tmp_path):
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    _snap(conn, "d1", "ETH/USD", -3.0)
    _snap(conn, "d2", "ETH/USD", -7.0)
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        res = adt.analyze(conn)
    finally:
        conn.close()
    assert res["overall"]["profit_factor"] == pytest.approx(0.0)
    assert res["overall"]["net_pnl"] == pytest.approx(-10.0)


def test_zero_trades_does_not_crash(tmp_path):
    """Empty decision_snapshots: no division by zero, PF None, net 0."""
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        res = adt.analyze(conn)
        out = adt.render(res)
    finally:
        conn.close()
    assert res["closed_trades"] == 0
    assert res["overall"]["profit_factor"] is None
    assert res["overall"]["net_pnl"] == 0.0
    assert "OVERALL" in out


def test_return_pct_magnitude_anomaly_flagged(tmp_path):
    """The -70,218% / -804% shape must be flagged, with implied notional shown."""
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    _snap(conn, "d1", "SOL/USD", -0.04)
    conn.execute("UPDATE decision_snapshots SET return_pct=? WHERE decision_id='d1'", (-70218.0,))
    _snap(conn, "d2", "ETH/USD", -0.04)
    conn.execute("UPDATE decision_snapshots SET return_pct=? WHERE decision_id='d2'", (-804.76,))
    _snap(conn, "d3", "BTC/USD", -0.50)
    conn.execute("UPDATE decision_snapshots SET return_pct=? WHERE decision_id='d3'", (-0.50,))
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        res = adt.analyze(conn)
    finally:
        conn.close()
    an = res["return_pct_anomalies"]
    assert an["flagged"] == 2  # d1 and d2; d3 (-0.5%) is sane
    ids = {e["decision_id"] for e in an["examples"]}
    assert ids == {"d1", "d2"}
    # implied notional exposes the near-zero denominator instead of guessing a cause
    sol = next(e for e in an["examples"] if e["decision_id"] == "d1")
    assert sol["implied_notional"] == pytest.approx(-0.04 / (-702.18), rel=1e-6)
    assert sol["reason"] == "magnitude_out_of_bounds"


def test_return_pct_zero_with_nonzero_pnl_flagged(tmp_path):
    """Legacy-row shape: return_pct 0.0 but realized_pnl != 0."""
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    _snap(conn, "d1", "BTC/USD", -3.0)  # _snap writes return_pct=0.0
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        res = adt.analyze(conn)
    finally:
        conn.close()
    an = res["return_pct_anomalies"]
    assert an["flagged"] == 1
    assert an["examples"][0]["reason"] == "zero_return_with_nonzero_pnl"


def test_return_pct_zero_with_zero_pnl_not_flagged(tmp_path):
    """A genuine flat trade is NOT an anomaly."""
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    _snap(conn, "d1", "BTC/USD", 0.0)
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        res = adt.analyze(conn)
    finally:
        conn.close()
    assert res["return_pct_anomalies"]["flagged"] == 0


def test_partial_fill_orders_do_not_inflate_join_counts(tmp_path):
    """Unfilled/partial rows must not count as filled joins."""
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    _snap(conn, "d1", "BTC/USD", 1.0)
    conn.executemany(
        "INSERT INTO orders (order_id, decision_id, symbol, side, qty, filled_qty, "
        "filled_avg_price, commission, status, type) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("o1", "d1", "BTC/USD", "buy", 1.0, 1.0, 100.0, 0.25, "filled", "market"),
            # partial fill, still status filled, but filled_qty < qty
            ("o2", "d1", "BTC/USD", "buy", 2.0, 0.5, 100.0, 0.125, "filled", "market"),
            ("o3", "d1", "BTC/USD", "buy", 1.0, 0.0, 0.0, 0.0, "canceled", "market"),
        ],
    )
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        res = adt.analyze(conn)
    finally:
        conn.close()
    # Only the two 'filled' rows count; the canceled row is excluded entirely.
    assert res["orders"]["filled"] == 2
    assert res["orders"]["joinable_to_decision"] == 2
    assert res["orders"]["recorded_commission_sum"] == pytest.approx(0.375)


def test_duplicate_order_ids_are_reported_by_row_count(tmp_path):
    """The analyzer counts rows, not distinct orders -- duplicates stay visible."""
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    _snap(conn, "d1", "BTC/USD", 1.0)
    conn.executemany(
        "INSERT INTO orders (order_id, decision_id, symbol, side, qty, filled_qty, "
        "filled_avg_price, commission, status, type) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("o1", "d1", "BTC/USD", "buy", 1.0, 1.0, 100.0, 0.25, "filled", "market"),
            ("o2", "d1", "BTC/USD", "buy", 1.0, 1.0, 100.0, 0.25, "filled", "market"),
        ],
    )
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        res = adt.analyze(conn)
    finally:
        conn.close()
    assert res["orders"]["filled"] == 2  # two rows, even if they look like one fill


# --- Trade-by-trade reconciliation (handoff §3) ---


def test_reconcile_reconstructs_exit_price_and_return(tmp_path):
    """Implied exit price and recalculated return from named denominator."""
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    # entry 100, qty 2 => notional 200; pnl +10 => implied exit 105, ret +5%
    conn.execute(
        "INSERT INTO decision_snapshots (decision_id, symbol, regime, final_action, "
        "entry_price, qty, status, exit_reason, max_favorable_pct, max_adverse_pct, "
        "realized_pnl, return_pct, holding_period_sec, created_at, closed_at) "
        "VALUES ('d1','BTC/USD','bull','buy',100.0,2.0,'closed','trailing_stop',3.0,1.0,"
        "10.0,5.0,60.0,'2026-10-10T00:00:00','2026-10-10T01:00:00')"
    )
    conn.execute(
        "INSERT INTO orders (order_id, decision_id, symbol, side, qty, filled_qty, "
        "filled_avg_price, commission, status, type) VALUES "
        "('e1','d1','BTC/USD','buy',2.0,2.0,100.0,0.50,'filled','market'),"
        "('x1','d1','BTC/USD','sell',2.0,2.0,105.0,0.525,'filled','market')"
    )
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        rc = adt.reconcile_trades(conn)
    finally:
        conn.close()
    assert rc["n"] == 1
    t = rc["trades"][0]
    assert t["implied_exit_price"] == pytest.approx(105.0)
    assert t["recalculated_return_pct"] == pytest.approx(5.0)
    assert t["return_pct_denominator"] == pytest.approx(200.0)
    assert t["recorded_return_pct"] == pytest.approx(5.0)  # agrees here
    assert t["entry_order_ids"] == ["e1"]
    assert t["exit_order_ids"] == ["x1"]
    assert t["recorded_commission"] == pytest.approx(1.025)
    assert t["unreconciled"] == []
    assert t["mfe_mae_trustworthy"] is True  # trailing_stop path


def test_reconcile_flags_return_pct_disagreement(tmp_path):
    """A recorded return_pct that disagrees with the named denominator shows up."""
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    # notional 200, pnl -0.04 => true ret -0.02%; recorded -70218% must differ
    conn.execute(
        "INSERT INTO decision_snapshots (decision_id, symbol, regime, final_action, "
        "entry_price, qty, status, exit_reason, max_favorable_pct, max_adverse_pct, "
        "realized_pnl, return_pct, holding_period_sec, created_at, closed_at) "
        "VALUES ('d1','SOL/USD','sideways','buy',100.0,2.0,'closed','max_hold_time_exceeded',"
        "0.0,0.0,-0.04,-70218.0,60.0,'2026-10-10T00:00:00','2026-10-10T01:00:00')"
    )
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        rc = adt.reconcile_trades(conn)
    finally:
        conn.close()
    t = rc["trades"][0]
    assert t["recalculated_return_pct"] == pytest.approx(-0.02)
    assert t["recorded_return_pct"] == pytest.approx(-70218.0)
    assert t["mfe_mae_trustworthy"] is False  # max_hold path


def test_reconcile_marks_missing_orders_and_bad_entry(tmp_path):
    """A row with no linked orders and a zero entry price is marked, not guessed."""
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    conn.execute(
        "INSERT INTO decision_snapshots (decision_id, symbol, regime, final_action, "
        "entry_price, qty, status, exit_reason, max_favorable_pct, max_adverse_pct, "
        "realized_pnl, return_pct, holding_period_sec, created_at, closed_at) "
        "VALUES ('d1','ETH/USD','sideways','buy',0.0,0.0,'closed','max_hold_time_exceeded',"
        "0.0,0.0,0.0,0.0,60.0,'2026-10-10T00:00:00','2026-10-10T01:00:00')"
    )
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        rc = adt.reconcile_trades(conn)
    finally:
        conn.close()
    t = rc["trades"][0]
    assert t["implied_exit_price"] is None       # qty == 0, cannot reconstruct
    assert t["recalculated_return_pct"] is None  # notional == 0
    assert set(t["unreconciled"]) == {"entry_price", "qty", "no_orders_linked_by_decision_id"}
    assert rc["with_gaps"] == 1


def test_mfe_mae_trust_rule_per_exit_path():
    """Only the trailing-stop path can record real excursions."""
    assert adt._mfe_mae_trustworthy("trailing_stop") is True
    assert adt._mfe_mae_trustworthy("[MOMENTUM] Momentum: Loss of bullish momentum") is False
    assert adt._mfe_mae_trustworthy("max_hold_time_exceeded") is False
    assert adt._mfe_mae_trustworthy("Trend Following: Macro trend flipped bearish") is False
    assert adt._mfe_mae_trustworthy("reconciled_after_restart:actual_fill") is False
    assert adt._mfe_mae_trustworthy("killswitch_liquidation") is False


def test_reconcile_short_side_implied_exit(tmp_path):
    """Short side: implied exit price is below entry when the trade profits."""
    db = tmp_path / "snap.db"
    conn = _make_db(db)
    conn.execute(
        "INSERT INTO decision_snapshots (decision_id, symbol, regime, final_action, "
        "entry_price, qty, status, exit_reason, max_favorable_pct, max_adverse_pct, "
        "realized_pnl, return_pct, holding_period_sec, created_at, closed_at) "
        "VALUES ('d1','BTC/USD','bear','sell',100.0,1.0,'closed','trailing_stop',0.0,0.0,"
        "5.0,5.0,60.0,'2026-10-10T00:00:00','2026-10-10T01:00:00')"
    )
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        rc = adt.reconcile_trades(conn)
    finally:
        conn.close()
    assert rc["trades"][0]["implied_exit_price"] == pytest.approx(95.0)
    assert rc["trades"][0]["recalculated_return_pct"] == pytest.approx(5.0)


# ── slippage read path (2026-10-25) ────────────────────────────────────────
#
# The instrumentation columns (mid_at_submit/spread_at_submit/quote_ts/fee_source)
# exist only on snapshots written after the 2026-10-25 work, so the analyzer must
# tolerate their absence. Every expected value below is hand-computed.


def _make_db_with_instrumentation(path: Path) -> sqlite3.Connection:
    """Snapshot schema INCLUDING the 2026-10-25 instrumentation columns."""
    conn = _make_db(path)
    conn.executescript(
        """
        ALTER TABLE orders ADD COLUMN fee_source TEXT;
        ALTER TABLE orders ADD COLUMN mid_at_submit REAL;
        ALTER TABLE orders ADD COLUMN spread_at_submit REAL;
        ALTER TABLE orders ADD COLUMN quote_ts TEXT;
        ALTER TABLE orders ADD COLUMN submitted_at TEXT;
        ALTER TABLE orders ADD COLUMN filled_at TEXT;
        """
    )
    return conn


def _order(conn, oid, side, fill, *, mid=None, spread=None, quote_ts=None,
           status="filled", submitted_at="2026-10-10T00:00:00",
           filled_at="2026-10-10T00:00:00", symbol="BTC/USD"):
    conn.execute(
        "INSERT INTO orders (order_id, decision_id, symbol, side, qty, filled_qty, "
        "filled_avg_price, commission, status, type, fee_source, mid_at_submit, "
        "spread_at_submit, quote_ts, submitted_at, filled_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (oid, "d1", symbol, side, 1.0, 1.0, fill, 0.0, status, "market", None,
         mid, spread, quote_ts, submitted_at, filled_at),
    )


def test_slippage_buy_positive_when_paying_above_mid():
    """BUY fill 101 vs mid 100 => (101-100)/100*1e4 = +100 bps (a cost)."""
    assert adt._slippage_bps("buy", 101.0, 100.0) == pytest.approx(100.0)


def test_slippage_sell_positive_when_receiving_below_mid():
    """SELL fill 99 vs mid 100 => (100-99)/100*1e4 = +100 bps (a cost).

    The sign flip between sides is the easiest thing to get wrong, so it is
    pinned directly.
    """
    assert adt._slippage_bps("sell", 99.0, 100.0) == pytest.approx(100.0)


def test_slippage_negative_when_execution_beats_the_mid():
    """Both sides go NEGATIVE when the fill is better than mid."""
    assert adt._slippage_bps("buy", 99.0, 100.0) == pytest.approx(-100.0)
    assert adt._slippage_bps("sell", 101.0, 100.0) == pytest.approx(-100.0)


def test_half_spread_and_excess_decomposition_hand_computed():
    """mid=100, spread=1 => half_spread = (1/100)/2*1e4 = 50 bps.

    BUY fill 101 => slippage 100 bps => excess = 100 - 50 = 50 bps.
    """
    assert adt._half_spread_bps(1.0, 100.0) == pytest.approx(50.0)
    assert adt._slippage_bps("buy", 101.0, 100.0) - adt._half_spread_bps(1.0, 100.0) == pytest.approx(50.0)


def test_slippage_bps_rejects_nonpositive_mid():
    with pytest.raises(ValueError):
        adt._slippage_bps("buy", 100.0, 0.0)
    with pytest.raises(ValueError):
        adt._half_spread_bps(1.0, 0.0)


def test_analyze_slippage_end_to_end_hand_computed(tmp_path):
    """Two usable BUY rows, one excluded; means/median hand-computed.

    Row A: mid=100 spread=1 fill=101 -> slip=100, half=50, excess=50
    Row B: mid=100 spread=3 fill=100 -> slip=0,   half=150, excess=-150
    overall slippage mean = (100+0)/2 = 50
    overall half mean     = (50+150)/2 = 100
    overall excess mean   = (50-150)/2 = -50
    excess median (2 values) = (-150+50)/2 = -50
    share_excess_positive = 1/2 = 0.5
    """
    db = tmp_path / "snap.db"
    conn = _make_db_with_instrumentation(db)
    _order(conn, "o1", "buy", 101.0, mid=100.0, spread=1.0, quote_ts="2026-10-10T00:00:00")
    _order(conn, "o2", "buy", 100.0, mid=100.0, spread=3.0, quote_ts="2026-10-10T00:00:00")
    _order(conn, "o3", "buy", 101.0, mid=None, spread=1.0, quote_ts="2026-10-10T00:00:00")  # no_quote
    conn.commit()
    conn.close()

    conn = adt.connect_readonly(str(db))
    try:
        orders, missing = adt._orders(conn)
        sl = adt.analyze_slippage(orders, missing)
    finally:
        conn.close()

    assert sl["available"] is True
    assert missing == []
    assert sl["rows_considered"] == 2
    assert sl["excluded"]["no_quote"] == 1
    o = sl["overall"]
    assert o["n"] == 2
    assert o["slippage_bps_mean"] == pytest.approx(50.0)
    assert o["half_spread_bps_mean"] == pytest.approx(100.0)
    assert o["excess_slippage_bps_mean"] == pytest.approx(-50.0)
    assert o["excess_slippage_bps_median"] == pytest.approx(-50.0)
    assert o["share_excess_positive"] == pytest.approx(0.5)
    assert sl["per_symbol"]["BTC/USD"]["n"] == 2


def test_slippage_excludes_stale_quote_and_no_quote_ts(tmp_path):
    """A quote >5s from the fill, and a missing quote_ts, are both excluded."""
    db = tmp_path / "snap.db"
    conn = _make_db_with_instrumentation(db)
    _order(conn, "o1", "buy", 101.0, mid=100.0, spread=1.0,
           quote_ts="2026-10-10T00:00:00", filled_at="2026-10-10T00:00:00")   # fresh -> counted
    _order(conn, "o2", "buy", 101.0, mid=100.0, spread=1.0,
           quote_ts="2026-10-10T00:00:00", filled_at="2026-10-10T00:00:30")   # 30s -> stale
    _order(conn, "o3", "buy", 101.0, mid=100.0, spread=1.0, quote_ts=None)    # no_quote_ts
    _order(conn, "o4", "buy", 101.0, mid=100.0, spread=0.0,
           quote_ts="2026-10-10T00:00:00")                                    # bad_spread
    conn.commit()
    conn.close()

    conn = adt.connect_readonly(str(db))
    try:
        orders, missing = adt._orders(conn)
        sl = adt.analyze_slippage(orders, missing)
    finally:
        conn.close()

    assert sl["rows_considered"] == 1
    assert sl["excluded"]["stale_quote"] == 1
    assert sl["excluded"]["no_quote_ts"] == 1
    assert sl["excluded"]["bad_spread"] == 1


def test_slippage_unavailable_on_legacy_schema(tmp_path):
    """A pre-2026-10-25 snapshot must NOT crash; slippage is reported unavailable."""
    db = tmp_path / "snap.db"
    _make_db(db).close()          # legacy schema: no instrumentation columns
    conn = adt.connect_readonly(str(db))
    try:
        orders, missing = adt._orders(conn)
        sl = adt.analyze_slippage(orders, missing)
        res = adt.analyze(conn)
    finally:
        conn.close()

    assert set(missing) == {
        "mid_at_submit", "spread_at_submit", "quote_ts", "fee_source",
        "submitted_at", "filled_at",
    }
    assert sl["available"] is False
    assert sl["rows_considered"] == 0
    # analyze() must complete and carry the section through.
    assert res["slippage"]["available"] is False
    assert res["orders"]["missing_instrumentation_columns"] == missing


def test_slippage_render_reports_missing_columns(tmp_path):
    db = tmp_path / "snap.db"
    _make_db(db).close()
    conn = adt.connect_readonly(str(db))
    try:
        res = adt.analyze(conn)
    finally:
        conn.close()
    text = adt.render(res)
    assert "SLIPPAGE" in text
    assert "not captured on this snapshot" in text
    assert "mid_at_submit" in text


def test_slippage_render_shows_stats_when_available(tmp_path):
    db = tmp_path / "snap.db"
    conn = _make_db_with_instrumentation(db)
    _order(conn, "o1", "buy", 101.0, mid=100.0, spread=1.0, quote_ts="2026-10-10T00:00:00")
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        text = adt.render(adt.analyze(conn))
    finally:
        conn.close()
    assert "rows_considered=1" in text
    assert "excess_mean=50.00bps" in text      # 100 - 50


def test_slippage_median_even_count_is_mean_of_middle_pair():
    """Median of [10, 20] = 15 (the documented even-count convention)."""
    rows = [
        {"slippage_bps": 10.0, "half_spread_bps": 0.0, "excess_slippage_bps": 10.0},
        {"slippage_bps": 20.0, "half_spread_bps": 0.0, "excess_slippage_bps": 20.0},
    ]
    assert adt._slippage_stats(rows)["slippage_bps_median"] == pytest.approx(15.0)


def test_slippage_ignores_unfilled_orders(tmp_path):
    """Only filled orders contribute; a 'new' order with a quote is skipped."""
    db = tmp_path / "snap.db"
    conn = _make_db_with_instrumentation(db)
    _order(conn, "o1", "buy", 101.0, mid=100.0, spread=1.0, quote_ts="2026-10-10T00:00:00")
    _order(conn, "o2", "buy", 101.0, mid=100.0, spread=1.0,
           quote_ts="2026-10-10T00:00:00", status="new")
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        orders, missing = adt._orders(conn)
        sl = adt.analyze_slippage(orders, missing)
    finally:
        conn.close()
    assert sl["rows_considered"] == 1
    assert sl["excluded"]["no_quote"] == 0     # unfilled is skipped, not "no_quote"


def test_slippage_excludes_rows_when_age_cannot_be_verified(tmp_path):
    """Quote columns present but no timestamp => freshness unknown => excluded.

    This is the fail-closed branch: without a fill/submit time the analyzer
    cannot tell a fresh quote from a stale one, so it refuses to score the row
    rather than assuming it is fresh.
    """
    db = tmp_path / "snap.db"
    conn = _make_db_with_instrumentation(db)
    _order(conn, "o1", "buy", 101.0, mid=100.0, spread=1.0,
           quote_ts="2026-10-10T00:00:00", filled_at=None, submitted_at=None)
    conn.commit()
    conn.close()
    conn = adt.connect_readonly(str(db))
    try:
        orders, missing = adt._orders(conn)
        sl = adt.analyze_slippage(orders, missing)
    finally:
        conn.close()
    assert sl["rows_considered"] == 0
    assert sl["excluded"]["age_unverified"] == 1
