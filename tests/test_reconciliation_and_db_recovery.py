"""Regression tests for startup reconciliation and SQLite corruption recovery.

These paths decide what the bot believes it holds after a crash or restart,
so a regression can close real positions or silently lose trade history.
Each case was a confirmed bug fixed during the 2026-09-21/22 crash-recovery
audits; none had a permanent test until now.
"""
import os

import polars as pl
import pytest

import src.bot as bot
import src.db as db
from src.config import settings


class FakeExchange:
    def __init__(self, positions=None, orders=None, price=100.0, fail_positions=False):
        self.positions = positions or []
        self.orders = orders or []
        self.price = price
        self.fail_positions = fail_positions
        self.submitted = []

    async def get_positions(self, *a, **k):
        if self.fail_positions:
            raise RuntimeError("exchange unavailable at startup")
        return list(self.positions)

    async def get_latest_bar(self, symbol):
        return pl.DataFrame({"close": [self.price]})

    async def get_orders(self, status=None, limit=100):
        return list(self.orders)

    async def create_order(self, symbol, qty, side, **kwargs):
        self.submitted.append((symbol, qty, side))
        return {"id": f"ord-{len(self.submitted)}", "symbol": symbol, "qty": qty, "status": "filled"}


@pytest.fixture
def fresh_db(tmp_path):
    original = settings.DATABASE_URL
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    db._open_snapshot_cache.clear()
    settings.DATABASE_URL = "sqlite:///" + str(tmp_path / "bot.db").replace("\\", "/")
    db.init_db()
    yield
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    db._open_snapshot_cache.clear()
    settings.DATABASE_URL = original


@pytest.fixture
def clean_state():
    before = set(bot._state._background_tasks)
    yield
    for task in list(bot._state._background_tasks - before):
        task.cancel()
    bot._state.reconciliation_incomplete = False
    bot._state._reconciliation_retry_active = False
    bot._state.symbols_with_unresolved_orders = set()


def _open_snapshot(decision_id, symbol, entry=100.0, qty=2.0):
    db.save_decision_snapshot(
        decision_id=decision_id, symbol=symbol, regime="trending", final_action="buy",
        confidence=0.7, size_multiplier=1.0, entry_price=entry, qty=qty, brain_votes={"quant": "buy"},
    )


async def test_positions_fetch_failure_leaves_snapshots_open(fresh_db, clean_state):
    """An exchange outage at startup is an UNKNOWN state, not a flat one:
    every open snapshot used to be closed as a ghost (1 -> 0)."""
    _open_snapshot("snap-1", "BTC/USD")

    await bot.reconcile_open_snapshots(FakeExchange(fail_positions=True))

    assert db.get_open_snapshot("BTC/USD") is not None
    assert bot._state.reconciliation_incomplete is True


async def test_snapshot_read_failure_blocks_entries_instead_of_reading_empty(fresh_db, clean_state, monkeypatch):
    """A DB read error used to look identical to 'nothing is open'."""
    def boom():
        raise RuntimeError("database is locked")
    monkeypatch.setattr(db, "get_all_open_snapshots", boom)

    await bot.reconcile_open_snapshots(FakeExchange())

    assert bot._state.reconciliation_incomplete is True


async def test_stale_open_order_marks_symbol_unresolved(fresh_db, clean_state):
    ex = FakeExchange(orders=[{"id": "o1", "symbol": "ETH/USD", "status": "new", "qty": "1", "filled_qty": "0"}])

    await bot.reconcile_open_snapshots(ex)

    assert "ETHUSD" in bot._state.symbols_with_unresolved_orders


async def test_crash_gap_fill_is_reattached_not_liquidated(fresh_db, clean_state):
    """Position on the exchange, no snapshot, but the order ledger shows the
    bot's own recent fill: re-attach it instead of force-selling it."""
    db.save_order_record(order_id="ord-gap", decision_id="dec-gap", symbol="BTC/USD", side="buy",
                         qty=0.1, filled_qty=0.1, filled_avg_price=123.45, status="filled")
    ex = FakeExchange(positions=[{"symbol": "BTCUSD", "qty": "0.1", "avg_entry_price": 123.45}])

    await bot.reconcile_open_snapshots(ex)

    assert ex.submitted == []
    snap = db.get_open_snapshot("BTC/USD")
    assert snap is not None and snap["entry_price"] == pytest.approx(123.45)


async def test_ghost_snapshot_closes_at_real_fill_not_current_bar(fresh_db, clean_state):
    """Snapshot open, position gone: P&L must come from the ledger's real
    exit fill (150), not the current bar (999)."""
    _open_snapshot("snap-ghost", "ETH/USD", entry=100.0, qty=2.0)
    db.save_order_record(order_id="ord-close", symbol="ETH/USD", side="sell",
                         qty=2.0, filled_qty=2.0, filled_avg_price=150.0, status="filled")

    await bot.reconcile_open_snapshots(FakeExchange(price=999.0))

    with db.get_db_session() as session:
        row = session.query(db.DecisionSnapshot).filter_by(decision_id="snap-ghost").first()
    assert row.status == "closed"
    assert row.realized_pnl == pytest.approx(100.0)
    assert "actual_fill" in row.exit_reason


def test_corrupt_db_file_is_rebuilt_not_crash_looped(tmp_path):
    """Header-level corruption used to fail the connection test before the
    rebuild path was ever reached, retrying then crashing on every start."""
    original = settings.DATABASE_URL
    path = tmp_path / "corrupt.db"
    path.write_bytes(b"not a valid SQLite format 3" + b"\x00" * 4096)
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    settings.DATABASE_URL = "sqlite:///" + str(path).replace("\\", "/")
    try:
        assert db.init_db() is True
        from sqlalchemy import text
        with db.get_engine().connect() as conn:
            assert conn.execute(text("SELECT 1")).scalar() == 1
        assert os.listdir(tmp_path / "corrupt_backups")
    finally:
        if db._engine is not None:
            db._engine.dispose()
        db._engine = None
        db._tables_ensured = False
        settings.DATABASE_URL = original
