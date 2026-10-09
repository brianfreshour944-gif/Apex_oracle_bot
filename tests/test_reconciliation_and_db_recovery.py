"""Regression tests for startup reconciliation and SQLite corruption recovery.

These paths decide what the bot believes it holds after a crash or restart,
so a regression can close real positions or silently lose trade history.
Each case was a confirmed bug fixed during the 2026-09-21/22 crash-recovery
audits; none had a permanent test until now.
"""
import datetime
import os

import polars as pl
import pytest

import src.bot as bot
import src.db as db
from src.config import settings


class FakeExchange:
    def __init__(self, positions=None, orders=None, price=100.0, fail_positions=False, orders_by_id=None):
        self.positions = positions or []
        self.orders = orders or []
        self.orders_by_id = orders_by_id or {}
        self.price = price
        self.fail_positions = fail_positions
        self.submitted = []
        self.cancelled = []

    async def get_positions(self, *a, **k):
        if self.fail_positions:
            raise RuntimeError("exchange unavailable at startup")
        return list(self.positions)

    async def get_latest_bar(self, symbol):
        return pl.DataFrame({"close": [self.price]})

    async def get_orders(self, status=None, limit=100):
        return list(self.orders)

    async def get_order(self, order_id):
        return (self.orders_by_id or {}).get(str(order_id))

    async def create_order(self, symbol, qty, side, **kwargs):
        self.submitted.append((symbol, qty, side))
        return {"id": f"ord-{len(self.submitted)}", "symbol": symbol, "qty": qty, "status": "filled"}

    async def cancel_order(self, order_id, *a, **k):
        self.cancelled.append(order_id)
        return True


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


def test_corruption_after_engine_exists_is_rebuilt_and_reported(tmp_path):
    """The raw pre-probe only runs on the first init_db() call (while _engine
    is None). If the file is corrupted *after* the engine exists -- e.g. the
    WAL sidecars were poisoned mid-run -- a fresh pooled connection fails with
    "file is not a database" and, without a fallback, init_db() retried 5x and
    gave up. It must instead move the file aside, rebuild, and return True so
    bot.py alerts that live data was moved aside (the tenacity @retry would
    otherwise swallow the signal and return False)."""
    original = settings.DATABASE_URL
    path = tmp_path / "bot.db"
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    settings.DATABASE_URL = "sqlite:///" + str(path).replace("\\", "/")
    try:
        assert db.init_db() is False  # healthy, engine now cached
        # Corrupt in place and drain the pool so the next connect() opens a
        # brand-new connection to the corrupt file (pre-probe is skipped
        # because _engine is not None).
        db.get_engine().dispose()
        path.write_bytes(b"NOT a SQLite format 3 file" + b"\x00" * 8192)
        for suffix in ("-wal", "-shm"):
            try:
                os.remove(str(path) + suffix)
            except FileNotFoundError:
                pass
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


@pytest.mark.asyncio
async def test_restart_readopts_stop_for_held_position(fresh_db, clean_state, monkeypatch):
    """A restart loses the in-memory stop registry; a resting stop_limit sell
    for a still-held symbol must be re-adopted so a later exit cancels it."""
    monkeypatch.setattr(settings, "PROTECTIVE_STOPS_ENABLED", True)
    bot._state.protective_stops = {}
    ex = FakeExchange(
        positions=[{"symbol": "BTCUSD", "qty": "0.1", "avg_entry_price": 100.0}],
        orders=[{"id": "stop-held", "symbol": "BTC/USD", "type": "stop_limit",
                 "side": "sell", "status": "new", "qty": "0.1"}],
    )

    await bot.reconcile_open_snapshots(ex)

    assert bot._state.protective_stops.get("BTCUSD") == "stop-held"
    assert ex.cancelled == []


@pytest.mark.asyncio
async def test_restart_cancels_dangling_stop_without_position(fresh_db, clean_state, monkeypatch):
    """A stop_limit sell whose position was closed while the bot was down is
    dangling -- cancel it rather than leaving a resting sell forever."""
    monkeypatch.setattr(settings, "PROTECTIVE_STOPS_ENABLED", True)
    bot._state.protective_stops = {}
    ex = FakeExchange(
        positions=[],
        orders=[{"id": "stop-orphan", "symbol": "ETH/USD", "type": "stop_limit",
                 "side": "sell", "status": "new", "qty": "1"}],
    )

    await bot.reconcile_open_snapshots(ex)

    assert ex.cancelled == ["stop-orphan"]
    assert "ETHUSD" not in bot._state.protective_stops


@pytest.mark.asyncio
async def test_restart_ignores_stops_when_feature_disabled(fresh_db, clean_state, monkeypatch):
    """With the flag OFF the bot never armed stops, so it must not touch
    (or cancel) resting stop_limit orders it did not create."""
    monkeypatch.setattr(settings, "PROTECTIVE_STOPS_ENABLED", False)
    bot._state.protective_stops = {}
    ex = FakeExchange(
        positions=[{"symbol": "BTCUSD", "qty": "0.1"}],
        orders=[{"id": "stop-held", "symbol": "BTC/USD", "type": "stop_limit",
                 "side": "sell", "status": "new", "qty": "0.1"}],
    )

    await bot.reconcile_open_snapshots(ex)

    assert ex.cancelled == []
    assert bot._state.protective_stops == {}


class TestExitFillSelection:
    """Which filled exit order closes the snapshot.

    A symbol can have several filled SELL orders in the ledger (a market exit
    followed by a later re-entry's exit, or a protective stop resting while a
    later market sell fires). Closing the snapshot must use the order that
    actually closed THIS position, or the adaptive learner is trained on a
    fabricated exit price.
    """

    def _sell(self, order_id, price, when, decision_id=None, otype="market"):
        return {
            "order_id": order_id, "side": "sell", "filled_avg_price": price,
            "filled_at": when, "decision_id": decision_id, "type": otype,
        }

    def test_decision_id_match_wins_over_latest(self):
        """The order recorded for the snapshot's decision_id is the exit, even
        when a later re-entry's exit is also present."""
        candidates = [
            self._sell("exit-new", 90.0, "2026-10-07T12:00:00Z"),            # latest
            self._sell("exit-old", 150.0, "2026-10-07T10:00:00Z", decision_id="snap-ghost"),
        ]
        picked = bot._select_exit_fill(candidates, "snap-ghost")
        assert picked["order_id"] == "exit-old"
        assert picked["filled_avg_price"] == pytest.approx(150.0)

    def test_latest_filled_sell_wins_without_decision_id(self):
        """No decision_id correlation available -> the latest filled sell."""
        candidates = [
            self._sell("exit-early", 150.0, "2026-10-07T10:00:00Z"),
            self._sell("exit-late", 120.0, "2026-10-07T11:00:00Z"),
        ]
        assert bot._select_exit_fill(candidates, None)["order_id"] == "exit-late"

    def test_buy_fills_are_never_chosen_as_exit(self):
        """A long position is closed by a SELL; a more recent buy must not be
        mistaken for the exit."""
        candidates = [
            {"order_id": "reentry-buy", "side": "buy", "filled_avg_price": 95.0,
             "filled_at": "2026-10-07T12:00:00Z", "type": "market"},
            self._sell("exit", 150.0, "2026-10-07T10:00:00Z"),
        ]
        assert bot._select_exit_fill(candidates, None)["order_id"] == "exit"

    def test_returns_none_when_no_filled_sell(self):
        assert bot._select_exit_fill([{"order_id": "b", "side": "buy", "filled_avg_price": 1.0}], None) is None

    def test_short_position_exit_uses_buy_fill(self):
        """A short is closed by a BUY; the caller passes exit_side='buy'."""
        candidates = [
            {"order_id": "open-short", "side": "sell", "filled_avg_price": 100.0,
             "filled_at": "2026-10-07T10:00:00Z"},
            {"order_id": "close-short", "side": "buy", "filled_avg_price": 90.0,
             "filled_at": "2026-10-07T11:00:00Z"},
        ]
        picked = bot._select_exit_fill(candidates, None, exit_side="buy")
        assert picked["order_id"] == "close-short"

    def test_client_order_id_match_wins_when_decision_id_absent(self):
        """No ledger row carries the decision_id (its record was lost), but the
        exit order's client_order_id is known: match on that, not recency."""
        candidates = [
            {"order_id": "newer", "side": "sell", "filled_avg_price": 90.0,
             "filled_at": "2026-10-07T12:00:00Z", "client_order_id": "other"},
            {"order_id": "real-exit", "side": "sell", "filled_avg_price": 150.0,
             "filled_at": "2026-10-07T10:00:00Z", "client_order_id": "ETHUSD_ps_abc"},
        ]
        picked = bot._select_exit_fill(candidates, "snap-missing", client_order_id="ETHUSD_ps_abc")
        assert picked["order_id"] == "real-exit"

    def test_decision_id_match_beats_client_order_id(self):
        candidates = [
            {"order_id": "by-decision", "side": "sell", "filled_avg_price": 150.0,
             "filled_at": "2026-10-07T10:00:00Z", "decision_id": "snap-1"},
            {"order_id": "by-coid", "side": "sell", "filled_avg_price": 90.0,
             "filled_at": "2026-10-07T12:00:00Z", "client_order_id": "ETHUSD_ps_abc"},
        ]
        picked = bot._select_exit_fill(candidates, "snap-1", client_order_id="ETHUSD_ps_abc")
        assert picked["order_id"] == "by-decision"


class TestProtectiveStopClientOrderId:
    """The resting stop's client_order_id must be recomputable from the
    snapshot after a restart, so reconcile can match its fill by id."""

    def test_is_deterministic_for_a_decision_id(self):
        a = bot._protstop_client_order_id("ETH/USD", "snap-42")
        b = bot._protstop_client_order_id("ETH/USD", "snap-42")
        assert a == b
        assert a is not None and "snap-42" not in a  # hashed, not embedded

    def test_differs_per_decision(self):
        assert bot._protstop_client_order_id("ETH/USD", "snap-1") != \
            bot._protstop_client_order_id("ETH/USD", "snap-2")

    def test_none_without_a_decision_id(self):
        assert bot._protstop_client_order_id("ETH/USD", None) is None

    def test_within_the_exchange_client_order_id_limit(self):
        """Alpaca rejects client_order_id longer than 48 chars."""
        cid = bot._protstop_client_order_id("LONGSYMBOL/USD", "x" * 200)
        assert len(cid) <= 48


@pytest.mark.asyncio
async def test_ghost_close_uses_correct_exit_when_two_fills_exist(fresh_db, clean_state):
    """End-to-end: with two filled sell orders for one symbol, the ghost-close
    must score at the one linked to the snapshot's decision_id, not the newer
    unrelated fill and not the current bar."""
    _open_snapshot("snap-ghost", "ETH/USD", entry=100.0, qty=2.0)
    # The real exit for this snapshot.
    db.save_order_record(order_id="exit-for-snap", decision_id="snap-ghost", symbol="ETH/USD",
                         side="sell", qty=2.0, filled_qty=2.0, filled_avg_price=150.0,
                         status="filled", type="market")
    # A newer, unrelated filled sell for the same symbol (e.g. a later re-entry
    # that already exited). records[0] would pick this one.
    db.save_order_record(order_id="exit-other", symbol="ETH/USD", side="sell", qty=1.0,
                         filled_qty=1.0, filled_avg_price=90.0, status="filled", type="market")

    await bot.reconcile_open_snapshots(FakeExchange(price=999.0))

    with db.get_db_session() as session:
        row = session.query(db.DecisionSnapshot).filter_by(decision_id="snap-ghost").first()
    assert row.status == "closed"
    # (150 - 100) * 2 = 100, NOT (90-100)*2 = -20 and NOT the 999 bar.
    assert row.realized_pnl == pytest.approx(100.0)
    assert "actual_fill" in row.exit_reason


@pytest.mark.asyncio
async def test_ghost_close_uses_latest_sell_when_no_decision_id_match(fresh_db, clean_state):
    """Two filled sells and NO ledger row carries this snapshot's decision_id:
    fall back to the latest filled sell, not the current bar."""
    import datetime

    _open_snapshot("snap-ghost2", "ETH/USD", entry=100.0, qty=2.0)
    now = datetime.datetime.now(datetime.UTC)
    db.save_order_record(order_id="exit-old", symbol="ETH/USD", side="sell", qty=2.0,
                         filled_qty=2.0, filled_avg_price=140.0, status="filled", type="market",
                         submitted_at=now - datetime.timedelta(minutes=30))
    db.save_order_record(order_id="exit-new", symbol="ETH/USD", side="sell", qty=2.0,
                         filled_qty=2.0, filled_avg_price=130.0, status="filled", type="market",
                         submitted_at=now)

    await bot.reconcile_open_snapshots(FakeExchange(price=999.0))

    with db.get_db_session() as session:
        row = session.query(db.DecisionSnapshot).filter_by(decision_id="snap-ghost2").first()
    assert row.status == "closed"
    # The most recent submit (exit-new, 130) wins -> (130-100)*2 = 60.
    assert row.realized_pnl == pytest.approx(60.0)
    assert "actual_fill" in row.exit_reason


class _ProtStopExchange(FakeExchange):
    """Exchange that accepts a protective stop_limit sell and reports it filled
    at a real exit price."""

    def __init__(self, fill_price=140.0, **kw):
        super().__init__(**kw)
        self.protective = []
        self._fill_price = fill_price

    async def submit_protective_stop(self, symbol, qty, stop_price, limit_price, client_order_id):
        self.protective.append(client_order_id)
        return {
            "id": "prot-1", "client_order_id": client_order_id, "symbol": symbol,
            "qty": qty, "side": "sell", "type": "stop_limit", "status": "filled",
            "filled_qty": qty, "filled_avg_price": self._fill_price,
            "filled_at": "2026-10-07T12:00:00Z",
        }

    async def cancel_order(self, order_id, *a, **k):
        self.cancelled.append(order_id)
        return True


@pytest.mark.asyncio
async def test_protective_stop_fill_is_recorded_with_decision_id_and_filled_at(
    fresh_db, clean_state, monkeypatch
):
    """The resting protective stop is a real exit. Its ledger row must carry
    this position's decision_id and the real filled_at, so reconcile can pick
    it over an unrelated later fill and train the learner on the true price."""
    monkeypatch.setattr(settings, "PROTECTIVE_STOPS_ENABLED", True)
    ex = _ProtStopExchange(fill_price=140.0)

    await bot._arm_protective_stop(ex, "ETH/USD", qty=2.0, stop_price=90.0, decision_id="snap-stop")

    assert ex.protective, "protective stop was never submitted"
    rows = [r for r in db.get_recent_order_records("ETHUSD") if r["type"] == "stop_limit"]
    assert len(rows) == 1
    row = rows[0]
    assert row["decision_id"] == "snap-stop"
    assert row["filled_at"] == datetime.datetime(2026, 10, 7, 12, 0)
    assert row["filled_avg_price"] == pytest.approx(140.0)
    assert row["client_order_id"] == ex.protective[0]


@pytest.mark.asyncio
async def test_protective_stop_wins_over_a_newer_unrelated_exit(fresh_db, clean_state, monkeypatch):
    """End-to-end: the protective stop's fill is the exit that closes the
    snapshot even though a newer unrelated filled sell exists for the symbol."""
    monkeypatch.setattr(settings, "PROTECTIVE_STOPS_ENABLED", True)
    _open_snapshot("snap-stop-e2e", "ETH/USD", entry=100.0, qty=2.0)
    ex = _ProtStopExchange(fill_price=140.0)
    await bot._arm_protective_stop(ex, "ETH/USD", qty=2.0, stop_price=90.0, decision_id="snap-stop-e2e")

    # A newer, unrelated filled sell for the same symbol (a later re-entry).
    db.save_order_record(order_id="exit-other", symbol="ETH/USD", side="sell", qty=1.0,
                         filled_qty=1.0, filled_avg_price=80.0, status="filled", type="market")

    await bot.reconcile_open_snapshots(FakeExchange(price=999.0))

    with db.get_db_session() as session:
        row = session.query(db.DecisionSnapshot).filter_by(decision_id="snap-stop-e2e").first()
    assert row.status == "closed"
    # (140 - 100) * 2 = 80, not (80-100)*2 = -40 and not the 999 bar.
    assert row.realized_pnl == pytest.approx(80.0)
    assert "actual_fill" in row.exit_reason


@pytest.mark.asyncio
async def test_ghost_close_matches_protective_stop_by_client_order_id(fresh_db, clean_state):
    """The stop's ledger row lost its decision_id (a partial write), but its
    client_order_id is the one derived from the snapshot's decision_id -- the
    fill must still be matched by that id, not by the newer unrelated sell."""
    _open_snapshot("snap-coid", "ETH/USD", entry=100.0, qty=2.0)
    coid = bot._protstop_client_order_id("ETH/USD", "snap-coid")
    # Real exit: the protective stop fill, WITHOUT a decision_id on the row.
    db.save_order_record(order_id="prot-fill", symbol="ETH/USD", side="sell", qty=2.0,
                         filled_qty=2.0, filled_avg_price=150.0, status="filled",
                         type="stop_limit", client_order_id=coid)
    # A newer, unrelated filled sell for the same symbol.
    db.save_order_record(order_id="exit-other", symbol="ETH/USD", side="sell", qty=1.0,
                         filled_qty=1.0, filled_avg_price=80.0, status="filled", type="market")

    await bot.reconcile_open_snapshots(FakeExchange(price=999.0))

    with db.get_db_session() as session:
        row = session.query(db.DecisionSnapshot).filter_by(decision_id="snap-coid").first()
    assert row.status == "closed"
    assert row.realized_pnl == pytest.approx(100.0)  # (150 - 100) * 2
    assert "actual_fill" in row.exit_reason


@pytest.mark.asyncio
async def test_resting_stop_fill_is_refreshed_into_the_ledger(fresh_db, clean_state, monkeypatch):
    """A resting stop armed while the position was open fills server-side with
    no bot code in the path, so its ledger row is frozen at arm time
    (status="new", filled_avg_price=0). Reconcile must refresh it from the
    exchange, or a later restart can only estimate the exit from a bar.

    The in-memory ``_state.protective_stops`` registry is deliberately EMPTY
    here: reconcile runs at startup, before any cycle repopulates it, so the
    refresh must find its candidates in the orders ledger instead."""
    monkeypatch.setattr(settings, "PROTECTIVE_STOPS_ENABLED", True)
    _open_snapshot("snap-held", "ETH/USD", entry=100.0, qty=2.0)
    # Arm-time row: still working, no fill yet. Registry is NOT populated.
    db.save_order_record(order_id="stop_live", decision_id="snap-held", symbol="ETH/USD",
                         side="sell", qty=2.0, status="new", type="stop_limit",
                         client_order_id="ETHUSD_ps_x", time_in_force="gtc")
    bot._state.protective_stops.clear()
    # The stop triggered and filled while the position was still held.
    ex = FakeExchange(
        positions=[{"symbol": "ETH/USD", "qty": "2.0", "side": "long", "avg_entry_price": 100.0}],
        orders_by_id={"stop_live": {
            "id": "stop_live", "client_order_id": "ETHUSD_ps_x", "symbol": "ETH/USD",
            "qty": 2.0, "filled_qty": 2.0, "filled_avg_price": 88.0, "status": "filled",
            "type": "stop_limit", "side": "sell", "filled_at": "2026-10-07T12:00:00Z",
        }},
    )

    await bot.reconcile_open_snapshots(ex)

    rows = [r for r in db.get_recent_order_records("ETHUSD") if r["type"] == "stop_limit"]
    assert len(rows) == 1
    assert rows[0]["filled_avg_price"] == pytest.approx(88.0)
    assert rows[0]["status"] == "filled"
    assert rows[0]["decision_id"] == "snap-held"


@pytest.mark.asyncio
async def test_stop_that_flattened_position_is_reconciled_from_its_fill(fresh_db, clean_state, monkeypatch):
    """A stop that filled server-side and flattened the position must still be
    refreshed from the exchange BEFORE the ghost-close loop, so the ghost close
    records exit_price_source == "actual_fill" at the stop's real fill price --
    not the current bar. The registry is EMPTY (the restart case): the refresh
    finds the stop via the orders ledger, so it does not depend on
    _state.protective_stops being repopulated first."""
    monkeypatch.setattr(settings, "PROTECTIVE_STOPS_ENABLED", True)
    _open_snapshot("snap-flat", "ETH/USD", entry=100.0, qty=2.0)
    # Arm-time ledger row: still working, no fill yet.
    db.save_order_record(order_id="stop_flat", decision_id="snap-flat", symbol="ETH/USD",
                         side="sell", qty=2.0, status="new", type="stop_limit",
                         client_order_id="ETHUSD_ps_x", time_in_force="gtc")
    bot._state.protective_stops.clear()
    # The stop triggered and filled at 90, flattening the position -- so the
    # exchange reports NO open position for ETH/USD.
    ex = FakeExchange(
        positions=[],
        price=999.0,
        orders_by_id={"stop_flat": {
            "id": "stop_flat", "client_order_id": "ETHUSD_ps_x", "symbol": "ETH/USD",
            "qty": 2.0, "filled_qty": 2.0, "filled_avg_price": 90.0, "status": "filled",
            "type": "stop_limit", "side": "sell", "filled_at": "2026-10-07T12:00:00Z",
        }},
    )

    await bot.reconcile_open_snapshots(ex)

    with db.get_db_session() as session:
        row = session.query(db.DecisionSnapshot).filter_by(decision_id="snap-flat").first()
    assert row.status == "closed"
    assert row.realized_pnl == pytest.approx(-20.0)  # (90 - 100) * 2
    assert "actual_fill" in row.exit_reason


class _RestingStopExchange(FakeExchange):
    """Fake exchange that models Alpaca's client_order_id uniqueness: an id is
    rejected while a LIVE order holds it. Once that order is terminal
    (canceled/filled/expired/...), the id is free to reuse."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.protective = []              # (client_order_id, order_id) in submit order
        self._by_cid: dict[str, str] = {}   # client_order_id -> order_id
        self._status: dict[str, str] = {}   # order_id -> status
        self._next = 0

    def _live_cid(self, cid):
        oid = self._by_cid.get(cid)
        return oid is not None and self._status.get(oid) in (
            "new", "accepted", "open", "partially_filled",
            "pending", "pending_new", "accepted_for_bidding", "held",
        )

    def _make(self, symbol, qty, client_order_id):
        self._next += 1
        oid = f"stop-{self._next}"
        self._by_cid[client_order_id] = oid
        self._status[oid] = "new"
        self.protective.append((client_order_id, oid))
        return {"id": oid, "symbol": symbol, "qty": qty, "filled_qty": 0.0,
                "filled_avg_price": 0.0, "status": "new", "type": "stop_limit",
                "side": "sell", "client_order_id": client_order_id, "filled_at": None}

    async def submit_protective_stop(self, symbol, qty, stop_price, limit_price, client_order_id):
        # Model the real adapter's recovery contract: adopt a LIVE stop that
        # already holds this id; otherwise (id held by a DEAD order, or free)
        # submit -- re-arming under a fresh id when the id is used up.
        if self._live_cid(client_order_id):
            oid = self._by_cid[client_order_id]
            return {"id": oid, "symbol": symbol, "qty": qty, "filled_qty": 0.0,
                    "filled_avg_price": 0.0, "status": "new", "type": "stop_limit",
                    "side": "sell", "client_order_id": client_order_id, "filled_at": None}
        if client_order_id in self._by_cid:
            client_order_id = f"{client_order_id[:36]}_{self._next + 1}"
        return self._make(symbol, qty, client_order_id)

    async def cancel_order(self, order_id, *a, **k):
        self.cancelled.append(order_id)
        self._status[order_id] = "canceled"
        return True

    def live_ids(self):
        """Order ids still working on the book (not canceled/filled/expired)."""
        return [oid for oid, s in self._status.items() if s == "new"]


@pytest.mark.asyncio
async def test_scale_in_rearm_same_decision_id_replaces_not_duplicates(fresh_db, clean_state, monkeypatch):
    """(a) A scale-in re-arms the stop with the SAME decision_id (the original
    snapshot's). The re-arm must cancel the old stop first and leave exactly
    one resting stop -- never two resting sells against one position."""
    monkeypatch.setattr(settings, "PROTECTIVE_STOPS_ENABLED", True)
    ex = _RestingStopExchange()
    cid = bot._protstop_client_order_id("ETH/USD", "snap-A")

    await bot._arm_protective_stop(ex, "ETH/USD", qty=1.0, stop_price=90.0, decision_id="snap-A")
    first_id = bot._state.protective_stops["ETHUSD"]
    # Scale-in: same decision_id, bigger qty.
    await bot._arm_protective_stop(ex, "ETH/USD", qty=2.0, stop_price=85.0, decision_id="snap-A")

    assert len(ex.protective) == 2
    # The first arm used the deterministic id; after the cancel that id is
    # dead, so the re-arm fell back to a fresh unique id (same decision_id).
    assert ex.protective[0][0] == cid
    assert ex.protective[1][0] != cid
    assert ex.protective[1][0].startswith(cid[:36] + "_")
    assert first_id in ex.cancelled                      # old stop cancelled first
    assert list(bot._state.protective_stops) == ["ETHUSD"]  # exactly one tracked
    assert len(ex.live_ids()) == 1                         # exactly one live resting stop


@pytest.mark.asyncio
async def test_cancel_then_rearm_rests_a_fresh_stop(fresh_db, clean_state, monkeypatch):
    """(b) After a cancel (an exit that didn't complete, or a manual clear),
    re-arming must rest a new stop and track it again."""
    monkeypatch.setattr(settings, "PROTECTIVE_STOPS_ENABLED", True)
    ex = _RestingStopExchange()

    await bot._arm_protective_stop(ex, "ETH/USD", qty=1.0, stop_price=90.0, decision_id="snap-B")
    await bot._cancel_protective_stop(ex, "ETH/USD")
    assert "ETHUSD" not in bot._state.protective_stops
    assert ex.live_ids() == []

    await bot._arm_protective_stop(ex, "ETH/USD", qty=1.0, stop_price=88.0, decision_id="snap-B")
    assert "ETHUSD" in bot._state.protective_stops
    assert len(ex.live_ids()) == 1


@pytest.mark.asyncio
async def test_predeploy_protstop_prefix_still_reconciles(fresh_db, clean_state):
    """(c) A stop armed BEFORE this deploy used the old '_protstop_<ts>'
    client_order_id. Its ledger row still carries the snapshot's decision_id,
    so reconcile must still match it (tier 1) and use its real fill -- the
    post-deploy hashed '_ps_' id is irrelevant to that path."""
    _open_snapshot("snap-old", "ETH/USD", entry=100.0, qty=2.0)
    # Old-prefix stop, but WITH the decision_id (what pre-deploy code wrote).
    db.save_order_record(order_id="prot-old", decision_id="snap-old", symbol="ETH/USD",
                         side="sell", qty=2.0, filled_qty=2.0, filled_avg_price=155.0,
                         status="filled", type="stop_limit",
                         client_order_id="ETHUSD_protstop_1712345678")
    # A newer unrelated sell that must NOT be chosen.
    db.save_order_record(order_id="exit-other", symbol="ETH/USD", side="sell", qty=1.0,
                         filled_qty=1.0, filled_avg_price=80.0, status="filled", type="market")

    await bot.reconcile_open_snapshots(FakeExchange(price=999.0))

    with db.get_db_session() as session:
        row = session.query(db.DecisionSnapshot).filter_by(decision_id="snap-old").first()
    assert row.status == "closed"
    assert row.realized_pnl == pytest.approx(110.0)  # (155 - 100) * 2
    assert "actual_fill" in row.exit_reason


@pytest.mark.asyncio
async def test_scale_in_cancel_then_rearm_exchange_rejects_reused_dead_id(fresh_db, clean_state, monkeypatch):
    """The requested scenario: a scale-in (same decision_id) where the reused
    stop id is held by a now-CANCELED order, so Alpaca rejects the reuse. The
    adapter must re-arm under a FRESH id. The position must end with exactly
    ONE live resting stop, and reconcile must still find it by decision_id."""
    from src.db import get_recent_order_records

    monkeypatch.setattr(settings, "PROTECTIVE_STOPS_ENABLED", True)
    ex = _RestingStopExchange()
    cid = bot._protstop_client_order_id("ETH/USD", "snap-E")

    # 1) First arm rests stop-1 under the deterministic id.
    await bot._arm_protective_stop(ex, "ETH/USD", qty=1.0, stop_price=90.0, decision_id="snap-E")
    dead_id = bot._state.protective_stops["ETHUSD"]

    # 2) Scale-in. The old stop was canceled on the exchange but the cancel
    #    never cleared local bookkeeping, so the re-arm reuses the same id
    #    while a (now dead) order still holds it.
    ex._status[dead_id] = "canceled"

    await bot._arm_protective_stop(ex, "ETH/USD", qty=2.0, stop_price=85.0, decision_id="snap-E")

    # Exactly one live resting stop, tracked, under a fresh (non-reused) id.
    live = ex.live_ids()
    assert len(live) == 1, ex._status
    tracked = bot._state.protective_stops["ETHUSD"]
    assert tracked == live[0]
    assert tracked != dead_id
    new_cid = next(c for c, o in ex.protective if o == tracked)
    assert new_cid != cid
    assert new_cid.startswith(cid[:36] + "_")

    # Reconcile must still find the re-armed stop by decision_id (tier 1).
    db.save_order_record(order_id=tracked, decision_id="snap-E", symbol="ETH/USD",
                         side="sell", qty=2.0, filled_qty=2.0, filled_avg_price=150.0,
                         status="filled", type="stop_limit", client_order_id=new_cid)
    chosen = bot._select_exit_fill(
        get_recent_order_records("ETHUSD", max_age_sec=3600.0), "snap-E", "sell", cid
    )
    assert chosen is not None
    assert chosen["order_id"] == tracked
    assert chosen["client_order_id"] == new_cid


@pytest.mark.asyncio
async def test_rearm_when_existing_id_order_is_filled_rearms_fresh(fresh_db, clean_state, monkeypatch):
    """A dead (FILLED) existing order holding the reused id must also trigger a
    fresh-id re-arm, never adoption of the dead order."""
    monkeypatch.setattr(settings, "PROTECTIVE_STOPS_ENABLED", True)
    ex = _RestingStopExchange()

    await bot._arm_protective_stop(ex, "ETH/USD", qty=1.0, stop_price=90.0, decision_id="snap-F")
    first_id = bot._state.protective_stops["ETHUSD"]
    # The resting stop triggered and filled while a scale-in was in flight.
    ex._status[first_id] = "filled"
    bot._state.protective_stops.pop("ETHUSD", None)

    await bot._arm_protective_stop(ex, "ETH/USD", qty=1.0, stop_price=88.0, decision_id="snap-F")

    tracked = bot._state.protective_stops["ETHUSD"]
    assert tracked != first_id                 # did NOT adopt the filled order
    assert len(ex.live_ids()) == 1
