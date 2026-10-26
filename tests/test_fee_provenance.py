"""Fee provenance on the orders ledger (2026-10-25, item 1).

Storage-only change: the exchange adapter cannot observe a fee (Alpaca reports
none), so it estimates one and flags ``commission_estimated``. This suite pins
that the *provenance* of a stored fee is persisted, so a stored 0.0 is never
mistaken for a verified fee-free fill:

  - ``fee_source`` round-trips for "estimated" and "exchange",
  - an omitted ``fee_source`` is NULL (unknown), never fabricated,
  - ``bot._persist_order_record`` derives provenance from the adapter's flag,
  - the additive migration adds the column to a legacy table and is idempotent,

Nothing here populates historical "exchange" provenance; legacy rows stay NULL.
"""



import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import select, text

import src.bot as bot
import src.db as db
from src.config import settings


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


def _read_order(order_id: str):
    from src.db import OrderRecord, get_db_session
    with get_db_session() as session:
        return session.execute(
            select(OrderRecord).where(OrderRecord.order_id == order_id)
        ).scalar_one()


# ── 1 & 2: round-trip both approved values ────────────────────────────────

def test_fee_source_round_trips_estimated(fresh_db):
    db.save_order_record(
        order_id="o-est", symbol="ETH/USD", side="buy", qty=1.0,
        filled_qty=1.0, filled_avg_price=100.0, commission=0.25,
        fee_source="estimated", status="filled",
    )
    assert _read_order("o-est").fee_source == "estimated"


def test_fee_source_round_trips_exchange(fresh_db):
    db.save_order_record(
        order_id="o-exch", symbol="ETH/USD", side="buy", qty=1.0,
        filled_qty=1.0, filled_avg_price=100.0, commission=0.25,
        fee_source="exchange", status="filled",
    )
    assert _read_order("o-exch").fee_source == "exchange"


# ── 3: omitted provenance is NULL, never fabricated ───────────────────────

def test_omitted_fee_source_is_null_not_exchange(fresh_db):
    db.save_order_record(
        order_id="o-none", symbol="BTC/USD", side="buy", qty=1.0,
        filled_qty=1.0, filled_avg_price=100.0, commission=0.0,
        status="filled",
    )
    row = _read_order("o-none")
    assert row.fee_source is None
    assert row.fee_source != "exchange"


# ── 4-6: derivation in the single persistence funnel ──────────────────────

def test_persist_derives_estimated_from_adapter_flag(fresh_db):
    bot._persist_order_record(
        {"id": "f-est", "qty": 1.0, "filled_qty": 1.0, "filled_avg_price": 100.0,
         "commission": 0.25, "commission_estimated": True, "status": "filled"},
        "ETH/USD", "buy", "cid-1",
    )
    row = _read_order("f-est")
    assert row.fee_source == "estimated"
    assert row.commission == pytest.approx(0.25)


def test_persist_derives_exchange_from_observed_fee(fresh_db):
    bot._persist_order_record(
        {"id": "f-exch", "qty": 1.0, "filled_qty": 1.0, "filled_avg_price": 100.0,
         "commission": 0.25, "status": "filled"},
        "ETH/USD", "buy", "cid-2",
    )
    assert _read_order("f-exch").fee_source == "exchange"


def test_persist_zero_fee_without_flag_is_unknown(fresh_db):
    bot._persist_order_record(
        {"id": "f-zero", "qty": 1.0, "filled_qty": 1.0, "filled_avg_price": 100.0,
         "commission": 0.0, "status": "filled"},
        "BTC/USD", "buy", "cid-3",
    )
    row = _read_order("f-zero")
    assert row.fee_source is None
    assert row.commission == 0.0


# ── 7: additive migration on a legacy table, and idempotency ──────────────

def test_migration_adds_fee_source_to_legacy_orders_and_is_idempotent(tmp_path):
    original = settings.DATABASE_URL
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    settings.DATABASE_URL = "sqlite:///" + str(tmp_path / "legacy.db").replace("\\", "/")
    try:
        engine = db.get_engine()
        # Simulate a deployed DB whose `orders` predates fee_source.
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE orders (order_id VARCHAR(64) PRIMARY KEY, "
                "decision_id VARCHAR(64), symbol VARCHAR(32), side VARCHAR(16), "
                "qty FLOAT, filled_qty FLOAT, filled_avg_price FLOAT, "
                "commission FLOAT, status VARCHAR(32), type VARCHAR(16), "
                "time_in_force VARCHAR(16), client_order_id VARCHAR(64), "
                "submitted_at DATETIME, filled_at DATETIME)"
            ))
            conn.execute(text(
                "INSERT INTO orders (order_id, symbol, side, commission, status) "
                "VALUES ('legacy-1', 'BTC/USD', 'buy', 0.0, 'filled')"
            ))
        assert "fee_source" not in {c["name"] for c in sa_inspect(engine).get_columns("orders")}

        db._ensure_additive_columns()
        cols = {c["name"] for c in sa_inspect(engine).get_columns("orders")}
        assert "fee_source" in cols
        # Legacy row keeps NULL provenance -- never fabricated.
        with engine.begin() as conn:
            assert conn.execute(text(
                "SELECT fee_source FROM orders WHERE order_id='legacy-1'"
            )).scalar() is None

        # Idempotent: a second run must not raise or duplicate.
        db._ensure_additive_columns()
        names = [c["name"] for c in sa_inspect(engine).get_columns("orders")]
        assert names.count("fee_source") == 1
    finally:
        if db._engine is not None:
            db._engine.dispose()
        db._engine = None
        db._tables_ensured = False
        settings.DATABASE_URL = original
