"""Fee recording in decision-snapshot P&L.

The exchange adapter reports no commission (alpaca-py>=0.43 Order has no
.commission), so the bot estimates the fee from ESTIMATED_TAKER_FEE_BPS and
flags it (`commission_estimated`). These tests cover:

  - exchange.create_order filling in an estimated commission when the
    exchange reports none (and NOT overriding a directly observed fee),
  - bot._record_committee_outcome subtracting the full ROUND TRIP fee
    (exit fee passed in + entry fee read back from the orders ledger) from
    realized_pnl and return_pct,
  - the orders ledger storing the (estimated) commission.
"""


import datetime

import pytest

import src.bot as bot
import src.db as db
from src.config import settings
from src.exchange import AlpacaExchange


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


def _closed_snapshot(decision_id: str):
    from sqlalchemy import select

    from src.db import DecisionSnapshot, get_db_session
    with get_db_session() as session:
        return session.execute(
            select(DecisionSnapshot).where(DecisionSnapshot.decision_id == decision_id)
        ).scalar_one()


def test_apply_estimated_commission_fills_flag_when_none_observed():
    info = {
        "status": "filled", "filled_avg_price": 100.0, "filled_qty": 1.0,
        "qty": 1.0, "commission": 0.0,
    }
    out = AlpacaExchange._apply_estimated_commission(dict(info))
    expected = 100.0 * 1.0 * (settings.ESTIMATED_TAKER_FEE_BPS / 10000.0)
    assert out["commission"] == pytest.approx(expected)
    assert out["commission_estimated"] is True


def test_apply_estimated_commission_ignores_ordered_qty_on_partial_fill():
    """The fee is charged on what FILLED, so a partial fill (filled_qty < qty)
    must not scale the estimate down: notional already uses filled_qty."""
    info = {
        "status": "filled", "filled_avg_price": 100.0, "filled_qty": 1.0,
        "qty": 2.0, "commission": 0.0,
    }
    out = AlpacaExchange._apply_estimated_commission(dict(info))
    # 100 * 1.0 filled (NOT * 0.5 shortfall) * rate
    expected = 100.0 * 1.0 * (settings.ESTIMATED_TAKER_FEE_BPS / 10000.0)
    assert out["commission"] == pytest.approx(expected)
    assert out["commission_estimated"] is True


def test_apply_estimated_commission_does_not_override_observed_fee():
    info = {
        "status": "filled", "filled_avg_price": 100.0, "filled_qty": 1.0,
        "qty": 1.0, "commission": 2.5,
    }
    out = AlpacaExchange._apply_estimated_commission(dict(info))
    assert out["commission"] == 2.5
    assert "commission_estimated" not in out


def test_apply_estimated_commission_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "ESTIMATED_TAKER_FEE_BPS", 0.0)
    info = {
        "status": "filled", "filled_avg_price": 100.0, "filled_qty": 1.0,
        "qty": 1.0, "commission": 0.0,
    }
    out = AlpacaExchange._apply_estimated_commission(dict(info))
    assert out["commission"] == 0.0
    assert "commission_estimated" not in out


@pytest.mark.asyncio
async def test_snapshot_pnl_subtracts_both_legs_of_round_trip_fee(fresh_db):
    """A 1-round-trip trade must net out BOTH the entry and exit fees, not
    just the exit fee the close path can see."""
    db.save_decision_snapshot(
        decision_id="dec-fee", symbol="BTC/USD", regime="trending",
        final_action="buy", confidence=0.7, size_multiplier=1.0,
        entry_price=100.0, qty=1.0, brain_votes={"transformer": "buy"},
    )
    # Entry-leg fill on the ledger with an estimated fee.
    db.save_order_record(
        order_id="ord-entry", decision_id="dec-fee", symbol="BTC/USD", side="buy",
        qty=1.0, filled_qty=1.0, filled_avg_price=100.0, commission=0.30,
        status="filled", type="market",
    )

    # Exit fee observed on the close fill.
    await bot._record_committee_outcome(
        "BTC/USD", 101.0, exit_reason="signal_close",
        entry_price=100.0, qty=1.0, commission=0.25,
    )

    snap = _closed_snapshot("dec-fee")
    # gross +1.0, minus exit 0.25 and entry 0.30 -> +0.45
    assert snap.realized_pnl == pytest.approx(0.45)
    assert snap.return_pct == pytest.approx(0.45)  # notional 100 -> % == pnl/1
    assert snap.status == "closed"


@pytest.mark.asyncio
async def test_snapshot_pnl_without_entry_leg_only_subtracts_exit_fee(fresh_db):
    """If no entry-leg record exists, only the exit fee is subtracted
    (fail-safe: never guess an entry fee out of thin air)."""
    db.save_decision_snapshot(
        decision_id="dec-fee2", symbol="ETH/USD", regime="trending",
        final_action="buy", confidence=0.7, size_multiplier=1.0,
        entry_price=200.0, qty=2.0, brain_votes={"transformer": "buy"},
    )
    await bot._record_committee_outcome(
        "ETH/USD", 201.0, exit_reason="signal_close",
        entry_price=200.0, qty=2.0, commission=0.50,
    )
    snap = _closed_snapshot("dec-fee2")
    # gross (201-200)*2 = 2.0, minus exit 0.50 -> 1.5
    assert snap.realized_pnl == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_orders_ledger_stores_estimated_commission(fresh_db):
    """_persist_order_record writes the (estimated) commission to the orders
    table so a later reconcile/snapshot sees it."""
    db.save_order_record(
        order_id="ord-1", decision_id="dec-x", symbol="BTC/USD", side="buy",
        qty=1.0, filled_qty=1.0, filled_avg_price=100.0, commission=0.42,
        status="filled", type="market",
    )
    fee = db.get_entry_fee_estimate("BTCUSD", entry_price=100.0)
    assert fee == pytest.approx(0.42)


@pytest.mark.asyncio
async def test_entry_fee_matched_by_decision_id_ignores_age(fresh_db):
    """An 8h hold must STILL subtract the entry fee: the match is by
    decision_id, so the old 2h age cap must not drop it."""
    eight_h_ago = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=8)
    db.save_decision_snapshot(
        decision_id="dec-long", symbol="BTC/USD", regime="trending",
        final_action="buy", confidence=0.7, size_multiplier=1.0,
        entry_price=100.0, qty=1.0, brain_votes={"transformer": "buy"},
    )
    db.save_order_record(
        order_id="ord-entry-long", decision_id="dec-long", symbol="BTC/USD", side="buy",
        qty=1.0, filled_qty=1.0, filled_avg_price=100.0, commission=0.30,
        status="filled", type="market", submitted_at=eight_h_ago,
    )

    # Direct lookup keyed by decision_id: no age cap.
    assert db.get_entry_fee_estimate("BTCUSD", 100.0, "dec-long") == pytest.approx(0.30)

    await bot._record_committee_outcome(
        "BTC/USD", 101.0, exit_reason="signal_close",
        entry_price=100.0, qty=1.0, commission=0.25,
    )
    snap = _closed_snapshot("dec-long")
    # gross +1.0, minus exit 0.25 and the 8h-old entry 0.30 -> +0.45
    assert snap.realized_pnl == pytest.approx(0.45)


@pytest.mark.asyncio
async def test_entry_fee_sums_all_scale_in_buys(fresh_db):
    """A scale-in folds several buys into ONE snapshot; the entry fee must be
    the SUM of every one of their commissions, not just the first/latest."""
    db.save_decision_snapshot(
        decision_id="dec-scale", symbol="ETH/USD", regime="trending",
        final_action="buy", confidence=0.7, size_multiplier=1.0,
        entry_price=100.0, qty=4.0, brain_votes={"transformer": "buy"},
    )
    for i in range(4):
        db.save_order_record(
            order_id=f"ord-add-{i}", decision_id="dec-scale", symbol="ETH/USD", side="buy",
            qty=1.0, filled_qty=1.0, filled_avg_price=100.0, commission=0.10,
            status="filled", type="market",
        )

    assert db.get_entry_fee_estimate("ETHUSD", 100.0, "dec-scale") == pytest.approx(0.40)

    await bot._record_committee_outcome(
        "ETH/USD", 101.0, exit_reason="signal_close",
        entry_price=100.0, qty=4.0, commission=0.25,
    )
    snap = _closed_snapshot("dec-scale")
    # gross (101-100)*4 = 4.0, minus exit 0.25 and entry 0.40 -> 3.35
    assert snap.realized_pnl == pytest.approx(3.35)


def test_entry_fee_falls_back_to_symbol_price_without_decision_id(fresh_db):
    """When no decision_id is available, the symbol/price heuristic still
    finds the entry fill."""
    db.save_order_record(
        order_id="ord-fb", decision_id="dec-other", symbol="BTC/USD", side="buy",
        qty=1.0, filled_qty=1.0, filled_avg_price=100.0, commission=0.55,
        status="filled", type="market",
    )
    assert db.get_entry_fee_estimate("BTCUSD", 100.0, None) == pytest.approx(0.55)