"""Unit tests for RiskManager ATR sizing, correlation downscaling, trailing stops
and killswitch recovery.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.config import settings
from src.risk import RiskManager


def test_atr_position_sizing():
    """Verify position sizing uses ATR volatility parity when available."""
    rm = RiskManager(AsyncMock())
    
    # Standard sizing with ATR=500, price=50000, risk_amount = 10000 * 0.01 = 100
    # Stop distance = 500 * 2.0 = 1000 -> Units = 100 / 1000 = 0.1
    size, status = rm.calculate_position_size("BTC/USD", 50000.0, "trending", atr=500.0, confidence=1.0)
    assert status == "ok"
    assert size > 0.0


def test_min_order_bump_clears_exchange_floor_after_rounding():
    """Regression test: the min-order-size floor-bump must round UP (ceil)
    to the nearest 1e-6 unit, not round-to-nearest, or the bumped notional
    can land fractionally BELOW MIN_ORDER_USD and the exchange rejects it
    again anyway -- defeating the whole point of the bump. Confirmed with
    price=2672.18: round(10.0/price, 6) -> notional $9.9993 (still rejected);
    math.ceil(10.0/price*1e6)/1e6 -> notional $10.002 (clears). This exact
    bug existed in an earlier version of this fix and was corrected
    2026-09-21 (verified as already-fixed while cross-checking an external
    review's claims)."""
    rm = RiskManager(AsyncMock())
    price = 2672.18  # chosen because round(10.0/price, 6) rounds DOWN below $10 notional
    size, status = rm.calculate_position_size(
        "ETH/USD", price, "sideways", atr=62.9368, confidence=0.62,
        expected_return_pct=0.03, current_equity=97.21, drawdown_pct=0.0,
    )
    assert status == "ok"
    assert size * price >= settings.MIN_ORDER_USD, (
        f"bumped notional ${size * price:.4f} is below the ${settings.MIN_ORDER_USD} "
        "exchange minimum -- the exchange would reject this order"
    )


def test_correlation_downscaling():
    """Verify position size is downscaled when portfolio correlation is high."""
    rm = RiskManager(AsyncMock())

    # Create highly correlated returns
    np.random.seed(42)
    ret_a = np.random.randn(100)
    ret_b = ret_a + np.random.randn(100) * 0.01  # ~0.99 correlation

    returns_matrix = {"BTC/USD": ret_a, "ETH/USD": ret_b}

    size_normal, _ = rm.calculate_position_size("BTC/USD", 50000.0, "neutral")
    size_corr, _ = rm.calculate_position_size("BTC/USD", 50000.0, "neutral", returns_matrix=returns_matrix)

    assert size_corr < size_normal


def test_trailing_stop_activation_and_trigger():
    """Verify trailing stop activates after profit threshold (4%) and triggers after peak pullback (3%)."""
    rm = RiskManager(AsyncMock())
    symbol = "BTC/USD"
    entry_price = 100.0
    qty = 1.0

    # 1. Price at entry -> hold
    assert rm.check_trailing_stop(symbol, 100.0, entry_price, qty) == "hold"

    # 2. Price rises 5% (activation threshold is 4.0%) -> peak tracked at 105.0
    assert rm.check_trailing_stop(symbol, 105.0, entry_price, qty) == "hold"
    assert rm.peak_prices.get(symbol) == 105.0

    # 3. Price drops 4% from peak (105.0 -> 100.8, distance threshold is 3.0%) -> triggers close
    assert rm.check_trailing_stop(symbol, 100.8, entry_price, qty) == "close"
    assert symbol not in rm.peak_prices


# ── Killswitch recovery ───────────────────────────────────────────────────────
# Regression: killswitch_active had NO clear path for a max_drawdown breach. The
# only clear was inside the daily-loss branch (and required "daily_loss_limit" in
# the reason), so once a drawdown breach latched the flag the bot refused every
# new entry and force-flattened positions until someone restarted it -- and a
# restart re-tripped immediately, because bot.py restores peak_equity from
# bot_state.json.

def _account(equity: float) -> dict:
    return {"equity": equity, "cash": equity, "portfolio_value": equity}


# ── Gap-risk circuit breaker ──────────────────────────────────────────────────
# Regression: risk.py used `timedelta` in both gap-risk paths but only imported
# `datetime`/`UTC`, so each path raised NameError at exactly the moment it was
# supposed to act:
#   - check_gap_risk_on_fill() blew up as soon as a real slippage gap was
#     detected (instead of raising the circuit breaker / recording the event),
#   - get_gap_risk_multiplier() raised instead of AUTO-RECOVERING once the
#     recovery window elapsed, leaving the breaker latched at 0.5x size forever.
# (Found via ruff F821; both call sites are on the live sizing path.)

def test_gap_risk_fill_check_records_event_without_name_error():
    rm = RiskManager(AsyncMock())
    result = rm.check_gap_risk_on_fill(
        "BTC/USD", expected_price=100.0, filled_price=90.0,
        exit_type="stop_loss", entry_price=105.0, side="long",
    )
    assert result["gap_risk_detected"] is True
    assert result["slippage_bps"] == pytest.approx(1000.0)
    assert len(rm._gap_risk_events) == 1


def test_gap_risk_circuit_breaker_triggers_after_max_events():
    rm = RiskManager(AsyncMock())
    for _ in range(rm._gap_risk_max_events):
        rm.check_gap_risk_on_fill("BTC/USD", 100.0, 90.0, "stop_loss", 105.0, "long")
    assert rm._gap_risk_active is True
    assert rm.get_gap_risk_multiplier() == pytest.approx(rm._gap_risk_size_multiplier)


def test_gap_risk_multiplier_auto_recovers_after_window():
    rm = RiskManager(AsyncMock())
    rm._gap_risk_active = True
    rm._gap_risk_last_triggered = datetime.now(UTC) - timedelta(hours=rm._gap_risk_recovery_hours + 1)
    assert rm.get_gap_risk_multiplier() == pytest.approx(1.0)
    assert rm._gap_risk_active is False
    assert rm._gap_risk_last_triggered is None


@pytest.mark.asyncio
async def test_drawdown_killswitch_clears_after_equity_recovery(monkeypatch):
    """Latch on at the limit, stay latched mid-band, clear past the recovery line."""
    monkeypatch.setattr(settings, "DAILY_LOSS_LIMIT", -50.0)  # isolate drawdown logic
    ex = AsyncMock()
    ex.get_positions.return_value = []
    rm = RiskManager(ex)

    ex.get_account.return_value = _account(10_000.0)
    await rm.update_account_status()  # establishes peak_equity
    assert rm.is_killswitch_active() is False

    # -12% breaches the -10% limit -> killswitch + liquidation request
    ex.get_account.return_value = _account(8_800.0)
    status = await rm.update_account_status()
    assert status["status"] == "killswitch_activated"
    assert status["reason"] == "max_drawdown_exceeded"
    assert rm.is_killswitch_active() is True
    assert "max_drawdown" in rm.killswitch_reason

    # -6% is still inside the breach band (recovery line is -5%) -> stays latched,
    # so equity hovering just above the limit cannot flap the killswitch.
    ex.get_account.return_value = _account(9_400.0)
    await rm.update_account_status()
    assert rm.is_killswitch_active() is True, "cleared too early -- flapping risk"

    # -4% is past the halfway recovery line -> cleared
    ex.get_account.return_value = _account(9_600.0)
    status = await rm.update_account_status()
    assert status["status"] == "risk_ok"
    assert rm.is_killswitch_active() is False
    assert rm.killswitch_reason == ""


@pytest.mark.asyncio
async def test_drawdown_breach_still_fires_at_the_limit(monkeypatch):
    """The recovery path must not weaken breach detection itself."""
    monkeypatch.setattr(settings, "DAILY_LOSS_LIMIT", -50.0)
    ex = AsyncMock()
    ex.get_positions.return_value = []
    rm = RiskManager(ex)

    ex.get_account.return_value = _account(10_000.0)
    await rm.update_account_status()

    ex.get_account.return_value = _account(8_999.0)  # -10.01%
    status = await rm.update_account_status()
    assert status["status"] == "killswitch_activated"
    assert status["action"] == "liquidate_all"
    assert rm.is_killswitch_active() is True


@pytest.mark.asyncio
async def test_daily_loss_killswitch_still_clears_on_new_day(monkeypatch):
    """The pre-existing daily-loss reset must keep working (and only for it)."""
    monkeypatch.setattr(settings, "MAX_DRAWDOWN_STOP", -50.0)  # isolate daily-loss logic
    ex = AsyncMock()
    ex.get_positions.return_value = []
    rm = RiskManager(ex)

    ex.get_account.return_value = _account(10_000.0)
    await rm.update_account_status()

    ex.get_account.return_value = _account(9_500.0)  # -5% day, limit is -3%
    status = await rm.update_account_status()
    assert status["status"] == "killswitch_activated"
    assert status["reason"] == "daily_loss_limit_exceeded"
    assert rm.is_killswitch_active() is True

    # Same equity, but a new calendar day: daily PnL resets and the flag clears.
    rm.last_check_time = datetime.now(UTC) - timedelta(days=1)
    status = await rm.update_account_status()
    assert status["status"] == "risk_ok"
    assert rm.is_killswitch_active() is False
    assert rm.killswitch_reason == ""



# ── Drawdown killswitch: flat-book cooldown recovery ──────────────────────────
# Regression: once a max-drawdown breach liquidated the book, equity was frozen
# (all cash, entries blocked), so it could never climb back to the -5% recovery
# line; new days didn't clear it and a restart re-tripped (peak_equity restored
# from bot_state.json). Now: once FLAT for DRAWDOWN_KILLSWITCH_COOLDOWN_HOURS
# after the breach, peak_equity is re-based to current equity and the
# killswitch clears -- future drawdowns are measured from the new base.

async def _breach(monkeypatch, positions=None):
    monkeypatch.setattr(settings, "DAILY_LOSS_LIMIT", -50.0)  # isolate drawdown logic
    monkeypatch.setattr(settings, "DRAWDOWN_KILLSWITCH_COOLDOWN_HOURS", 24.0)
    ex = AsyncMock()
    ex.get_positions.return_value = positions or []
    rm = RiskManager(ex)
    ex.get_account.return_value = _account(10_000.0)
    await rm.update_account_status()
    ex.get_account.return_value = _account(8_800.0)  # -12% breach
    status = await rm.update_account_status()
    assert status["action"] == "liquidate_all"
    return rm, ex


@pytest.mark.asyncio
async def test_drawdown_killswitch_stays_latched_during_cooldown(monkeypatch):
    rm, _ = await _breach(monkeypatch)
    rm._drawdown_tripped_at = datetime.now(UTC) - timedelta(hours=23)
    for _ in range(10):
        await rm.update_account_status()
    assert rm.is_killswitch_active() is True
    assert rm.peak_equity == 10_000.0


@pytest.mark.asyncio
async def test_drawdown_killswitch_tripped_at_not_extended_by_repeat_breaches(monkeypatch):
    """Every cycle while breached re-enters the breach branch; the cooldown
    must run from the FIRST breach, not restart each cycle."""
    rm, _ = await _breach(monkeypatch)
    first = rm._drawdown_tripped_at
    assert first is not None
    await rm.update_account_status()
    assert rm._drawdown_tripped_at == first


@pytest.mark.asyncio
async def test_drawdown_killswitch_clears_after_cooldown_when_flat(monkeypatch):
    rm, _ = await _breach(monkeypatch)
    rm._drawdown_tripped_at = datetime.now(UTC) - timedelta(hours=25)
    status = await rm.update_account_status()
    assert status["status"] == "risk_ok", status
    assert rm.is_killswitch_active() is False
    assert rm.killswitch_reason == ""
    assert rm.peak_equity == 8_800.0  # re-based
    assert rm._drawdown_tripped_at is None


@pytest.mark.asyncio
async def test_drawdown_killswitch_rebased_peak_still_trips_on_new_drawdown(monkeypatch):
    rm, ex = await _breach(monkeypatch)
    rm._drawdown_tripped_at = datetime.now(UTC) - timedelta(hours=25)
    await rm.update_account_status()  # clears, peak -> 8800
    ex.get_account.return_value = _account(7_900.0)  # -10.2% from 8800
    status = await rm.update_account_status()
    assert status["status"] == "killswitch_activated"
    assert status["reason"] == "max_drawdown_exceeded"


@pytest.mark.asyncio
async def test_drawdown_killswitch_does_not_clear_with_open_positions(monkeypatch):
    """Positions still open (liquidation failed/partial) -> not flat -> stay latched."""
    rm, _ = await _breach(monkeypatch, positions=[{"symbol": "BTC/USD", "qty": "0.01"}])
    rm._drawdown_tripped_at = datetime.now(UTC) - timedelta(hours=25)
    await rm.update_account_status()
    assert rm.is_killswitch_active() is True
    assert rm.peak_equity == 10_000.0


@pytest.mark.asyncio
async def test_drawdown_cooldown_survives_restart(monkeypatch):
    """bot.py restores peak_equity and the breach time; a restart must not
    restart the cooldown clock, or frequent redeploys would block forever."""
    rm, ex = await _breach(monkeypatch)
    tripped = datetime.now(UTC) - timedelta(hours=25)
    restarted = RiskManager(ex)
    restarted.peak_equity = rm.peak_equity
    restarted._drawdown_tripped_at = tripped  # restored by bot.py
    status = await restarted.update_account_status()
    assert status["status"] == "risk_ok", status
    assert restarted.is_killswitch_active() is False


@pytest.mark.asyncio
async def test_liquidate_all_positions_cancels_protective_stops():
    """The killswitch flatten path must cancel resting protective stops before
    closing, or a dangling sell lingers after the position is gone."""
    ex = AsyncMock()
    ex.get_positions = AsyncMock(return_value=[
        {"symbol": "BTC/USD", "qty": "0.1", "market_value": "5000"},
    ])
    ex.create_order = AsyncMock(return_value={"id": "close_1", "filled_avg_price": 50000.0})
    ex.cancel_order = AsyncMock(return_value=True)
    rm = RiskManager(ex)
    rm.protective_stops["BTCUSD"] = "stop_1"

    await rm.liquidate_all_positions()

    ex.cancel_order.assert_awaited_once_with("stop_1")
    assert "BTCUSD" not in rm.protective_stops


@pytest.mark.asyncio
async def test_reduce_exposure_cancels_protective_stops(monkeypatch):
    """The exposure-cap reduction path also closes positions, so it must
    cancel the resting stops for the symbols it closes."""
    ex = AsyncMock()
    ex.get_positions = AsyncMock(return_value=[
        {"symbol": "BTC/USD", "qty": "0.1", "market_value": "9000", "unrealized_pl": "-10"},
    ])
    ex.create_order = AsyncMock(return_value={
        "id": "close_1", "filled_avg_price": 50000.0, "filled_qty": 0.1,
    })
    ex.cancel_order = AsyncMock(return_value=True)
    rm = RiskManager(ex)
    rm.protective_stops["BTCUSD"] = "stop_9"
    monkeypatch.setattr(settings, "MAX_PORTFOLIO_PCT", 0.01)  # cap $100 < $9000 held
    rm.peak_equity = 1000.0

    result = await rm.reduce_exposure_to_cap()

    assert result["status"] == "exposure_reduced", result
    ex.cancel_order.assert_awaited_once_with("stop_9")
    assert "BTCUSD" not in rm.protective_stops


@pytest.mark.asyncio
async def test_cancel_protective_stop_is_failsafe():
    """A failed cancel must not raise out of an emergency flatten -- the
    position is the real risk, not the leftover order."""
    ex = AsyncMock()
    ex.cancel_order = AsyncMock(side_effect=RuntimeError("network down"))
    rm = RiskManager(ex)
    rm.protective_stops["BTCUSD"] = "stop_1"

    await rm._cancel_protective_stop("BTC/USD")  # must not raise
    assert "BTCUSD" not in rm.protective_stops
