"""bot.py committee-override MIN_HOLD gate.

A held long whose strategy emitted a DISCRETIONARY close (momentum, trend,
mean-reversion, breakout, grid, scalp) must not be closed by a committee
"sell" override before MIN_HOLD_MINUTES. strategies.py gates its own closes,
but the committee override path in bot.py never went through that gate, so a
committee sell bypassed MIN_HOLD entirely -- the exact 2026-10-02 churn.

PRICE-BASED risk exits (stop loss, trailing stop, profit target, max hold) are
handled earlier in bot.py and must still fire immediately, inside MIN_HOLD.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot import process_signal_for_symbol
from src.config import settings
from src.exchange import AlpacaExchange
from src.risk import RiskManager
from src.strategies import TradingStrategy

MOMENTUM_CLOSE_REASON = "[MOMENTUM] Momentum: Loss of bullish momentum"


def _committee_sell():
    from src.committee.models import CommitteeResult
    return CommitteeResult(
        action="sell", score=0.62, size_multiplier=1.0, entropy=0.0,
        votes=[], decision_id="sell-exit",
    )


@pytest.fixture
def mock_exchange():
    ex = AsyncMock(spec=AlpacaExchange)
    ex.get_positions = AsyncMock(return_value=[])
    ex.get_latest_bar = AsyncMock()
    ex.create_order = AsyncMock(return_value={
        "id": "close_1", "filled_avg_price": 40000.0, "filled_qty": 0.1,
        "commission": 1.0, "status": "filled",
    })
    ex.get_account = AsyncMock(return_value={"equity": 10000.0, "cash": 10000.0, "portfolio_value": 10000.0})
    ex.load = AsyncMock()
    ex.close = AsyncMock()
    return ex


@pytest.fixture
def mock_risk_manager():
    rm = AsyncMock(spec=RiskManager)
    rm.update_account_status = AsyncMock(return_value={
        "status": "risk_ok", "equity": 10000.0, "cash": 10000.0,
        "portfolio_value": 10000.0, "drawdown_pct": -2.0, "daily_pnl": 0.0,
        "open_positions": 1, "current_exposure": 4000.0,
    })
    rm.is_killswitch_active = MagicMock(return_value=False)
    rm.check_killswitch_conditions = AsyncMock(return_value=False)
    rm.check_trailing_stop = MagicMock(return_value="hold")
    rm.calculate_position_size = MagicMock(return_value=(0.1, "ok"))
    rm.check_and_reserve_exposure = AsyncMock(return_value=(1000.0, "ok"))
    rm.reserve_position_slot = AsyncMock(return_value=(True, "ok"))
    rm.release_position_slot = AsyncMock()
    rm.release_reserved_exposure = AsyncMock()
    rm.peak_prices = {}
    rm.peak_equity = 10000.0
    rm.record_fill_costs = MagicMock()
    return rm


@pytest.fixture(autouse=True)
def _reset_state():
    import src.bot as bot_mod
    bot_mod._state.last_fill_times = {}
    bot_mod._state.protective_stops = {}
    bot_mod._state.position_adds = {}
    bot_mod._state.cooldowns = {}
    yield
    bot_mod._state.last_fill_times = {}
    bot_mod._state.protective_stops = {}
    bot_mod._state.position_adds = {}
    bot_mod._state.cooldowns = {}


def _held_long():
    return [{"symbol": "BTCUSD", "qty": "0.1", "side": "long", "avg_entry_price": 40000.0}]


def _strategy_selling(held_minutes):
    strat = AsyncMock(spec=TradingStrategy)
    strat.generate_trading_signal = AsyncMock(return_value={
        "action": "sell", "confidence": 0.5, "reason": MOMENTUM_CLOSE_REASON,
        "regime": "trending", "rsi": 30.0, "atr": 500.0, "features": {},
    })
    strat.backtest = False
    strat.get_position_held_minutes = MagicMock(return_value=held_minutes)
    strat._position_first_seen = {}
    return strat


@pytest.mark.asyncio
async def test_committee_discretionary_sell_held_within_min_hold(mock_exchange, mock_risk_manager):
    """Committee sell of a discretionary strategy exit inside MIN_HOLD must
    NOT close the position."""
    import src.bot as bot_mod
    strat = _strategy_selling(held_minutes=5.0)  # < MIN_HOLD_MINUTES (30)
    mock_exchange.get_positions = AsyncMock(return_value=_held_long())

    async def _sell(symbol, price, signal):
        return _committee_sell()

    with patch.object(settings, "MIN_HOLD_MINUTES", 30), \
         patch.object(settings, "COMMITTEE_MIN_HOLD_EXEMPT", False), \
         patch("src.committee.committee.run_committee", new=_sell):
        await process_signal_for_symbol(
            symbol="BTC/USD", current_price=40000.0, risk_manager=mock_risk_manager,
            strategy=strat, ex=mock_exchange, positions=_held_long(),
            regime_flag=None, banned_symbols=set(),
        )

    mock_exchange.create_order.assert_not_called()
    assert bot_mod._state.latest_scan_results["BTC/USD"]["action"] == "HOLD_MIN_HOLD"


@pytest.mark.asyncio
async def test_committee_discretionary_sell_allowed_after_min_hold(mock_exchange, mock_risk_manager):
    """Once MIN_HOLD_MINUTES has elapsed the same committee sell closes."""
    strat = _strategy_selling(held_minutes=45.0)  # > MIN_HOLD_MINUTES
    mock_exchange.get_positions = AsyncMock(return_value=_held_long())

    async def _sell(symbol, price, signal):
        return _committee_sell()

    with patch.object(settings, "MIN_HOLD_MINUTES", 30), \
         patch.object(settings, "COMMITTEE_MIN_HOLD_EXEMPT", False), \
         patch("src.committee.committee.run_committee", new=_sell):
        await process_signal_for_symbol(
            symbol="BTC/USD", current_price=40000.0, risk_manager=mock_risk_manager,
            strategy=strat, ex=mock_exchange, positions=_held_long(),
            regime_flag=None, banned_symbols=set(),
        )

    mock_exchange.create_order.assert_called_once()
    assert mock_exchange.create_order.call_args.kwargs["side"] == "sell"


@pytest.mark.asyncio
async def test_risk_exit_closes_immediately_inside_min_hold(mock_exchange, mock_risk_manager):
    """A risk exit (trailing stop here) must fire immediately even inside
    MIN_HOLD -- it is handled before the committee gate is reached."""
    mock_exchange.get_positions = AsyncMock(return_value=_held_long())
    mock_risk_manager.check_trailing_stop = MagicMock(return_value="close")
    strat = _strategy_selling(held_minutes=1.0)  # deep inside MIN_HOLD

    with patch.object(settings, "MIN_HOLD_MINUTES", 30):
        await process_signal_for_symbol(
            symbol="BTC/USD", current_price=40000.0, risk_manager=mock_risk_manager,
            strategy=strat, ex=mock_exchange, positions=_held_long(),
            regime_flag=None, banned_symbols=set(),
        )

    mock_exchange.create_order.assert_called_once()
    assert mock_exchange.create_order.call_args.kwargs["side"] == "sell"


@pytest.mark.asyncio
async def test_exempt_flag_restores_old_churn_behaviour(mock_exchange, mock_risk_manager):
    """COMMITTEE_MIN_HOLD_EXEMPT=True disables the gate (opt-out escape hatch)."""
    strat = _strategy_selling(held_minutes=2.0)
    mock_exchange.get_positions = AsyncMock(return_value=_held_long())

    async def _sell(symbol, price, signal):
        return _committee_sell()

    with patch.object(settings, "MIN_HOLD_MINUTES", 30), \
         patch.object(settings, "COMMITTEE_MIN_HOLD_EXEMPT", True), \
         patch("src.committee.committee.run_committee", new=_sell):
        await process_signal_for_symbol(
            symbol="BTC/USD", current_price=40000.0, risk_manager=mock_risk_manager,
            strategy=strat, ex=mock_exchange, positions=_held_long(),
            regime_flag=None, banned_symbols=set(),
        )

    mock_exchange.create_order.assert_called_once()


@pytest.mark.asyncio
async def test_real_strategy_momentum_close_held_end_to_end(mock_exchange, mock_risk_manager):
    """End-to-end with the REAL strategy: a momentum close on a fresh long is
    held back (no order), so the churn cannot be re-opened through the
    committee override either."""
    real = TradingStrategy(mock_exchange, backtest=False)
    real._active_strategy["BTC/USD"] = "momentum"
    real._position_first_seen["BTC/USD"] = None  # forces None -> fail-open first sighting
    # Patch regime so the momentum strategy sees RSI ROC < -5 (its close rule).
    real.analyze_market_regime = AsyncMock(return_value={
        "regime": "trending", "rsi": 30.0, "prev_rsi": 45.0, "atr": 500.0,
        "hurst": 0.65, "htf_trend": "up",
    })
    # Fresh position (held ~0 min) with an entry timestamp of now.
    from datetime import UTC, datetime
    pos = [{"symbol": "BTCUSD", "qty": "0.1", "side": "long", "avg_entry_price": 40000.0,
            "created_at": datetime.now(UTC).isoformat()}]
    mock_exchange.get_positions = AsyncMock(return_value=pos)

    async def _sell(symbol, price, signal):
        return _committee_sell()

    with patch.object(settings, "MIN_HOLD_MINUTES", 30), \
         patch.object(settings, "COMMITTEE_MIN_HOLD_EXEMPT", False), \
         patch("src.committee.committee.run_committee", new=_sell):
        await process_signal_for_symbol(
            symbol="BTC/USD", current_price=40000.0, risk_manager=mock_risk_manager,
            strategy=real, ex=mock_exchange, positions=pos,
            regime_flag=None, banned_symbols=set(),
        )

    mock_exchange.create_order.assert_not_called()