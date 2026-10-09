"""MIN_HOLD gate behavior for strategy-based closes.

Regression tests for the production churn of 2026-10-02: 8+ BTC/USD
positions closed on "[MOMENTUM] Momentum: Loss of bullish momentum" after
only ~3 scans (holding_period_sec 177-179), far inside MIN_HOLD_MINUTES,
because the consecutive-signal override let a persistent discretionary
close bypass the gate.

Contract under test (src/strategies.py generate_trading_signal):
  - A discretionary strategy close (momentum / trend / mean-reversion /
    breakout / grid / scalp) inside MIN_HOLD is suppressed, even after
    many consecutive identical signals -- the override does not apply.
  - Price-based risk exits (stop loss / take profit / trailing / max hold,
    handled by _check_price_based_exits) still fire immediately.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from src.config import settings
from src.strategies import TradingStrategy


def _fresh_entry_position(price: float = 100.0, held_seconds: int = 60) -> dict:
    """Position whose entry timestamp is well inside MIN_HOLD_MINUTES."""
    return {
        "symbol": "BTCUSD",
        "qty": 1.0,
        "avg_entry_price": price,
        "created_at": (datetime.now(UTC) - timedelta(seconds=held_seconds)).isoformat(),
    }


def _aged_entry_position(price: float = 100.0) -> dict:
    """Position held past MIN_HOLD_MINUTES."""
    minutes = int(settings.MIN_HOLD_MINUTES) + 5
    return {
        "symbol": "BTCUSD",
        "qty": 1.0,
        "avg_entry_price": price,
        "created_at": (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat(),
    }


def _patch_regime():
    return patch.object(
        TradingStrategy,
        "analyze_market_regime",
        new_callable=AsyncMock,
        return_value={"regime": "neutral", "atr": 0.0, "rsi": 50.0, "prev_rsi": 50.0},
    )


@pytest.mark.asyncio
async def test_discretionary_close_suppressed_for_many_consecutive_signals():
    """A momentum 'loss of momentum' close must stay gated inside MIN_HOLD
    even after more than MIN_HOLD_CONSECUTIVE_SIGNALS scans of the same
    signal."""
    strat = TradingStrategy(AsyncMock(), cache_ttl=0.0)
    position = _fresh_entry_position()
    strat._active_strategy["BTC/USD"] = "momentum"

    momentum_signal = {
        "action": "close",
        "reason": "Momentum: Loss of bullish momentum",
    }
    with _patch_regime(), \
         patch("src.execution_strategies.MomentumStrategy.generate_signal", return_value=momentum_signal):
        scans = settings.MIN_HOLD_CONSECUTIVE_SIGNALS + 5
        actions = []
        for _ in range(scans):
            sig = await strat.generate_trading_signal("BTC/USD", 100.0, position)
            actions.append(sig["action"])

    # Every scan is suppressed: never 'close', and never even queued for a
    # consecutive override (discretionary closes are not counted).
    assert actions == ["hold"] * scans
    assert "BTCUSD" not in strat._pending_close


@pytest.mark.asyncio
async def test_discretionary_close_allowed_once_min_hold_elapsed():
    """The same momentum close fires immediately once MIN_HOLD is satisfied."""
    strat = TradingStrategy(AsyncMock(), cache_ttl=0.0)
    position = _aged_entry_position()
    strat._active_strategy["BTC/USD"] = "momentum"

    momentum_signal = {
        "action": "close",
        "reason": "Momentum: Loss of bullish momentum",
    }
    with _patch_regime(), \
         patch("src.execution_strategies.MomentumStrategy.generate_signal", return_value=momentum_signal):
        sig = await strat.generate_trading_signal("BTC/USD", 100.0, position)

    assert sig["action"] == "close"
    assert sig["reason"] == "[MOMENTUM] Momentum: Loss of bullish momentum"


@pytest.mark.asyncio
async def test_stop_loss_fires_immediately_inside_min_hold():
    """A price-based stop-loss exit is a risk exit and must NOT be gated,
    even on the very first scan of a young position."""
    strat = TradingStrategy(AsyncMock(), cache_ttl=0.0)
    position = _fresh_entry_position(price=100.0)
    strat._active_strategy["BTC/USD"] = "momentum"

    # -5% move breaches STOP_LOSS_PCT (4%).
    with _patch_regime():
        sig = await strat.generate_trading_signal("BTC/USD", 95.0, position)

    assert sig["action"] == "close"
    assert sig["reason"] == "stop_loss_hit"


@pytest.mark.asyncio
async def test_mean_reversion_close_is_discretionary_and_gated():
    """Mean-reversion target closes are discretionary too and respect the
    strict gate inside MIN_HOLD."""
    strat = TradingStrategy(AsyncMock(), cache_ttl=0.0)
    position = _fresh_entry_position()
    strat._active_strategy["BTC/USD"] = "mean_reversion"

    mr_signal = {
        "action": "close",
        "reason": "Mean Reversion: Target RSI reached (55.0), price reverted",
    }
    with _patch_regime(), \
         patch("src.execution_strategies.MeanReversionStrategy.generate_signal", return_value=mr_signal):
        scans = settings.MIN_HOLD_CONSECUTIVE_SIGNALS + 2
        actions = [
            (await strat.generate_trading_signal("BTC/USD", 100.0, position))["action"]
            for _ in range(scans)
        ]

    assert actions == ["hold"] * scans


@pytest.mark.asyncio
async def test_risk_reason_close_keeps_consecutive_override():
    """A close whose reason is a risk kind (not discretionary) still uses the
    consecutive-signal override, so the bypass is retained for risk exits
    while being unavailable to alpha closes."""
    strat = TradingStrategy(AsyncMock(), cache_ttl=0.0)
    position = _fresh_entry_position()
    strat._active_strategy["BTC/USD"] = "momentum"

    synthetic = {"action": "close", "reason": "trailing_stop_hit"}
    with _patch_regime(), \
         patch("src.execution_strategies.MomentumStrategy.generate_signal", return_value=synthetic):
        actions = []
        for _ in range(settings.MIN_HOLD_CONSECUTIVE_SIGNALS):
            actions.append(
                (await strat.generate_trading_signal("BTC/USD", 100.0, position))["action"]
            )

    # First (N-1) scans gated, the Nth allowed by the override.
    assert actions[:-1] == ["hold"] * (settings.MIN_HOLD_CONSECUTIVE_SIGNALS - 1)
    assert actions[-1] == "close"


def test_discretionary_classifier_flags_alpha_exits_and_not_risk_exits():
    assert TradingStrategy._is_discretionary_close("Momentum: Loss of bullish momentum") is True
    assert TradingStrategy._is_discretionary_close("Trend Following: Macro trend flipped bearish") is True
    assert TradingStrategy._is_discretionary_close("Mean Reversion: Target RSI reached (55.0), price reverted") is True
    assert TradingStrategy._is_discretionary_close("Breakout: Breakout failed / retraced") is True
    assert TradingStrategy._is_discretionary_close("Grid: Profit band hit (+0.5%)") is True
    assert TradingStrategy._is_discretionary_close("Scalp: Quick target reached") is True
    # Risk exits are never classified discretionary (belt-and-braces; they
    # are normally returned before the gate).
    assert TradingStrategy._is_discretionary_close("stop_loss_hit") is False
    assert TradingStrategy._is_discretionary_close("trailing_stop_hit") is False
    assert TradingStrategy._is_discretionary_close("max_hold_time_exceeded") is False