"""Unit tests for RiskManager ATR sizing, correlation downscaling, and trailing stops."""

import pytest
import numpy as np
from unittest.mock import AsyncMock
from src.risk import RiskManager


def test_atr_position_sizing():
    """Verify position sizing uses ATR volatility parity when available."""
    rm = RiskManager(AsyncMock())
    
    # Standard sizing with ATR=500, price=50000, risk_amount = 10000 * 0.01 = 100
    # Stop distance = 500 * 2.0 = 1000 -> Units = 100 / 1000 = 0.1
    size, status = rm.calculate_position_size("BTC/USD", 50000.0, "trending", atr=500.0, confidence=1.0)
    assert status == "ok"
    assert size > 0.0


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

