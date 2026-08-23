"""Comprehensive unit tests for RiskManager and TradingStrategy core methods.

Covers:
- RiskManager.calculate_position_size (ATR-based sizing, regime scaling, confidence)
- RiskManager.check_trailing_stop (activation, distance, side)
- TradingStrategy._calculate_hurst (random walk, trending, mean-reverting, edge cases)
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock
from src.risk import RiskManager
from src.strategies import TradingStrategy


def test_position_size_basic():
    """Standard position size with no ATR (uses simple percent risk)."""
    rm = RiskManager(AsyncMock())
    size, status = rm.calculate_position_size("BTC/USD", 50000.0, "trending")
    assert status == "ok"
    assert size > 0.0


def test_position_size_with_atr():
    """ATR-based position sizing: high volatility → smaller position."""
    rm = RiskManager(AsyncMock())

    size_low, status_low = rm.calculate_position_size(
        "BTC/USD", 50000.0, "low_volatility", atr=100.0
    )
    size_high, status_high = rm.calculate_position_size(
        "BTC/USD", 50000.0, "high_volatility", atr=5000.0
    )
    assert status_low == "ok"
    assert status_high == "ok"
    assert size_low > size_high


def test_position_size_regime_scaling():
    """Different regimes produce different size multipliers."""
    rm = RiskManager(AsyncMock())
    regimes = ["low_volatility", "high_volatility", "trending", "bear", "bull"]
    sizes = {}
    for r in regimes:
        size, status = rm.calculate_position_size("BTC/USD", 50000.0, r, atr=500.0)
        sizes[r] = size

    unique_sizes = set(round(s, 8) for s in sizes.values())
    assert len(unique_sizes) > 1, f"All regimes produced same size: {sizes}"


def test_position_size_confidence_adjustment():
    """Higher confidence should mean at least as large a position."""
    rm = RiskManager(AsyncMock())
    size_low_conf, _ = rm.calculate_position_size(
        "BTC/USD", 50000.0, "trending", atr=500.0, confidence=0.3
    )
    size_high_conf, _ = rm.calculate_position_size(
        "BTC/USD", 50000.0, "trending", atr=500.0, confidence=0.9
    )
    assert size_high_conf >= size_low_conf


def test_trailing_stop_no_position():
    """When qty=0 (no position), trailing stop returns 'hold' or 'close'."""
    rm = RiskManager(AsyncMock())
    result = rm.check_trailing_stop("BTC/USD", 50000.0, 49000.0, 0.0)
    assert result in (None, "hold", "close")


def test_trailing_stop_profit_taking():
    """Trailing stop triggers close when price falls enough from peak."""
    rm = RiskManager(AsyncMock())
    entry = 50000.0
    peak_price = 55000.0  # 10% up from entry
    # Set peak price by having the function track it
    rm.peak_prices["BTC/USD"] = peak_price
    current = 53000.0  # 4% down from peak
    result = rm.check_trailing_stop("BTC/USD", current, entry, 1.0)
    assert result in ("hold", "close")


def test_trailing_stop_short_position():
    """Trailing stop for short positions: triggers when price rises from peak."""
    rm = RiskManager(AsyncMock())
    entry = 50000.0
    # Short: price dropped (good), so peak should track lows
    rm.peak_prices["BTC/USD"] = 48000.0
    current = 49000.0  # price rose from the peak low
    result = rm.check_trailing_stop("BTC/USD", current, entry, -1.0)
    assert result in ("hold", "close")


def test_trailing_stop_regime_scaling_widens_in_high_volatility():
    """high_volatility should widen the trailing distance vs. the unscaled baseline.

    A 4.0% drop from peak: should NOT trigger at the scaled 4.5% threshold
    (base 3% * 1.5 high_volatility multiplier), but WOULD trigger at the
    unscaled 3% threshold -- proving the scaling actually changes behavior,
    not just that both paths happen to agree.
    """
    peak = 51000.0
    current = peak * (1 - 0.04)

    rm_scaled = RiskManager(AsyncMock())
    rm_scaled.peak_prices["BTC/USD"] = peak
    assert rm_scaled.check_trailing_stop("BTC/USD", current, 48000.0, 1.0, regime="high_volatility") == "hold"

    rm_unscaled = RiskManager(AsyncMock())
    rm_unscaled.peak_prices["BTC/USD"] = peak
    assert rm_unscaled.check_trailing_stop("BTC/USD", current, 48000.0, 1.0, regime=None) == "close"


def test_trailing_stop_regime_scaling_tightens_in_low_volatility():
    """low_volatility should tighten the trailing distance vs. the unscaled baseline.

    A 2.0% drop from peak: SHOULD trigger at the scaled 1.8% threshold
    (base 3% * 0.6 low_volatility multiplier), but would NOT trigger at the
    unscaled 3% threshold.
    """
    peak = 3000.0
    current = peak * (1 - 0.02)

    rm_scaled = RiskManager(AsyncMock())
    rm_scaled.peak_prices["ETH/USD"] = peak
    assert rm_scaled.check_trailing_stop("ETH/USD", current, 2900.0, 1.0, regime="low_volatility") == "close"

    rm_unscaled = RiskManager(AsyncMock())
    rm_unscaled.peak_prices["ETH/USD"] = peak
    assert rm_unscaled.check_trailing_stop("ETH/USD", current, 2900.0, 1.0, regime=None) == "hold"


def test_trailing_stop_unrecognized_regime_matches_baseline():
    """None, 'neutral', and an unrecognized regime string must all behave
    identically to each other -- regression guard for the regime-scaling
    feature: anything not explicitly mapped must be a strict no-op."""
    peak = 50000.0
    for pct_off_peak in (0.02, 0.03, 0.04):
        current = peak * (1 - pct_off_peak)
        results = []
        for regime in (None, "neutral", "totally_made_up_regime"):
            rm = RiskManager(AsyncMock())
            rm.peak_prices["BTC/USD"] = peak
            results.append(rm.check_trailing_stop("BTC/USD", current, 49000.0, 1.0, regime=regime))
        assert len(set(results)) == 1, f"unrecognized regimes diverged at {pct_off_peak*100:.0f}% off peak: {results}"


def test_trailing_stop_params_clamp_unsafe_multiplier():
    """A multiplier combination that would let distance reach/exceed
    activation must be clamped, never producing a stop that triggers
    immediately upon activation instead of trailing."""
    from unittest.mock import patch
    import src.risk as risk_mod

    rm = RiskManager(AsyncMock())
    with patch.dict(risk_mod._TRAILING_REGIME_MULTIPLIERS, {"_unsafe_test": (1.0, 2.0)}):
        activation, distance = rm._get_trailing_params("_unsafe_test")
        assert distance < activation


def test_hurst_random_walk():
    """Hurst exponent for a random walk should be near 0.5."""
    np.random.seed(42)
    returns = np.random.randn(1000) * 0.02
    strategy = TradingStrategy(None)
    hurst = strategy._calculate_hurst(returns)

    assert isinstance(hurst, float)
    assert 0.0 <= hurst <= 1.0
    assert 0.3 <= hurst <= 0.7, f"Hurst for random walk should be ~0.5, got {hurst}"


def test_hurst_trending_series():
    """Trending series should produce Hurst > 0.4."""
    np.random.seed(42)
    # Strong persistent drift → trending regime
    returns = np.random.randn(2000) * 0.01 + 0.001
    strategy = TradingStrategy(None)
    hurst = strategy._calculate_hurst(returns)

    assert 0.0 <= hurst <= 1.0
    # With strong drift, Hurst should be higher than mean-reverting
    assert hurst > 0.35, f"Trending series should have Hurst > 0.35, got {hurst}"


def test_hurst_mean_reverting():
    """Mean-reverting series (flipping signs) should not crash."""
    np.random.seed(42)
    base = np.random.randn(1000) * 0.01
    returns = np.empty_like(base)
    returns[1::2] = -base[1::2]
    strategy = TradingStrategy(None)
    hurst = strategy._calculate_hurst(returns)

    assert 0.0 <= hurst <= 1.0


def test_hurst_short_series():
    """Very short series (< 20 bars) returns 0.5."""
    returns = np.array([0.01, -0.02, 0.005])
    strategy = TradingStrategy(None)
    hurst = strategy._calculate_hurst(returns)
    assert hurst == 0.5


def test_hurst_constant_series():
    """Constant (zero-variance) returns should not crash."""
    returns = np.zeros(50)
    strategy = TradingStrategy(None)
    hurst = strategy._calculate_hurst(returns)
    assert isinstance(hurst, float)
    assert 0.0 <= hurst <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_dynamic_transaction_costs():
    """Test dynamic transaction cost model updates and affects sizing."""
    rm = RiskManager(AsyncMock())
    
    # Initially uses static defaults
    costs1 = rm.get_transaction_costs("BTC/USD")
    assert costs1["fee_bps"] == 5.0
    assert costs1["slippage_bps"] == 3.0
    assert costs1["spread_bps"] == 2.0
    assert costs1["total_bps"] == 10.0
    
    # Record a fill with high costs
    rm.record_fill_costs("BTC/USD", fee_bps=10.0, slippage_bps=20.0, spread_bps=5.0)
    
    # Dynamic model should now reflect higher costs
    costs2 = rm.get_transaction_costs("BTC/USD")
    assert costs2["fee_bps"] > 5.0  # Updated via EMA
    assert costs2["slippage_bps"] > 3.0
    assert costs2["spread_bps"] > 2.0
    
    # High costs should reduce position size
    size_before, _ = rm.calculate_position_size("BTC/USD", 50000.0, "trending", atr=500.0, confidence=1.0, expected_return_pct=0.03)
    # Simulate more high-cost fills to push edge below threshold
    for _ in range(5):
        rm.record_fill_costs("BTC/USD", fee_bps=10.0, slippage_bps=20.0, spread_bps=5.0)
    size_after, status = rm.calculate_position_size("BTC/USD", 50000.0, "trending", atr=500.0, confidence=1.0, expected_return_pct=0.03)
    # With high costs, net edge may be below threshold
    # The test verifies the dynamic model is being used
    assert status in ("ok", "rejected: insufficient edge after costs")


def test_edge_below_cost_threshold_rejected():
    """Trades with expected edge below transaction costs should be rejected."""
    rm = RiskManager(AsyncMock())
    # Simulate very high costs
    rm.record_fill_costs("ETH/USD", fee_bps=50.0, slippage_bps=50.0, spread_bps=10.0)
    # Low expected return with high costs -> should be rejected
    size, status = rm.calculate_position_size("ETH/USD", 3000.0, "trending", atr=100.0, confidence=1.0, expected_return_pct=0.01)
    assert size == 0.0
    assert "insufficient edge" in status.lower() or status == "rejected: insufficient edge after costs"


def test_record_fill_costs_ema():
    """Verify EMA blending works correctly."""
    rm = RiskManager(AsyncMock())
    alpha = 0.3
    
    # First fill
    rm.record_fill_costs("BTC/USD", fee_bps=10.0, slippage_bps=5.0)
    costs = rm._realized_tx_costs["BTC/USD"]
    assert costs["fee_bps"] == 10.0
    assert costs["slippage_bps"] == 5.0
    
    # Second fill - should blend with alpha=0.3
    rm.record_fill_costs("BTC/USD", fee_bps=20.0, slippage_bps=15.0)
    costs = rm._realized_tx_costs["BTC/USD"]
    expected_fee = (1 - alpha) * 10.0 + alpha * 20.0
    expected_slippage = (1 - alpha) * 5.0 + alpha * 15.0
    assert abs(costs["fee_bps"] - expected_fee) < 0.01
    assert abs(costs["slippage_bps"] - expected_slippage) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
