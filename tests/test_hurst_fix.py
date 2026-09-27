"""Test to verify the Hurst calculation fix works correctly."""

import numpy as np


def test_hurst_calculation():
    """Test that Hurst calculation doesn't crash and returns a valid exponent."""
    from src.strategies import TradingStrategy

    # Create a dummy strategy (no exchange needed for _calculate_hurst)
    strategy = TradingStrategy(None)

    np.random.seed(42)
    returns = np.random.randn(100) * 0.02

    hurst = strategy._calculate_hurst(returns)

    assert isinstance(hurst, float), "Hurst should be a float"
    assert 0 <= hurst <= 1, f"Hurst should be between 0 and 1, got {hurst}"