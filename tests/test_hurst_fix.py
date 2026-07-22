"""Test to verify the Hurst calculation fix works correctly."""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_hurst_calculation():
    """Test that Hurst calculation doesn't crash."""
    try:
        from strategies import TradingStrategy

        # Create a dummy strategy (no exchange needed for _calculate_hurst)
        strategy = TradingStrategy(None)

        # Test with random returns
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02

        hurst = strategy._calculate_hurst(returns)
        print(f"Hurst exponent: {hurst:.4f}")

        # Verify it's a valid float between 0 and 1
        assert isinstance(hurst, float), "Hurst should be a float"
        assert 0 <= hurst <= 1, f"Hurst should be between 0 and 1, got {hurst}"

        print("Hurst calculation test passed!")
        return True

    except Exception as e:
        print(f"Hurst calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Hurst calculation fix...")
    print("=" * 50)

    if test_hurst_calculation():
        print("\nHurst fix verified - no more crash!")
    else:
        print("\nHurst fix failed!")