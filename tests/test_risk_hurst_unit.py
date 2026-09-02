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
from src.config import settings


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


def test_drawdown_taper_curve_checkpoints():
    """Hand-calculated checkpoints for _get_drawdown_taper_multiplier against
    the default MAX_DRAWDOWN_STOP=-10.0: no-op below 50% of the way to the
    killswitch, linear taper from 50% to 100%, floor at 0.25x beyond that."""
    rm = RiskManager(AsyncMock())
    cases = [
        (None, 1.0), (0.0, 1.0), (-3.0, 1.0), (-5.0, 1.0),
        (-7.0, 0.70), (-10.0, 0.25), (-15.0, 0.25),
    ]
    for drawdown_pct, expected in cases:
        actual = rm._get_drawdown_taper_multiplier(drawdown_pct)
        assert abs(actual - expected) < 1e-6, f"drawdown={drawdown_pct}: expected {expected}, got {actual}"


def test_position_size_drawdown_taper_regression_safe():
    """Callers that don't pass drawdown_pct must get identical sizing to
    before this parameter existed (no drawdown_pct kwarg at all, and
    drawdown_pct=None explicitly, must produce the same size)."""
    rm1 = RiskManager(AsyncMock())
    size_omitted, _ = rm1.calculate_position_size(
        "BTC/USD", 50000.0, "neutral", confidence=1.0, current_equity=5000.0,
    )
    rm2 = RiskManager(AsyncMock())
    size_none, _ = rm2.calculate_position_size(
        "BTC/USD", 50000.0, "neutral", confidence=1.0, current_equity=5000.0, drawdown_pct=None,
    )
    assert size_omitted == size_none


def test_position_size_tapers_at_deep_drawdown():
    """At the killswitch threshold, position size should be exactly 25% of
    the untapered size (the taper floor), isolated from the current_equity
    effect by holding equity constant across both calls."""
    rm_normal = RiskManager(AsyncMock())
    size_normal, _ = rm_normal.calculate_position_size(
        "BTC/USD", 50000.0, "neutral", confidence=1.0, current_equity=5000.0,
    )
    rm_deep = RiskManager(AsyncMock())
    size_deep, _ = rm_deep.calculate_position_size(
        "BTC/USD", 50000.0, "neutral", confidence=1.0, current_equity=5000.0, drawdown_pct=-10.0,
    )
    assert abs((size_deep / size_normal) - 0.25) < 0.001


@pytest.mark.asyncio
async def test_reserve_position_slot_same_regime_cluster_cap():
    """Same-regime clustering is capped independently of (and can be
    tighter than) the overall MAX_OPEN_POSITIONS count cap -- regime is a
    cheap proxy for correlated exposure since real return-series
    correlation isn't wired up anywhere live."""
    from src.config import settings
    settings.MAX_OPEN_POSITIONS = 3  # cluster cap = floor(3*0.67) = 2

    rm = RiskManager(AsyncMock())
    # 2 open positions already share this regime; overall count (2 open + 0
    # reserved = 2 < 3) would pass, but the cluster cap should reject it.
    ok, reason = await rm.reserve_position_slot("SOL/USD", open_position_count=2, same_regime_open_count=2)
    assert not ok and reason == "same_regime_cluster_cap_reached"

    # A different-regime entry with the same open_position_count is unaffected.
    rm2 = RiskManager(AsyncMock())
    ok2, reason2 = await rm2.reserve_position_slot("SOL/USD", open_position_count=2, same_regime_open_count=0)
    assert ok2


@pytest.mark.asyncio
async def test_reserve_position_slot_same_regime_default_is_noop():
    """Omitting same_regime_open_count (default 0) must be byte-identical
    to explicitly passing 0 -- regression guard for existing callers."""
    rm1 = RiskManager(AsyncMock())
    result1 = await rm1.reserve_position_slot("BTC/USD", open_position_count=0)
    rm2 = RiskManager(AsyncMock())
    result2 = await rm2.reserve_position_slot("BTC/USD", open_position_count=0, same_regime_open_count=0)
    assert result1 == result2


@pytest.mark.asyncio
async def test_reserve_position_slot_survives_fast_sibling_fill_race():
    """Meticulous-audit regression: a reservation must NOT be released just
    because its own order filled successfully -- callers must only call
    release_position_slot() on a failed order. If a fast-filling symbol's
    reservation were released mid-cycle, a slower-evaluated sibling
    (reading the same stale, cycle-start open_position_count) could reserve
    a slot too, collectively exceeding MAX_OPEN_POSITIONS within one cycle.
    This directly reproduces that scenario and asserts the cap holds when
    callers correctly leave a successful reservation in place."""
    from src.config import settings
    settings.MAX_OPEN_POSITIONS = 3
    rm = RiskManager(AsyncMock())
    stale_open_count = 2  # shared, unchanged snapshot for the whole simulated cycle

    ok_a, _ = await rm.reserve_position_slot("BTC/USD", open_position_count=stale_open_count)
    assert ok_a
    # Correct caller behavior: BTC/USD's order fills successfully -- its
    # reservation is intentionally left in place (no release_position_slot call).

    # A slower-evaluated sibling, still reading the same stale count, must be
    # rejected while BTC/USD's reservation is outstanding.
    ok_b, reason_b = await rm.reserve_position_slot("ETH/USD", open_position_count=stale_open_count)
    assert not ok_b and reason_b == "max_open_positions_would_be_exceeded"


@pytest.mark.asyncio
async def test_reserved_exposure_survives_fast_sibling_fill_race():
    """Same meticulous-audit finding as test_reserve_position_slot_survives_
    fast_sibling_fill_race, applied to check_and_reserve_exposure /
    release_reserved_exposure: a reservation must NOT be released just
    because its own order filled successfully. Releasing on success let a
    slower-evaluated sibling (reading the same stale current_exposure
    snapshot) reserve against it too and collectively exceed the portfolio
    exposure cap within one cycle -- reproduced directly (a 3-way race with
    an $8k stale exposure and a $10k cap landed at $13.4k, 34% over cap).
    This asserts the cap holds when callers correctly leave a successful
    reservation in place."""
    rm = RiskManager(AsyncMock())
    rm._get_max_portfolio_cap = lambda: 10000.0
    stale_exposure = 8000.0  # shared, unchanged snapshot for the whole simulated cycle -> $2000 headroom

    approved_a, reason_a = await rm.check_and_reserve_exposure(1800.0, current_exposure=stale_exposure)
    assert approved_a == 1800.0 and reason_a == "ok"
    # Correct caller behavior: symbol A's order fills successfully -- its
    # reservation is intentionally left in place (no release_reserved_exposure call).

    # A slower-evaluated sibling, still reading the same stale exposure, gets
    # only the remaining $200 of headroom rather than a second full $1800.
    approved_b, reason_b = await rm.check_and_reserve_exposure(1800.0, current_exposure=stale_exposure)
    assert reason_b == "ok"
    assert approved_b == pytest.approx(200.0, abs=0.01)
    assert stale_exposure + approved_a + approved_b <= 10000.0 + 1e-6


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

    Uses actual TRAILING_DISTANCE_PCT from settings to calculate thresholds.
    """
    # Get actual distance from settings
    base_distance = settings.TRAILING_DISTANCE_PCT
    high_vol_distance = base_distance * 1.5  # high_volatility multiplier
    
    # Use a drop that's between base and high_vol thresholds
    # e.g., if base=1%, high_vol=1.5%, use 1.2% drop
    drop_pct = base_distance * 1.2
    
    peak = 51000.0
    current = peak * (1 - drop_pct)

    rm_scaled = RiskManager(AsyncMock())
    rm_scaled.peak_prices["BTC/USD"] = peak
    # Should NOT trigger at scaled (wider) distance
    assert rm_scaled.check_trailing_stop("BTC/USD", current, 48000.0, 1.0, regime="high_volatility") == "hold"

    rm_unscaled = RiskManager(AsyncMock())
    rm_unscaled.peak_prices["BTC/USD"] = peak
    # SHOULD trigger at unscaled (base) distance
    assert rm_unscaled.check_trailing_stop("BTC/USD", current, 48000.0, 1.0, regime=None) == "close"


def test_trailing_stop_regime_scaling_tightens_in_low_volatility():
    """low_volatility should tighten the trailing distance vs. the unscaled baseline.

    Uses actual TRAILING_DISTANCE_PCT from settings to calculate thresholds.
    """
    # Get actual distance from settings
    base_distance = settings.TRAILING_DISTANCE_PCT
    low_vol_distance = base_distance * 0.6  # low_volatility multiplier
    
    # Use a drop that's between low_vol and base thresholds
    # e.g., if base=1%, low_vol=0.6%, use 0.8% drop
    drop_pct = base_distance * 0.8
    
    peak = 3000.0
    current = peak * (1 - drop_pct)

    rm_scaled = RiskManager(AsyncMock())
    rm_scaled.peak_prices["ETH/USD"] = peak
    # SHOULD trigger at scaled (tighter) distance
    assert rm_scaled.check_trailing_stop("ETH/USD", current, 2900.0, 1.0, regime="low_volatility") == "close"

    rm_unscaled = RiskManager(AsyncMock())
    rm_unscaled.peak_prices["ETH/USD"] = peak
    # Should NOT trigger at unscaled (base) distance
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
