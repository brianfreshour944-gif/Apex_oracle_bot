"""Unit tests for TradingStrategy market regime TTL caching."""

import pytest
import time
import polars as pl
import numpy as np
from unittest.mock import AsyncMock
from src.strategies import TradingStrategy


@pytest.mark.asyncio
async def test_regime_analysis_ttl_cache():
    """Verify analyze_market_regime reuses cached analysis within TTL."""
    ex = AsyncMock()

    # Generate synthetic bar data
    rng = np.random.RandomState(42)
    closes = np.cumsum(rng.randn(50) * 0.5) + 100.0
    highs = closes + 1.0
    lows = closes - 1.0

    df = pl.DataFrame({"high": highs, "low": lows, "close": closes})
    ex.get_bars.return_value = df

    strat = TradingStrategy(ex, cache_ttl=5.0)

    # First call -> computes regime, calls ex.get_bars once
    res1 = await strat.analyze_market_regime("BTC/USD")
    assert ex.get_bars.call_count == 1

    # Second immediate call -> hits cache, ex.get_bars NOT called again
    res2 = await strat.analyze_market_regime("BTC/USD")
    assert ex.get_bars.call_count == 1
    assert res1["regime"] == res2["regime"]
