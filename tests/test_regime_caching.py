"""Unit tests for TradingStrategy market regime TTL caching."""

import pytest
import time
import polars as pl
import numpy as np
from unittest.mock import AsyncMock, patch
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

    # Mock out network calls that would burn through the TTL:
    # - extract_sentiment makes real Groq/Alpaca HTTP requests (~5s on 401)
    # - fetch_derivatives_data makes real Binance Futures HTTP requests
    # Without these mocks the TTL expires before the second call, causing
    # a spurious cache miss and a second get_bars() call.
    _null_sentiment = {"sentiment_score": 0.0, "event_type": "none", "confidence": 0.0, "duration_hrs": 0.0}
    _null_deriv = {"funding_rate": 0.0, "open_interest": 0.0, "long_short_ratio": 1.0, "bid_ask_imbalance": 0.0}

    with patch("src.sentiment_analyzer.extract_sentiment", new_callable=AsyncMock, return_value=_null_sentiment), \
         patch("src.onchain_data.fetch_derivatives_data", new_callable=AsyncMock, return_value=_null_deriv):

        # First call -> computes regime, calls ex.get_bars once
        res1 = await strat.analyze_market_regime("BTC/USD")
        assert ex.get_bars.call_count == 1

        # Second immediate call -> hits cache, ex.get_bars NOT called again
        res2 = await strat.analyze_market_regime("BTC/USD")
        assert ex.get_bars.call_count == 1
        assert res1["regime"] == res2["regime"]
