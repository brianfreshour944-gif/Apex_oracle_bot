"""add_features() per-symbol cache must never serve another window's features.

Regression (2026-10-04): the cache key was df.index[-1]. Bars arrive as polars
and are converted with .to_pandas(), giving a RangeIndex, so with limit=100
the key was always 99 -- every call after the first returned the FIRST call's
features for that symbol (and only after recomputing the fresh ones). RSI,
ATR, roll_autocorr (Hurst) etc. froze at process start: a backtest probe saw
BTC pinned at RSI 5.6 for two months, and the live bot (AlpacaExchange.get_bars
also returns polars) traded on features frozen at its last restart.
scripts/generate_replay_dataset.py had worked around it by clearing the cache.
"""
import numpy as np
import polars as pl
import pytest

import src.feature_engineering as fe


@pytest.fixture(autouse=True)
def _clear_cache():
    fe._FEATURE_CACHE.clear()
    yield
    fe._FEATURE_CACHE.clear()


def _bars(trend: float, ts_col: str | None = "timestamp", start: int = 0, n: int = 100):
    c = 100 + np.cumsum(np.where(np.arange(n) % 3 == 0, -0.3, trend))
    data = {"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": np.ones(n)}
    if ts_col:
        data[ts_col] = [f"2026-09-{1 + (start + i) // 24:02d}T{(start + i) % 24:02d}:00:00+00:00" for i in range(n)]
    return pl.DataFrame(data).to_pandas()   # RangeIndex, like add_multi_timeframe_features


def _rsi(df, symbol="BTC/USD"):
    return float(fe.add_features(df, symbol=symbol)["rsi"].iloc[-1])


@pytest.mark.parametrize("ts_col", ["timestamp", "t"])
def test_different_windows_same_length_are_not_served_from_cache(ts_col):
    falling = _rsi(_bars(-1.0, ts_col))
    rising = _rsi(_bars(+1.0, ts_col, start=1))
    assert rising != falling
    assert rising == pytest.approx(_rsi(_bars(+1.0, ts_col, start=1), symbol=""))  # uncached truth


def test_same_window_is_served_from_cache():
    df = _bars(+1.0)
    first = fe.add_features(df, symbol="BTC/USD")
    assert fe.add_features(df.copy(), symbol="BTC/USD") is first


def test_no_timestamp_column_means_no_caching():
    a = _rsi(_bars(-1.0, ts_col=None))
    b = _rsi(_bars(+1.0, ts_col=None))
    assert a != b
    assert "BTC/USD" not in fe._FEATURE_CACHE


def test_symbols_cached_independently():
    btc = _rsi(_bars(-1.0), symbol="BTC/USD")
    eth = _rsi(_bars(+1.0), symbol="ETH/USD")
    assert btc != eth


def test_in_progress_bar_update_is_not_served_from_cache():
    """Live daily bars end with today's in-progress bar: same timestamp all
    day, changing close. A timestamp-only key would freeze features daily."""
    df = _bars(+1.0)
    first = _rsi(df)
    moved = df.copy()
    moved.loc[moved.index[-1], ["close", "low"]] = moved["close"].iloc[-1] * 0.95
    assert _rsi(moved) != first
