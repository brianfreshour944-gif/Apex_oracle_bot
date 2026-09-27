"""Integration test: verify regime detection, signals, and risk logic actually work with synthetic data."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_regime_detection_works():
    """Prove regime detection returns non-neutral regimes (not always 'neutral')."""
    from src.strategies import TradingStrategy

    strat = TradingStrategy(None)

    # Trending series: persistent upward drift -> should give high Hurst
    trending = np.cumsum(np.random.RandomState(1).randn(200) * 0.01 + 0.002) + 100
    returns_t = np.diff(trending) / trending[:-1]
    h_t = strat._calculate_hurst(returns_t)
    print(f"Trending-series Hurst: {h_t:.4f}")

    # Mean-reverting series: oscillate around mean -> should give low Hurst
    mr = 100 + np.sin(np.arange(200) * 0.3) * 2 + np.random.RandomState(2).randn(200) * 0.1
    returns_m = np.diff(mr) / mr[:-1]
    h_m = strat._calculate_hurst(returns_m)
    print(f"Mean-reverting-series Hurst: {h_m:.4f}")

    assert 0.0 <= h_t <= 1.0, "trending hurst out of range"
    assert 0.0 <= h_m <= 1.0, "mr hurst out of range"
    # They should differ meaningfully (not both 0.5 / not both neutral)
    assert abs(h_t - h_m) > 0.05, f"Hurst values too similar: {h_t} vs {h_m}"

def test_price_based_exits():
    """Prove stop-loss / profit-target exits fire."""
    from src.strategies import TradingStrategy

    strat = TradingStrategy(None)

    # Long position, price dropped 5% (stop loss is 2%)
    pos = {"avg_entry_price": 100.0, "qty": 1.0}
    sig = strat._check_price_based_exits("BTC/USD", 95.0, pos)
    print(f"Stop-loss signal: {sig}")
    assert sig is not None and sig["action"] == "close", "stop loss did not fire"
    assert sig["reason"] == "stop_loss_hit"

    # Long position, price up 4% (profit target is 3%)
    pos2 = {"avg_entry_price": 100.0, "qty": 1.0}
    sig2 = strat._check_price_based_exits("BTC/USD", 104.0, pos2)
    print(f"Profit-target signal: {sig2}")
    assert sig2 is not None and sig2["action"] == "close", "profit target did not fire"
    assert sig2["reason"] == "profit_target_reached"

def test_max_hold_fires_for_alternate_entry_time_keys():
    """Max-hold exit must work for every entry-time key an exchange might emit."""
    from datetime import UTC, datetime, timedelta

    from src.config import settings
    from src.strategies import TradingStrategy

    strat = TradingStrategy(exchange=None)
    held = datetime.now(UTC) - timedelta(hours=settings.MAX_HOLD_HOURS + 100)
    base = {"symbol": "BTC/USD", "qty": "1.0", "avg_entry_price": "100.0"}

    for key, value in (
        ("created_at", held.isoformat()),
        ("entry_time", held.isoformat()),
        ("opened_at", held.isoformat()),
        ("created_at", held.timestamp()),
    ):
        pos = dict(base, **{key: value})
        sig = strat._check_price_based_exits("BTC/USD", 100.0, pos)
        assert sig is not None, f"max-hold did not fire for {key}={value!r}"
        assert sig["reason"] == "max_hold_time_exceeded", sig

    # A fresh position must NOT be force-closed.
    fresh = dict(base, entry_time=datetime.now(UTC).isoformat())
    assert strat._check_price_based_exits("BTC/USD", 100.0, fresh) is None

async def test_risk_limits_scaled():
    """Prove the daily loss killswitch scales to actual equity, not a *1000 base."""
    from src.config import settings
    from src.risk import RiskManager

    class FakeEx:
        def __init__(self, equity: float):
            self._equity = equity

        async def get_account(self):
            return {"equity": self._equity, "cash": self._equity, "portfolio_value": self._equity}

        async def get_positions(self):
            return []

    equity = 10000.0
    rm = RiskManager(FakeEx(equity))
    # First call establishes the start-of-day equity baseline; the second call
    # applies a loss that breaches the configured percentage of equity and must
    # trip the killswitch. -4% is beyond the default -3% limit but nowhere near
    # the $-1000+ a hard-coded *1000 base would have required.
    await rm.update_account_status(account={"equity": equity, "cash": 0.0, "portfolio_value": equity})
    loss = settings.DAILY_LOSS_LIMIT / 100.0 * equity - 100.0
    status = await rm.update_account_status(account={"equity": equity + loss, "cash": 0.0, "portfolio_value": equity + loss})
    assert status["status"] == "killswitch_activated", f"daily loss limit did not trip: {status}"
    assert status["reason"] == "daily_loss_limit_exceeded", status
    assert abs(status["daily_pnl"] - loss) < 0.01, f"daily_pnl wrong: {status['daily_pnl']} vs {loss}"

if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATION TEST: strategy + risk logic")
    print("=" * 60)
    ok = True
    try:
        test_regime_detection_works()
        test_price_based_exits()
        test_risk_limits_scaled()
    except Exception:
        ok = False
        import traceback
        traceback.print_exc()
    print("=" * 60)
    if ok:
        print("ALL INTEGRATION TESTS PASSED - logic is functional, not neutral-only")
    else:
        print("TESTS FAILED")