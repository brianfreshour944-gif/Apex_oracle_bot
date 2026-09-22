"""C1: a peak/trailing/position_adds entry restored from disk for a symbol
that closed while the bot was down must not survive into a later re-entry
of that same symbol -- it silently governs the NEW position's trailing
stop using an unrelated OLD trade's price level. Found via an external
crash-recovery audit, confirmed by tracing check_trailing_stop's actual
peak-comparison logic, 2026-09-22.
"""
import os

os.environ.setdefault("ALPACA_API_KEY", "dummy")
os.environ.setdefault("ALPACA_SECRET_KEY", "dummy")

import src.bot as bot_mod
from src.risk import RiskManager


class _FakeExchange:
    pass


class _FakeStrategy:
    def __init__(self):
        self._trailing_peaks = {}
        self._trailing_troughs = {}


def test_prune_drops_peaks_for_symbols_not_held():
    bot_mod._state.risk_manager = RiskManager(_FakeExchange())
    bot_mod._state.risk_manager.peak_prices = {"BTC/USD": 120.0, "ETH/USD": 3000.0}
    bot_mod._state.strategy = _FakeStrategy()
    bot_mod._state.strategy._trailing_peaks = {"BTC/USD": 120.0}
    bot_mod._state.position_adds = {"BTC/USD": {"count": 2, "last_add_time": 0.0, "last_add_score": 0.5}}

    # Only ETH/USD is actually held right now -- BTC/USD's position closed
    # while the bot was down.
    dropped_peaks, dropped_adds = bot_mod._prune_stale_restored_state({"ETHUSD"})

    assert "BTC/USD" in dropped_peaks
    assert "BTC/USD" not in bot_mod._state.risk_manager.peak_prices
    assert "ETH/USD" in bot_mod._state.risk_manager.peak_prices
    assert "BTC/USD" not in bot_mod._state.strategy._trailing_peaks
    assert "BTC/USD" in dropped_adds
    assert "BTC/USD" not in bot_mod._state.position_adds


def test_prune_keeps_peaks_for_symbols_still_held():
    bot_mod._state.risk_manager = RiskManager(_FakeExchange())
    bot_mod._state.risk_manager.peak_prices = {"BTC/USD": 120.0}
    bot_mod._state.strategy = _FakeStrategy()
    bot_mod._state.position_adds = {}

    dropped_peaks, dropped_adds = bot_mod._prune_stale_restored_state({"BTCUSD"})

    assert dropped_peaks == []
    assert "BTC/USD" in bot_mod._state.risk_manager.peak_prices


def test_end_to_end_spurious_close_prevented_by_pruning():
    """The concrete scenario the audit traced: stale peak 120 from a closed
    trade, new entry at 105, price ticks to 116 -- must NOT trigger a
    trailing-stop close on the fresh position."""
    rm = RiskManager(_FakeExchange())
    rm.peak_prices["BTC/USD"] = 120.0  # stale, from a since-closed trade

    # Without pruning, the stale peak wrongly governs the new position.
    action_before = rm.check_trailing_stop("BTC/USD", 116.0, 105.0, 1.0)
    assert action_before == "close", "test setup didn't reproduce the underlying bug"

    # After pruning (BTC/USD not currently held), the new position starts fresh.
    bot_mod._state.risk_manager = rm
    bot_mod._state.strategy = _FakeStrategy()
    bot_mod._state.position_adds = {}
    bot_mod._prune_stale_restored_state(set())  # nothing held

    action_after = rm.check_trailing_stop("BTC/USD", 116.0, 105.0, 1.0)
    assert action_after == "hold"
