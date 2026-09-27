"""Crash-recovery restore contract (audit F1-F3).

Production restores persisted state in run_trading_bot() after building the
RiskManager/TradingStrategy, via apply_crash_recovery_state(). These tests pin
that contract directly instead of relying on construction-time side effects.
"""
import os
import time

os.environ.setdefault("ALPACA_API_KEY", "dummy")
os.environ.setdefault("ALPACA_SECRET_KEY", "dummy")

import pytest

import src.bot as bot_mod
from src.config import MAX_POSITION_ADDS
from src.risk import RiskManager
from src.strategies import TradingStrategy


class _FakeExchange:
    def get_positions(self):
        return []

    def get_account(self):
        return {}


@pytest.fixture
def live_state():
    bot_mod._state.risk_manager = RiskManager(_FakeExchange())
    bot_mod._state.strategy = TradingStrategy(exchange=None)
    bot_mod._state.cooldowns = {}
    bot_mod._state.position_adds = {}
    yield bot_mod._state
    bot_mod._state.risk_manager = None
    bot_mod._state.strategy = None


def test_restores_peaks_cooldowns_and_scale_in_cap(live_state):
    future = time.time() + 999
    counts = bot_mod.apply_crash_recovery_state({
        "peak_prices": {"BTC/USD": 120.0},
        "trailing_peaks": {"BTC/USD": 120.0},
        "trailing_troughs": {"BTC/USD": 90.0},
        "cooldowns": {"BTC/USD": future},
        "position_adds": {"BTC/USD": {"count": MAX_POSITION_ADDS, "last_add_time": time.time()}},
        "risk_peak_equity": 10_000.0,
    })

    assert counts == {"peak_prices": 1, "trailing_peaks": 1, "trailing_troughs": 1,
                      "cooldowns": 1, "position_adds": 1}
    assert live_state.risk_manager.peak_prices["BTC/USD"] == 120.0
    assert live_state.risk_manager.peak_equity == 10_000.0
    assert live_state.strategy._trailing_peaks["BTC/USD"] == 120.0
    assert live_state.cooldowns["BTC/USD"] == future
    assert live_state.position_adds["BTC/USD"]["count"] == MAX_POSITION_ADDS


def test_restored_peak_fires_trailing_stop(live_state):
    bot_mod.apply_crash_recovery_state({"peak_prices": {"BTC/USD": 120.0}})
    # Without the restored peak, the peak re-anchors at 108 and the stop holds.
    assert live_state.risk_manager.check_trailing_stop("BTC/USD", 108.0, 100.0, 1.0, regime="trending") == "close"


@pytest.mark.parametrize("bad", [None, {}, {"peak_prices": "not-a-dict"}])
def test_restore_is_fail_safe(live_state, bad):
    counts = bot_mod.apply_crash_recovery_state(bad)
    assert counts == {"peak_prices": 0, "trailing_peaks": 0, "trailing_troughs": 0,
                      "cooldowns": 0, "position_adds": 0}
