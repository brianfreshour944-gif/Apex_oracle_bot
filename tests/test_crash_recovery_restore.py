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


# ── Daily-loss baseline must not survive a day boundary across a restart ──────
# Regression (2026-10-04 deploy log): the bot tripped the daily-loss killswitch
# 1s after startup with no trades -- "DAILY LOSS LIMIT HIT: $-2.92" on a ~$97
# account whose restored peak_equity was 100.00. start_of_day_equity was
# persisted without its date and RiskManager.last_check_time is reset to "now"
# at construction, so a baseline from an earlier day was compared against
# today's equity and cumulative multi-day losses counted as "today".

@pytest.fixture
def async_state():
    from unittest.mock import AsyncMock
    ex = AsyncMock()
    ex.get_positions.return_value = []
    ex.get_account.return_value = {"equity": 97.08, "cash": 97.08, "portfolio_value": 97.08}
    bot_mod._state.risk_manager = RiskManager(ex)
    yield bot_mod._state
    bot_mod._state.risk_manager = None


def _today(offset_days: int = 0) -> str:
    from datetime import UTC, datetime, timedelta
    return (datetime.now(UTC) - timedelta(days=offset_days)).date().isoformat()


@pytest.mark.asyncio
async def test_stale_day_baseline_is_not_restored(async_state):
    bot_mod.apply_crash_recovery_state({
        "risk_peak_equity": 100.0,
        "risk_start_of_day_equity": 100.0,
        "risk_daily_pnl": -2.5,
        "risk_start_of_day_date": _today(offset_days=1),
    })
    status = await async_state.risk_manager.update_account_status()
    assert status["status"] != "killswitch_activated", status
    assert async_state.risk_manager.is_killswitch_active() is False


@pytest.mark.asyncio
async def test_undated_legacy_baseline_is_not_restored(async_state):
    """State files written before the date field existed can't prove the
    baseline is from today, so it must not be trusted."""
    bot_mod.apply_crash_recovery_state({
        "risk_peak_equity": 100.0,
        "risk_start_of_day_equity": 100.0,
    })
    status = await async_state.risk_manager.update_account_status()
    assert status["status"] != "killswitch_activated", status


@pytest.mark.asyncio
async def test_same_day_baseline_still_restored_and_trips(async_state):
    """A same-day crash/restart must keep today's losses: no loophole."""
    bot_mod.apply_crash_recovery_state({
        "risk_peak_equity": 100.0,
        "risk_start_of_day_equity": 100.0,
        "risk_start_of_day_date": _today(),
    })
    status = await async_state.risk_manager.update_account_status()
    assert status["status"] == "killswitch_activated"
    assert status["reason"] == "daily_loss_limit_exceeded"
