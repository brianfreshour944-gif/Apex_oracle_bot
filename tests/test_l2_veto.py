"""Tests for the L2 execution-impact hard veto in calculate_position_size.

Regression coverage: the L2 veto block used to reference `position_size`
before it was assigned, raising UnboundLocalError on every call. The except
swallowed it and set impact_bps=0, silently disabling the veto entirely.

NOTE 2026-09-21: these tests previously patched
src.onchain_data.fetch_derivatives_data (the async version), but
calculate_position_size is a synchronous function and internally calls
fetch_derivatives_data_sync (the sync wrapper) -- confirmed by reading
risk.py directly. The async-target patches never actually applied, so the
first two tests below were silently exercising a REAL (unmocked) network
call instead of the intended scenario, and the third happened to pass by
coincidence (a real network failure in the test sandbox landing in the same
graceful-degradation branch the mocked failure was meant to exercise).
Fixed to patch the function actually called, matching the same pattern
verified by reproducing the failure before this fix and the pass after it.
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.risk import RiskManager


def _deriv(oi: float, imbalance: float = 0.0) -> dict:
    return {
        "open_interest": oi,
        "bid_ask_imbalance": imbalance,
        "funding_rate": 0.0,
        "long_short_ratio": 1.0,
    }


@pytest.fixture
def rm() -> RiskManager:
    return RiskManager(AsyncMock())


def test_l2_veto_rejects_when_impact_kills_edge(rm):
    """Deep-illiquid book: impact (capped 200bps) must push net edge below the
    minimum and the trade must be REJECTED. Before the fix this scenario
    silently passed because impact was always 0 (UnboundLocalError swallowed)."""
    with patch("src.onchain_data.fetch_derivatives_data_sync",
               return_value=_deriv(oi=1.0)):  # essentially no visible depth
        size, status = rm.calculate_position_size(
            "BTC/USD", 50000.0, "trending",
            atr=500.0, confidence=1.0,
            expected_return_pct=0.005,  # 50bps edge; round-trip costs + 200bps impact < min
        )
    assert size == 0.0
    assert "insufficient edge after L2 impact" in status


def test_l2_veto_executes_and_passes_on_deep_book(rm, caplog):
    """Healthy OI: impact must be computed (mock called) and small enough that
    the trade proceeds with a normal size."""
    with patch("src.onchain_data.fetch_derivatives_data_sync",
               return_value=_deriv(oi=5_000_000.0)) as m:
        size, status = rm.calculate_position_size(
            "BTC/USD", 50000.0, "trending",
            atr=500.0, confidence=1.0,
            expected_return_pct=0.03,  # 300bps edge
        )
    m.assert_called()  # called by the veto AND by _apply_order_book_impact
    assert status == "ok"
    assert size > 0.0


def test_l2_veto_graceful_when_fetch_fails(rm, caplog):
    """If the derivatives fetch fails the veto must degrade to a no-op
    (impact=0) and the trade must proceed on tx-cost edge alone."""
    with patch("src.onchain_data.fetch_derivatives_data_sync",
               side_effect=RuntimeError("binance down")):
        size, status = rm.calculate_position_size(
            "BTC/USD", 50000.0, "trending",
            atr=500.0, confidence=1.0,
            expected_return_pct=0.03,
        )
    assert status == "ok"
    assert size > 0.0
