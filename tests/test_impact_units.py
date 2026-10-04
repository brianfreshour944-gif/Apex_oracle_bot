"""End-to-end: the order-book impact size cap must use the same USD units as
the L2 veto.

Binance /fapi/v1/openInterest reports OI in base-asset units (BTC, DOGE, ...),
so visible depth in USD is ``depth_coins * price`` -- which is what the L2
veto in calculate_position_size uses. _estimate_market_impact_bps instead
used ``depth_coins * 100`` (every contract valued at $100), overstating
impact for assets priced above $100 (over-shrinking BTC positions) and
understating it below $100 (letting cheap-coin positions skip the 50bps cap).
"""
from unittest.mock import AsyncMock, patch

import pytest

from src.risk import RiskManager

MAX_IMPACT_BPS = 50.0  # _apply_order_book_impact's max_acceptable_impact_bps


def _true_impact_bps(notional_usd: float, oi_coins: float, price: float) -> float:
    """Impact model with correct units: balanced book, depth = 10% of OI, half per side."""
    depth_usd = oi_coins * 0.1 / 2.0 * price
    return notional_usd / depth_usd * 0.5 * 10000


def _size(rm, symbol, price, atr, oi):
    d = {"open_interest": oi, "bid_ask_imbalance": 0.0}
    with patch("src.onchain_data.fetch_derivatives_data_sync", return_value=d):
        return rm.calculate_position_size(
            symbol, price, "trending", atr=atr, confidence=1.0,
            expected_return_pct=0.03, deriv_data=d,
        )


@pytest.fixture
def rm() -> RiskManager:
    return RiskManager(AsyncMock())


def test_high_price_asset_not_over_shrunk(rm):
    """BTC @ $50k: unscaled order is $2500; with oi=50 its true impact is
    100bps, so the cap should halve it to ~50bps -- not quarter it."""
    price, oi = 50000.0, 50.0
    size, status = _size(rm, "BTC/USD", price, atr=500.0, oi=oi)
    assert status == "ok"
    impact = _true_impact_bps(size * price, oi, price)
    assert impact == pytest.approx(MAX_IMPACT_BPS, rel=0.05), (
        f"final impact {impact:.1f}bps; size over-shrunk by unit mismatch"
    )


def test_low_price_asset_capped_at_max_impact(rm):
    """DOGE @ $0.15: book sized so the unscaled order's true impact is
    ~150bps -- passes the edge veto (300bps edge) but must still be scaled
    down to the 50bps impact cap."""
    price, atr = 0.15, 0.0015
    unscaled, _ = _size(rm, "DOGE/USD", price, atr=atr, oi=1e12)  # infinite depth
    unscaled_notional = unscaled * price
    # Choose OI so the unscaled order's true impact is 150bps.
    oi = unscaled_notional * 0.5 * 10000 / 150.0 / (0.1 / 2.0 * price)
    size, status = _size(rm, "DOGE/USD", price, atr=atr, oi=oi)
    assert status == "ok"
    impact = _true_impact_bps(size * price, oi, price)
    assert impact <= MAX_IMPACT_BPS * 1.05, (
        f"final impact {impact:.1f}bps exceeds {MAX_IMPACT_BPS}bps cap"
    )


def test_very_thin_book_scaled_to_max_impact(rm):
    """BTC @ $50k, oi=10: the unscaled $2500 order's true impact is 500bps,
    above the helper's 200bps display cap. Scaling must use the uncapped
    impact so the final order lands at the 50bps limit, not 125bps."""
    price, oi = 50000.0, 10.0
    size, status = _size(rm, "BTC/USD", price, atr=500.0, oi=oi)
    assert status == "ok"
    impact = _true_impact_bps(size * price, oi, price)
    assert impact == pytest.approx(MAX_IMPACT_BPS, rel=0.05), (
        f"final impact {impact:.1f}bps; 200bps cap under-scaled the order"
    )


def test_empty_book_side_scales_down(rm):
    """All-bids book (imbalance=+1) on a buy: zero ask depth. The L2 veto
    treats this as 200bps and, with a 300bps edge, lets the trade through;
    the size cap must not then treat it as an acceptable 50bps and pass the
    order through unscaled."""
    price = 50000.0
    deep = {"open_interest": 5_000_000.0, "bid_ask_imbalance": 0.0}
    empty_ask = {"open_interest": 5_000_000.0, "bid_ask_imbalance": 1.0}
    sizes = {}
    for name, d in (("deep", deep), ("empty_ask", empty_ask)):
        with patch("src.onchain_data.fetch_derivatives_data_sync", return_value=d):
            sizes[name] = rm.calculate_position_size(
                "BTC/USD", price, "trending", atr=500.0, confidence=1.0,
                expected_return_pct=0.03, deriv_data=d,
            )
    assert sizes["empty_ask"][1] == "ok"
    assert sizes["empty_ask"][0] <= sizes["deep"][0] * (MAX_IMPACT_BPS / 200.0) * 1.01


def test_missing_oi_disables_veto_not_rejects(rm):
    """oi=0 means no data (fetch failure / parse miss). The veto used to
    assume 1000 base-asset units of depth -- ~$75 for DOGE @ $0.15 -- and
    reject on a phantom 200bps impact. Missing data must degrade the veto to
    a no-op, matching the size cap and the fetch-failure path."""
    d = {"open_interest": 0.0, "bid_ask_imbalance": 0.0}
    with patch("src.onchain_data.fetch_derivatives_data_sync", return_value=d):
        size, status = rm.calculate_position_size(
            "DOGE/USD", 0.15, "trending", atr=0.0015, confidence=1.0,
            expected_return_pct=0.01, deriv_data=d,  # 100bps edge
        )
    assert status == "ok", status
    assert size > 0.0
