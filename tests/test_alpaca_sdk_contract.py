"""Regression tests for audit findings locked to the installed alpaca-py + runtime.

These pin two fixes:
  * H1/M2 - exchange.py must not read `Order.commission` (absent on alpaca-py>=0.43,
            extra='ignore' drops it from API payloads too).
  * H2    - committee.calculate_confidence_size_multiplier must reject non-finite
            scores (NaN/inf) instead of returning the max 1.75x multiplier.

Run:  python -m pytest tests/test_alpaca_sdk_contract.py -q
"""
import math

import pytest

pytest.importorskip("alpaca")
from alpaca.trading.models import Order


def test_order_model_has_no_commission_field():
    """Locks the SDK contract: `Order` must NOT expose `.commission` and must not
    use extra='allow' (so a `commission` key in an API payload is invisible)."""
    assert "commission" not in Order.model_fields
    assert Order.model_config.get("extra", "ignore") != "allow"


def test_order_model_drops_payload_commission():
    """Even a payload carrying `commission` is silently discarded by pydantic
    (extra='ignore') -> getattr(order, 'commission') must be None/absent."""
    order = Order.model_validate(
        {
            "id": "00000000-0000-0000-0000-000000000000",
            "client_order_id": "cid",
            "symbol": "AAPL",
            "qty": "1",
            "side": "buy",
            "type": "market",
            "order_class": "simple",
            "extended_hours": False,
            "time_in_force": "day",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "submitted_at": "2024-01-01T00:00:00Z",
            "status": "new",
            "filled_qty": "1",
            "filled_avg_price": "10",
            "commission": "0.5",  # must be dropped, not accessible
            "filled_at": "2024-01-01T00:00:05Z",
        }
    )
    assert getattr(order, "commission", None) is None
    assert getattr(order, "timestamp", None) is None
    # Fields the bot actually relies on MUST survive:
    assert float(order.filled_avg_price) == 10.0
    assert float(order.filled_qty) == 1.0


def test_size_multiplier_rejects_non_finite_score():
    from src.committee.committee import calculate_confidence_size_multiplier

    for bad in (float("nan"), float("inf"), float("-inf")):
        assert calculate_confidence_size_multiplier(bad, 0.0, 0.15) == 0.0, bad

    # finite sub-threshold -> still 0.0
    assert calculate_confidence_size_multiplier(0.1, 0.0, 0.15) == 0.0
    # finite above-threshold -> finite and scales up (matches test_committee)
    mult = calculate_confidence_size_multiplier(0.83, 0.0, 0.15)
    assert math.isfinite(mult) and mult >= 1.40
