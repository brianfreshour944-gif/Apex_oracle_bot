"""
Unit tests for AlpacaExchange, rewritten against the current implementation.

NOTE ON HISTORY: the previous version of this file tested an older shape of
this class (`ex.client.post()`/`.get()` raw HTTP calls, plus a standalone
`RateLimiter` class). That implementation no longer exists — `AlpacaExchange`
was refactored to use the official `alpaca-py` SDK (`self.trading_client`,
`self.data_client`) with `tenacity` retry decorators instead of a manual
token-bucket rate limiter. These tests cover the class as it exists today.
"""
from unittest.mock import MagicMock

import pytest

from src.exchange import AlpacaExchange
from alpaca.data.timeframe import TimeFrameUnit


@pytest.fixture
def exchange():
    """An AlpacaExchange instance with no real credentials — safe for unit tests
    since we never call .load() or hit the network; we inject mocked clients
    directly instead."""
    return AlpacaExchange()


class TestParseTimeframe:
    """_parse_timeframe is pure logic — no network or mocking needed."""

    def test_minutes(self, exchange):
        tf = exchange._parse_timeframe("5Min")
        assert tf.amount == 5
        assert tf.unit == TimeFrameUnit.Minute

    def test_hours(self, exchange):
        tf = exchange._parse_timeframe("1Hour")
        assert tf.amount == 1
        assert tf.unit == TimeFrameUnit.Hour

    def test_days(self, exchange):
        tf = exchange._parse_timeframe("1Day")
        assert tf.amount == 1
        assert tf.unit == TimeFrameUnit.Day

    def test_unrecognized_format_defaults_to_day(self, exchange):
        # TimeFrame doesn't implement __eq__, so compare fields directly
        tf = exchange._parse_timeframe("garbage")
        expected = exchange._parse_timeframe("1Day")
        assert tf.amount == expected.amount
        assert tf.unit == expected.unit


class TestGetAccount:
    @pytest.mark.asyncio
    async def test_formats_account_fields_correctly(self, exchange):
        fake_account = MagicMock()
        fake_account.id = "acct_123"
        fake_account.status = "ACTIVE"
        fake_account.currency = "USD"
        fake_account.buying_power = "1000.50"
        fake_account.equity = "5000.00"
        fake_account.portfolio_value = "5000.00"

        exchange.trading_client = MagicMock()
        exchange.trading_client.get_account = MagicMock(return_value=fake_account)

        result = await exchange.get_account()

        assert result["id"] == "acct_123"
        assert result["buying_power"] == pytest.approx(1000.50)
        assert result["equity"] == pytest.approx(5000.00)


class TestGetPositions:
    @pytest.mark.asyncio
    async def test_formats_position_list_correctly(self, exchange):
        fake_position = MagicMock()
        fake_position.symbol = "BTC/USD"
        fake_position.qty = "2.5"
        fake_position.avg_entry_price = "100.0"
        fake_position.market_value = "260.0"
        fake_position.unrealized_pl = "10.0"
        fake_position.unrealized_plpc = "0.04"

        exchange.trading_client = MagicMock()
        exchange.trading_client.get_all_positions = MagicMock(return_value=[fake_position])

        result = await exchange.get_positions()

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC/USD"
        assert result[0]["qty"] == pytest.approx(2.5)


class TestCreateOrderConfirmation:
    @pytest.mark.asyncio
    async def test_returns_final_status_once_filled(self, exchange):
        fake_order = MagicMock()
        fake_order.id = "ord_123"
        fake_order.symbol = "BTC/USD"
        fake_order.qty = "1.0"
        fake_order.status = "pending_new"

        fake_filled_order = MagicMock()
        fake_filled_order.id = "ord_123"
        fake_filled_order.symbol = "BTC/USD"
        fake_filled_order.qty = "1.0"
        fake_filled_order.filled_qty = "1.0"
        fake_filled_order.status = "filled"
        fake_filled_order.side = "buy"
        fake_filled_order.type = "market"

        exchange.trading_client = MagicMock()
        exchange.trading_client.submit_order = MagicMock(return_value=fake_order)
        exchange.trading_client.get_order_by_id = MagicMock(return_value=fake_filled_order)

        result = await exchange.create_order(
            "BTC/USD", 1.0, "buy", confirm=True, confirm_timeout=2.0
        )

        assert result["status"] == "filled"
        assert result["id"] == "ord_123"

    @pytest.mark.asyncio
    async def test_returns_immediately_when_confirm_is_false(self, exchange):
        fake_order = MagicMock()
        fake_order.id = "ord_456"
        fake_order.symbol = "ETH/USD"
        fake_order.qty = "3.0"
        fake_order.status = "pending_new"

        exchange.trading_client = MagicMock()
        exchange.trading_client.submit_order = MagicMock(return_value=fake_order)

        result = await exchange.create_order(
            "ETH/USD", 3.0, "buy", confirm=False
        )

        assert result["id"] == "ord_456"