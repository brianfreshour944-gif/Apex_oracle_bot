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
from tenacity import stop_after_attempt, RetryError

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


class TestCreateOrderIdempotentRecovery:
    """
    submit_order can fail in a way that's ambiguous about whether Alpaca
    actually processed the order server-side -- e.g. the HTTP response is
    lost to a network error right after the order was accepted. The old
    behavior: the bare @retry decorator (no exception filter) would just
    resubmit with the same client_order_id, and if Alpaca treats that as a
    duplicate rather than deduping it, the caller sees a failure for a
    trade that actually went through -- position/exposure tracking never
    learns about it.
    """

    @pytest.mark.asyncio
    async def test_uses_existing_order_instead_of_resubmitting(self, exchange):
        exchange.trading_client = MagicMock()
        exchange.trading_client.submit_order = MagicMock(side_effect=RuntimeError("connection reset"))

        existing_order = MagicMock()
        existing_order.id = "existing_order_999"
        existing_order.symbol = "BTC/USD"
        existing_order.qty = "1.0"
        existing_order.status = "accepted"
        exchange.trading_client.get_order_by_client_id = MagicMock(return_value=existing_order)

        result = await exchange.create_order(
            "BTC/USD", 1.0, "buy", confirm=False, client_order_id="my_client_order_id_123"
        )

        assert result["id"] == "existing_order_999"
        assert result["status"] == "accepted"
        # The whole point: must NOT have resubmitted a duplicate.
        exchange.trading_client.submit_order.assert_called_once()
        exchange.trading_client.get_order_by_client_id.assert_called_once_with("my_client_order_id_123")

    @pytest.mark.asyncio
    async def test_reraises_original_error_when_no_matching_order_exists(self, exchange):
        """If the lookup genuinely finds nothing, this must behave exactly
        as before: propagate the original error (letting @retry decide
        whether to resubmit) rather than silently swallowing it.

        create_order's @retry has no reraise=True (pre-existing, unrelated
        to this fix), so tenacity wraps the exhausted final attempt in its
        own RetryError rather than propagating RuntimeError directly --
        unwrap it to check the real cause."""
        exchange.trading_client = MagicMock()
        exchange.trading_client.submit_order = MagicMock(side_effect=RuntimeError("connection reset"))
        exchange.trading_client.get_order_by_client_id = MagicMock(side_effect=RuntimeError("404 not found"))

        # Make the outer @retry give up after one attempt so this test doesn't
        # sit through 5 real exponential-backoff sleeps.
        original_stop = exchange.create_order.retry.stop
        try:
            exchange.create_order.retry.stop = stop_after_attempt(1)
            with pytest.raises(RetryError) as exc_info:
                await exchange.create_order(
                    "BTC/USD", 1.0, "buy", confirm=False, client_order_id="my_client_order_id_456"
                )
            cause = exc_info.value.last_attempt.exception()
            assert isinstance(cause, RuntimeError)
            assert "connection reset" in str(cause)
        finally:
            exchange.create_order.retry.stop = original_stop

    @pytest.mark.asyncio
    async def test_does_not_look_up_existing_order_without_a_client_order_id(self, exchange):
        """No client_order_id means there's nothing to look up by -- must
        just re-raise, not call get_order_by_client_id at all."""
        exchange.trading_client = MagicMock()
        exchange.trading_client.submit_order = MagicMock(side_effect=RuntimeError("connection reset"))
        exchange.trading_client.get_order_by_client_id = MagicMock()

        original_stop = exchange.create_order.retry.stop
        try:
            exchange.create_order.retry.stop = stop_after_attempt(1)
            with pytest.raises(RetryError) as exc_info:
                await exchange.create_order("BTC/USD", 1.0, "buy", confirm=False)
            cause = exc_info.value.last_attempt.exception()
            assert isinstance(cause, RuntimeError)
            assert "connection reset" in str(cause)
        finally:
            exchange.create_order.retry.stop = original_stop

        exchange.trading_client.get_order_by_client_id.assert_not_called()