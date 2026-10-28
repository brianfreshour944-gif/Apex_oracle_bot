"""Fail-closed paper-trading guard: safety-boundary tests.

These prove the boundary from both sides:

* a LIVE (or malformed/unrecognised) endpoint is REJECTED, and
* the rejection happens BEFORE any order-submission call can occur.

The "before" claim is asserted by counting calls on a spy client: if the guard
let a live endpoint through, the spy's ``submit_order`` would record a call. It
must record zero.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.exchange import AlpacaExchange
from src.paper_guard import (
    LIVE_HOST,
    PAPER_BASE_URL,
    PAPER_HOST,
    PaperTradingViolation,
    _safe_endpoint_label,
    assert_paper_endpoint,
    classify_endpoint,
    looks_like_paper_account,
    validate_configured_endpoint,
    verify_account_for_order_placement,
)
from tests.helpers import paper_trading_client


class TestClassifyEndpoint:
    """Normalization: trailing slashes, case, scheme, and path must not matter."""

    @pytest.mark.parametrize(
        "raw",
        [
            PAPER_BASE_URL,
            PAPER_BASE_URL + "/",
            PAPER_HOST,
            f"  {PAPER_BASE_URL.upper()}  ",
            f"{PAPER_BASE_URL}/v2/account",
            f"http://{PAPER_HOST}",
        ],
    )
    def test_paper_variants_classified_paper(self, raw):
        assert classify_endpoint(raw) == "paper"

    @pytest.mark.parametrize(
        "raw",
        [
            f"https://{LIVE_HOST}",
            f"https://{LIVE_HOST}/",
            LIVE_HOST,
            f"https://{LIVE_HOST}/v2/account",
        ],
    )
    def test_live_variants_classified_live(self, raw):
        assert classify_endpoint(raw) == "live"

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "",
            "   ",
            "not a url",
            f"https://{PAPER_HOST}.evil.com",
            f"https://evil.com/{PAPER_HOST}",
            "https://paper-api.example.com",
        ],
    )
    def test_missing_or_unexpected_is_invalid_never_paper(self, raw):
        # Fail closed: anything we do not positively recognise is NOT paper.
        assert classify_endpoint(raw) == "invalid"

    def test_enum_like_object_uses_value(self):
        # alpaca-py stores the resolved host as a BaseURL enum with .value.
        class FakeEnum:
            value = PAPER_BASE_URL

        assert classify_endpoint(FakeEnum()) == "paper"

    def test_mock_endpoint_is_not_paper(self):
        # A bare mock is unrecognised -> invalid, so the guard refuses it.
        assert classify_endpoint(MagicMock()) == "invalid"


class TestAssertPaperEndpoint:
    def test_paper_passes(self):
        assert_paper_endpoint(PAPER_BASE_URL)

    def test_live_raises(self):
        with pytest.raises(PaperTradingViolation, match="live"):
            assert_paper_endpoint(f"https://{LIVE_HOST}")

    def test_missing_raises(self):
        with pytest.raises(PaperTradingViolation):
            assert_paper_endpoint(None)

    def test_error_message_names_the_context(self):
        with pytest.raises(PaperTradingViolation, match="order submission"):
            assert_paper_endpoint(f"https://{LIVE_HOST}", context="order submission")


class TestValidateConfiguredEndpoint:
    def test_paper_accepted(self):
        assert validate_configured_endpoint(PAPER_BASE_URL) == PAPER_BASE_URL

    def test_live_rejected_with_valueerror(self):
        # Startup validation must fail loudly, not warn.
        with pytest.raises(ValueError):
            validate_configured_endpoint(f"https://{LIVE_HOST}")

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            validate_configured_endpoint("")


class TestConfigRejectsLiveAtStartup:
    """Layer 1: a live ALPACA_BASE_URL must fail during settings validation."""

    def test_settings_validation_rejects_live(self):
        from src.config import TradingBotSettings

        with pytest.raises(ValueError):
            TradingBotSettings(ALPACA_BASE_URL=f"https://{LIVE_HOST}")

    def test_settings_validation_accepts_paper(self):
        from src.config import TradingBotSettings

        s = TradingBotSettings(ALPACA_BASE_URL=PAPER_BASE_URL)
        assert s.ALPACA_BASE_URL == PAPER_BASE_URL


class TestClientConstructionRefusesLive:
    """Layer 2 (configured): constructing the client on a live URL must fail."""

    def test_construction_rejects_live_configured_url(self, monkeypatch):
        from src import config

        monkeypatch.setattr(config.settings, "ALPACA_BASE_URL", f"https://{LIVE_HOST}")
        with pytest.raises(PaperTradingViolation):
            AlpacaExchange()


class TestLiveEndpointCannotReachSubmitOrder:
    """The core safety claim: a live endpoint never reaches submit_order."""

    @pytest.mark.asyncio
    async def test_create_order_live_client_refuses_before_submit(self):
        spy = MagicMock()
        spy.submit_order = MagicMock()
        spy._base_url = f"https://{LIVE_HOST}"

        exchange = AlpacaExchange.__new__(AlpacaExchange)
        exchange.trading_client = spy
        exchange.circuit_breaker = MagicMock()

        with pytest.raises(PaperTradingViolation):
            await exchange.create_order("BTC/USD", 1.0, "buy", confirm=False)

        assert spy.submit_order.call_count == 0, "submit_order was reached on a live endpoint"

    @pytest.mark.asyncio
    async def test_protective_stop_live_client_refuses_before_submit(self):
        spy = MagicMock()
        spy.submit_order = MagicMock()
        spy._base_url = f"https://{LIVE_HOST}"

        exchange = AlpacaExchange.__new__(AlpacaExchange)
        exchange.trading_client = spy
        exchange.circuit_breaker = MagicMock()

        with pytest.raises(PaperTradingViolation):
            await exchange.submit_protective_stop("BTC/USD", 1.0, 100.0, 99.0)

        assert spy.submit_order.call_count == 0

    @pytest.mark.asyncio
    async def test_stop_rearm_live_client_refuses_before_submit(self):
        spy = MagicMock()
        spy.submit_order = MagicMock()
        spy._base_url = f"https://{LIVE_HOST}"

        exchange = AlpacaExchange.__new__(AlpacaExchange)
        exchange.trading_client = spy
        exchange.circuit_breaker = MagicMock()

        with pytest.raises(PaperTradingViolation):
            await exchange._submit_protective_stop_once("BTC/USD", 1.0, 100.0, 99.0, None)

        assert spy.submit_order.call_count == 0

    @pytest.mark.asyncio
    async def test_bypass_circuit_breaker_does_not_bypass_paper_guard(self):
        """An exit that bypasses the circuit breaker must still be paper-only."""
        spy = MagicMock()
        spy.submit_order = MagicMock()
        spy._base_url = f"https://{LIVE_HOST}"

        exchange = AlpacaExchange.__new__(AlpacaExchange)
        exchange.trading_client = spy
        exchange.circuit_breaker = MagicMock()

        with pytest.raises(PaperTradingViolation):
            await exchange.create_order(
                "BTC/USD", 1.0, "sell", confirm=False, bypass_circuit_breaker=True
            )

        assert spy.submit_order.call_count == 0

    @pytest.mark.asyncio
    async def test_unrecognised_endpoint_refused(self):
        spy = MagicMock()
        spy.submit_order = MagicMock()
        spy._base_url = "https://paper-api.example.com"  # looks plausible, is not paper

        exchange = AlpacaExchange.__new__(AlpacaExchange)
        exchange.trading_client = spy
        exchange.circuit_breaker = MagicMock()

        with pytest.raises(PaperTradingViolation):
            await exchange.create_order("BTC/USD", 1.0, "buy", confirm=False)

        assert spy.submit_order.call_count == 0

    @pytest.mark.asyncio
    async def test_configured_url_disagreement_refuses(self, monkeypatch):
        """Client resolved paper, but the CONFIGURED url is live -> refuse.

        A disagreement between what the client would hit and what the settings
        say is exactly the confusing state the guard must not allow.
        """
        from src import config

        spy = MagicMock()
        spy.submit_order = MagicMock()
        spy._base_url = PAPER_BASE_URL

        exchange = AlpacaExchange.__new__(AlpacaExchange)
        exchange.trading_client = spy
        exchange.circuit_breaker = MagicMock()

        monkeypatch.setattr(config.settings, "ALPACA_BASE_URL", f"https://{LIVE_HOST}")
        with pytest.raises(PaperTradingViolation):
            await exchange.create_order("BTC/USD", 1.0, "buy", confirm=False)

        assert spy.submit_order.call_count == 0


class TestPaperEndpointStillWorks:
    """The guard must not break the normal paper path."""

    @pytest.mark.asyncio
    async def test_paper_client_reaches_submit(self):
        spy = paper_trading_client()
        order = MagicMock()
        order.id = "ord_1"
        order.symbol = "BTC/USD"
        order.qty = "1.0"
        order.status = "accepted"
        spy.submit_order = MagicMock(return_value=order)

        exchange = AlpacaExchange.__new__(AlpacaExchange)
        exchange.trading_client = spy
        exchange.circuit_breaker = MagicMock()
        exchange.circuit_breaker.call = _passthrough_call

        result = await exchange.create_order("BTC/USD", 1.0, "buy", confirm=False)
        assert spy.submit_order.call_count == 1
        assert result["id"] == "ord_1"


async def _passthrough_call(fn, *args, **kwargs):
    """Stand-in for CircuitBreaker.call that invokes and awaits ``fn``."""
    import inspect

    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


class TestAccountHeuristic:
    """Endpoint validation is not account verification; this is the weak check."""

    def test_paper_prefix_detected(self):
        assert looks_like_paper_account("PA123456") is True

    def test_non_paper_number(self):
        assert looks_like_paper_account("123456789") is False

    def test_missing_is_unknown_not_paper(self):
        # None must NOT be treated as paper.
        assert looks_like_paper_account(None) is None
        assert looks_like_paper_account("") is None


class _Account:
    """Concrete account stand-in (NOT a Mock: a Mock's attributes are all truthy,
    which would make every flag look blocked and the status look unparseable)."""

    def __init__(
        self,
        number: str | None = "PA123456",
        status: str | None = "ACTIVE",
        account_blocked: bool = False,
        trading_blocked: bool = False,
        trade_suspended_by_user: bool = False,
    ):
        self.account_number = number
        self.status = status
        self.account_blocked = account_blocked
        self.trading_blocked = trading_blocked
        self.trade_suspended_by_user = trade_suspended_by_user


def _load_with_account(monkeypatch, account):
    """Run AlpacaExchange.load() against a fake SDK client returning ``account``."""
    from src import config

    monkeypatch.setattr(config.settings, "ALPACA_API_KEY", "k")
    monkeypatch.setattr(config.settings, "ALPACA_SECRET_KEY", "s")
    monkeypatch.setattr(config.settings, "ALPACA_BASE_URL", PAPER_BASE_URL)

    fake_client = MagicMock()
    fake_client._base_url = PAPER_BASE_URL
    fake_client.get_account = MagicMock(return_value=account)

    exchange = AlpacaExchange.__new__(AlpacaExchange)
    exchange.api_key = "k"
    exchange.secret_key = "s"
    exchange.paper = True
    exchange.trading_client = None
    exchange.data_client = None
    return exchange, fake_client


class TestAccountVerification:
    """verify_account_for_order_placement is fail-closed and separate from the
    endpoint check."""

    def test_good_paper_account_verified(self):
        ok, reason = verify_account_for_order_placement(_Account())
        assert ok is True
        assert "paper account corroborated" in reason

    def test_missing_account_refused(self):
        ok, _ = verify_account_for_order_placement(None)
        assert ok is False

    def test_missing_account_number_refused(self):
        # The previous behaviour warned here; now it must refuse (not verified).
        ok, reason = verify_account_for_order_placement(_Account(number=None))
        assert ok is False
        assert "account_number missing" in reason

    def test_empty_account_number_refused(self):
        ok, _ = verify_account_for_order_placement(_Account(number="  "))
        assert ok is False

    def test_non_paper_number_refused(self):
        ok, reason = verify_account_for_order_placement(_Account(number="123456789"))
        assert ok is False
        assert "not a paper" in reason

    @pytest.mark.parametrize(
        "flag",
        ["account_blocked", "trading_blocked", "trade_suspended_by_user"],
    )
    def test_blocked_flags_refuse(self, flag):
        ok, reason = verify_account_for_order_placement(_Account(**{flag: True}))
        assert ok is False
        assert flag in reason

    @pytest.mark.parametrize(
        "status",
        ["ACCOUNT_CLOSED", "DISABLED", "REJECTED", "SUBMISSION_FAILED", "INACTIVE"],
    )
    def test_dead_status_refuses(self, status):
        ok, reason = verify_account_for_order_placement(_Account(status=status))
        assert ok is False
        assert status in reason

    def test_active_status_passes(self):
        ok, _ = verify_account_for_order_placement(_Account(status="ACTIVE"))
        assert ok is True

    def test_unfamiliar_status_does_not_bypass_number_check(self):
        # An odd-but-not-dead status must not let a non-paper number through.
        ok, _ = verify_account_for_order_placement(
            _Account(status="SOME_NEW_STATUS", number="123456789")
        )
        assert ok is False

    def test_enum_like_status_uses_value(self):
        class _Status:
            value = "ACTIVE"

        ok, _ = verify_account_for_order_placement(_Account(status=_Status()))
        assert ok is True


class TestStartupAccountCheck:
    """Layer 4: load() must refuse when the account cannot be verified."""

    @pytest.mark.asyncio
    async def test_non_paper_account_refused_at_load(self, monkeypatch):
        exchange, client = _load_with_account(monkeypatch, _Account(number="123456789"))
        with patch("src.exchange.TradingClient", return_value=client):
            with pytest.raises(RuntimeError, match="not verified for order placement"):
                await exchange.load()
        assert exchange.trading_client is None

    @pytest.mark.asyncio
    async def test_missing_account_number_refused_at_load(self, monkeypatch):
        exchange, client = _load_with_account(monkeypatch, _Account(number=None))
        with patch("src.exchange.TradingClient", return_value=client):
            with pytest.raises(RuntimeError, match="not verified for order placement"):
                await exchange.load()
        assert exchange.trading_client is None

    @pytest.mark.asyncio
    async def test_blocked_account_refused_at_load(self, monkeypatch):
        exchange, client = _load_with_account(monkeypatch, _Account(account_blocked=True))
        with patch("src.exchange.TradingClient", return_value=client):
            with pytest.raises(RuntimeError):
                await exchange.load()
        assert exchange.trading_client is None

    @pytest.mark.asyncio
    async def test_paper_account_passes_at_load(self, monkeypatch):
        exchange, client = _load_with_account(monkeypatch, _Account())
        with patch("src.exchange.TradingClient", return_value=client):
            await exchange.load()
        assert exchange.trading_client is client

    @pytest.mark.asyncio
    async def test_live_resolved_client_refused_at_load(self, monkeypatch):
        """If the SDK ever resolved a live host, load() must refuse it."""
        from src import config

        monkeypatch.setattr(config.settings, "ALPACA_API_KEY", "k")
        monkeypatch.setattr(config.settings, "ALPACA_SECRET_KEY", "s")
        monkeypatch.setattr(config.settings, "ALPACA_BASE_URL", PAPER_BASE_URL)

        fake_client = MagicMock()
        fake_client._base_url = f"https://{LIVE_HOST}"

        exchange = AlpacaExchange.__new__(AlpacaExchange)
        exchange.api_key = "k"
        exchange.secret_key = "s"
        exchange.paper = True
        exchange.trading_client = None
        exchange.data_client = None

        with patch("src.exchange.TradingClient", return_value=fake_client):
            with pytest.raises(PaperTradingViolation):
                await exchange.load()
        assert exchange.trading_client is None


class TestStartupErrorsDoNotLeakSecrets:
    """Startup errors must name the failed check without logging credentials,
    secrets, or sensitive account information."""

    def test_endpoint_label_redacts_embedded_credentials(self):
        label = _safe_endpoint_label("https://MYKEY:MYSECRET@api.alpaca.markets")
        assert "MYSECRET" not in label
        assert "MYKEY" not in label
        assert "api.alpaca.markets" in label
        assert "REDACTED" in label

    def test_paper_violation_message_has_no_credentials(self):
        with pytest.raises(PaperTradingViolation) as exc:
            assert_paper_endpoint("https://MYKEY:MYSECRET@api.alpaca.markets")
        text = str(exc.value)
        assert "MYSECRET" not in text and "MYKEY" not in text
        assert "order submission" in text  # still names the failed check

    def test_config_error_has_no_credentials(self):
        with pytest.raises(ValueError) as exc:
            validate_configured_endpoint("https://MYKEY:MYSECRET@api.alpaca.markets")
        text = str(exc.value)
        assert "MYSECRET" not in text and "MYKEY" not in text
        assert "ALPACA_BASE_URL" in text

    def test_account_number_masked_in_reason(self):
        ok, reason = verify_account_for_order_placement(
            _Account(number="123456789012345")
        )
        assert ok is False
        assert "123456789012345" not in reason
        assert "***" in reason

    def test_settings_validation_error_hides_input_value(self):
        """pydantic must not echo the raw input (it could embed credentials)."""
        from src.config import TradingBotSettings

        with pytest.raises(ValueError) as exc:
            TradingBotSettings(ALPACA_BASE_URL="https://MYKEY:MYSECRET@api.alpaca.markets")
        text = str(exc.value)
        assert "MYSECRET" not in text
        assert "input_value" not in text

    def test_auth_error_body_does_not_include_headers(self):
        """str(APIError) carries only the response body, never request headers,
        so the auth-failure path cannot leak the key/secret."""
        from alpaca.common.exceptions import APIError

        err = APIError('{"message":"unauthorized"}', 401)
        assert "unauthorized" in str(err)
        # The credentials live in request headers, which are not stringified.
        assert "APCA" not in str(err)


class TestStartupTreatsGuardFailureAsFatal:
    """A guard/account-verification failure is a configuration fault: the bot must
    fail fast at startup rather than silently retry in offline mode."""

    def _startup_catch(self, exc):
        """Reproduce the startup try/except decision for ``exc``.

        Returns "fatal" if it re-raises, "offline" if it swallows into offline mode.
        Mirrors src/bot.py's exchange-init handler.
        """
        msg = str(exc).lower()
        if "authentication failed" in msg or "unauthorized" in msg:
            return "fatal"
        if isinstance(exc, PaperTradingViolation) or "not verified for order placement" in msg:
            return "fatal"
        return "offline"

    def test_paper_violation_is_fatal(self):
        assert self._startup_catch(PaperTradingViolation("live endpoint")) == "fatal"

    def test_account_verification_failure_is_fatal(self):
        exc = RuntimeError(
            "Authenticated Alpaca account is not verified for order placement: "
            "account_number missing or empty; ... Refusing to enable order placement."
        )
        assert self._startup_catch(exc) == "fatal"

    def test_transient_network_error_stays_offline(self):
        assert self._startup_catch(RuntimeError("connection reset by peer")) == "offline"
