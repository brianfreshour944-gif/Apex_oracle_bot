"""Modern Alpaca exchange client using alpaca-py with rate limiting and circuit breakers."""

from __future__ import annotations

import asyncio
import datetime
import time
from typing import Any, Protocol, runtime_checkable

import polars as pl
from alpaca.common.exceptions import APIError
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestBarRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Import alpaca-py components
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, LimitOrderRequest
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.circuit_breaker import CircuitBreaker
from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


def _is_rate_limit_error(exception: Exception) -> bool:
    """Return True if the exception is an Alpaca 429 rate-limit error."""
    if isinstance(exception, APIError):
        return getattr(exception, "status_code", 0) == 429
    return False


def _retry_on_rate_limit(exception: Exception) -> bool:
    """Retry predicate: retry on rate-limit errors with longer backoff."""
    return _is_rate_limit_error(exception)


def _is_circuit_open_error(exception: Exception) -> bool:
    """True if `exception` is the RuntimeError CircuitBreaker.call() raises
    when the circuit is OPEN (message: "Circuit '<name>' is OPEN - calls
    blocked")."""
    return isinstance(exception, RuntimeError) and "is OPEN" in str(exception)


# alpaca-py's GetOrdersRequest.status is a QueryOrderStatus enum that only
# accepts "open", "closed", or "all" -- NOT Alpaca's own order-status values
# ("new", "filled", "cancelled", "expired", etc., which describe a single
# order's lifecycle, not a query filter). Every call site in this codebase
# was written against the order-status vocabulary, so every get_orders(
# status=...) call with "new"/"filled"/etc. raised a pydantic ValidationError
# at request-construction time -- before any network call -- which the
# @retry decorator (not filtering out ValidationError) retried up to 5 times
# over ~1-2 minutes before giving up, silently swallowed by every caller's
# broad try/except. This made stale-order reconciliation, the unresolved-
# order entry veto, and create_order's fill-confirmation fallback complete
# no-ops. Found via an external correctness audit, reproduced directly
# against the installed alpaca-py==0.44.0, 2026-09-22.
_ORDER_STATUS_TO_QUERY_STATUS = {
    "new": "open",
    "accepted": "open",
    "pending_new": "open",
    "accepted_for_bidding": "open",
    "partially_filled": "open",
    "open": "open",
    "filled": "closed",
    "cancelled": "closed",
    "canceled": "closed",
    "expired": "closed",
    "rejected": "closed",
    "done_for_day": "closed",
    "closed": "closed",
    "all": "all",
}


# Sentinel placed in AlpacaExchange._order_cache while a create_order() call
# is between its idempotency check and its real cache write (i.e. the order
# is being submitted/confirmed). Closes the TOCTOU window a concurrent call
# with the same client_order_id could otherwise slip through: the check and
# the write are on opposite sides of several `await`s (submit_order,
# get_order polling), so without a claim in between, two overlapping calls
# for the same id could both pass the "not cached yet" check and both
# submit. Found via an external concurrency audit, 2026-09-22.
_ORDER_PENDING = object()


def _retry_unless_circuit_open(exception: Exception) -> bool:
    """Retry predicate for methods that previously retried on ANY exception
    (get_positions, get_order, get_orders, create_order) now that their
    calls are routed through self.circuit_breaker.

    A circuit-open signal means the breaker has already decided this
    dependency is down -- retrying it just re-blocks for another 2-30s of
    exponential backoff before failing anyway, which defeats the point of
    having a circuit breaker (fail fast instead of piling onto a known-bad
    dependency). Every other exception still retries exactly as before.

    get_account/get_bars/get_latest_bar don't need this: they already use
    _retry_on_rate_limit, which only returns True for a 429 APIError, so a
    circuit-open RuntimeError there already stops retrying immediately.

    Also stops retrying a permanent (non-429) 4xx APIError -- e.g. invalid
    quantity/precision, insufficient buying power, invalid symbol. These are
    deterministic rejections that will fail identically on every retry, so
    retrying them up to stop_after_attempt(5) times only burns 2-30s of
    backoff per attempt for nothing AND -- since every attempt still counts
    as a failure against the shared 'alpaca_exchange' circuit -- a single
    bad order can single-handedly trip the circuit and block every OTHER
    symbol's bars/positions/risk calls too. Found 2026-09-21 diagnosing a
    live incident where 3 symbols' buy orders failed (root cause hidden --
    see the submit_err logging fix in create_order) and, via this retry
    storm, tripped the shared circuit for the whole bot.
    """
    if _is_circuit_open_error(exception):
        return False
    if isinstance(exception, APIError):
        status = getattr(exception, "status_code", None)
        if status is not None and 400 <= status < 500 and status != 429:
            return False
    return True


def _rate_limit_wait(retry_state: RetryCallState) -> float:
    """Wait strategy that respects the Retry-After header when available,
    otherwise falls back to exponential backoff.

    Alpaca APIError exposes ``_response`` (the raw HTTP response) which may
    carry a ``Retry-After`` or ``RateLimit-Reset`` header."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is not None:
        resp = getattr(exc, "_response", None)
        if resp is not None:
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return float(retry_after)
                except (ValueError, TypeError):
                    pass
            reset_ts = resp.headers.get("RateLimit-Reset")
            if reset_ts is not None:
                try:
                    import time as _time
                    offset = float(reset_ts) - _time.time()
                    if offset > 0:
                        return min(offset, 60.0)  # cap at 60 s
                except (ValueError, TypeError):
                    pass
    # Fallback: exponential backoff
    return wait_exponential(multiplier=1, min=4, max=30)(retry_state)


class RateLimitError(Exception):
    """Raised when Alpaca rate limit is exceeded and all retries are exhausted."""
    pass


@runtime_checkable
class BaseExchange(Protocol):
    """Abstract Base Exchange Protocol for multi-exchange adapters."""
    async def load(self) -> None: ...
    async def close(self) -> None: ...
    async def get_account(self) -> dict[str, Any]: ...
    async def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100, end: datetime.datetime | None = None) -> pl.DataFrame: ...
    async def get_positions(self, bypass_circuit_breaker: bool = False) -> list[dict[str, Any]]: ...
    async def create_order(self, symbol: str, qty: float, side: str, type: str = "market", time_in_force: str = "ioc", client_order_id: str | None = None, bypass_circuit_breaker: bool = False) -> dict[str, Any]: ...
    def invalidate_bars_cache(self, symbol: str | None = None, timeframe: str | None = None) -> int: ...


class AlpacaExchange:
    """Modern Alpaca exchange client using alpaca-py."""

    def __init__(self):
        self.api_key = settings.ALPACA_API_KEY
        self.secret_key = settings.ALPACA_SECRET_KEY
        base_url = settings.ALPACA_BASE_URL or "https://paper-api.alpaca.markets"
        self.paper = "paper" in base_url.lower()
        
        self.trading_client: TradingClient | None = None
        self.data_client: CryptoHistoricalDataClient | None = None
        
        self.circuit_breaker = CircuitBreaker("alpaca_exchange")
        # Internal lock for mimicking rate limiter if needed, though SDK handles some
        self._lock = asyncio.Lock()
        # Rate-limit tracking
        self._rate_limit_remaining = 200
        self._rate_limit_reset = 0.0
        self._rate_limit_hits = 0
        # Bar cache: (symbol, timeframe, limit) -> (cached_at, latest_bar_timestamp, DataFrame)
        # latest_bar_timestamp is the timestamp of the most recent bar in the cached data
        self._bars_cache: dict[tuple, tuple[float, str | None, pl.DataFrame]] = {}
        self._bars_cache_ttl: float = 15.0  # cache bars for 15 seconds (was 60s - reduced to prevent stale signals)
        # Idempotency cache: client_order_id -> (timestamp, order_info)
        self._order_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._order_cache_ttl: float = 300.0  # cache orders for 5 minutes
        # Duplicate-submission detector: (symbol, side) -> list of recent
        # submission timestamps. Distinct from the idempotency cache above --
        # this catches two DIFFERENT client_order_ids for the same
        # symbol+side arriving within a short window (e.g. a race between
        # concurrent scan cycles), which the idempotency cache can't detect
        # since it keys on an already-unique id. Alert-only, does not block
        # the order (found as a missing alert in ADVERSARIAL_AUDIT_2026-09-20.md).
        self._recent_order_submissions: dict[tuple[str, str], list[float]] = {}
        self._duplicate_order_window_sec: float = 10.0
        # Background task tracking for fire-and-forget alerts
        self._background_tasks: set[asyncio.Task] = set()

    async def load(self) -> None:
        """Initialize the exchange client and verify credentials."""
        if not (self.api_key and self.secret_key):
            raise RuntimeError(
                "Alpaca credentials missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "(e.g. in Coolify environment variables)."
            )
            
        try:
            # Initialize Alpaca-py clients (these are synchronous under the hood, but we wrap calls in threads/async if needed)
            self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
            self.data_client = CryptoHistoricalDataClient(self.api_key, self.secret_key)
            
            # Verify credentials by fetching account
            account = await asyncio.to_thread(self.trading_client.get_account)
            if account.account_blocked:
                raise RuntimeError("Alpaca account is blocked.")
        except APIError as e:
            # Auth/config errors (401, 403, etc.) - these are fatal configuration issues
            self.trading_client = None
            self.data_client = None
            if getattr(e, "status_code", None) in (401, 403):
                raise RuntimeError(f"Alpaca authentication failed: {e}") from e
            raise  # Re-raise other API errors as-is for retry logic
        except Exception:
            # Network/transient errors - let caller decide retry/offline mode
            self.trading_client = None
            self.data_client = None
            raise
            
        logger.info(f"Alpaca client initialized (Paper={self.paper})")

    async def close(self) -> None:
        """Close the exchange client."""
        self.trading_client = None
        self.data_client = None
        logger.info("Alpaca client closed")

    @retry(retry=retry_if_exception(_retry_on_rate_limit), stop=stop_after_attempt(5), wait=_rate_limit_wait)
    async def get_account(self) -> dict[str, Any]:
        """Get account information (formatted to dict for compatibility)."""
        if not self.trading_client:
            await self.load()

        account = await self.circuit_breaker.call(asyncio.to_thread, self.trading_client.get_account)
        # Return a dictionary mimicking old JSON response
        return {
            "id": str(account.id),
            "status": str(account.status),
            "currency": str(account.currency),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "portfolio_value": float(account.portfolio_value),
        }
        
    def _parse_timeframe(self, timeframe_str: str) -> TimeFrame:
        """Parse 1Min, 5Min, 1Hour, 1Day into alpaca-py TimeFrame."""
        import re
        match = re.match(r'(\d+)\s*([a-zA-Z]+)?', timeframe_str)
        if match:
            val = int(match.group(1))
            unit = match.group(2).lower() if match.group(2) else ""
            if "min" in unit or unit == "m":
                return TimeFrame(val, TimeFrameUnit.Minute)
            elif "hour" in unit or unit == "h":
                return TimeFrame(val, TimeFrameUnit.Hour)
            elif "day" in unit or unit == "d":
                return TimeFrame(val, TimeFrameUnit.Day)
        return TimeFrame.Day

    @retry(retry=retry_if_exception(_retry_on_rate_limit), stop=stop_after_attempt(5), wait=_rate_limit_wait)
    async def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100, end: datetime.datetime | None = None) -> pl.DataFrame:
        """Get crypto market data using alpaca-py and convert to Polars.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            timeframe: Bar timeframe (e.g., "1Min", "1Hour", "1Day")
            limit: Maximum number of bars to return
            end: End timestamp for strict point-in-time query. If provided, bars 
                 will only include data up to this timestamp. If None, uses current time.
                 Use this to prevent look-ahead bias in backtesting or when you need
                 bars as of a specific moment.
        """
        if not self.data_client:
            await self.load()

        cache_key = (symbol, timeframe, limit)
        now = time.time()
        cached = self._bars_cache.get(cache_key)
        if cached is not None:
            cached_ts, cached_latest_bar_ts, cached_df = cached
            # Check TTL
            if now - cached_ts < self._bars_cache_ttl and len(cached_df) >= limit:
                # Additional check: if we have a latest bar timestamp, verify it hasn't been superseded
                # This prevents using stale cached data when a new bar has closed
                if cached_latest_bar_ts is not None and end is not None:
                    try:
                        cached_bar_dt = datetime.datetime.fromisoformat(cached_latest_bar_ts.replace('Z', '+00:00'))
                        end_dt = end if isinstance(end, datetime.datetime) else datetime.datetime.fromisoformat(end.replace('Z', '+00:00'))
                        # If the requested end time is after our cached latest bar, we need fresh data
                        if end_dt > cached_bar_dt:
                            pass  # Fall through to fetch fresh data
                        else:
                            return cached_df
                    except Exception:
                        pass  # Fall through on parse error
                else:
                    return cached_df

        tf = self._parse_timeframe(timeframe)
        
        # Use provided end time or current time for strict PIT
        end_time = end if end is not None else datetime.datetime.now(datetime.UTC)
        
        # Calculate start time heuristically based on limit
        if tf.unit == TimeFrameUnit.Minute:
            delta = datetime.timedelta(minutes=tf.amount * limit * 1.5)
        elif tf.unit == TimeFrameUnit.Hour:
            delta = datetime.timedelta(hours=tf.amount * limit * 1.5)
        else:
            delta = datetime.timedelta(days=tf.amount * limit * 1.5)
            
        start_time = end_time - max(delta, datetime.timedelta(hours=1))
        
        request_params = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start_time,
            end=end_time,
            limit=limit
        )

        try:
            bars_df = await self.circuit_breaker.call(asyncio.to_thread, self.data_client.get_crypto_bars, request_params)
            if bars_df.data and symbol in bars_df.data:
                # Get the list of Bar objects
                bars = bars_df.data[symbol]
                
                # Convert to dict format expected by downstream
                data_list = []
                latest_bar_ts = None
                for b in bars:
                    ts = b.timestamp.isoformat() if hasattr(b.timestamp, "isoformat") else b.timestamp
                    data_list.append({
                        "timestamp": ts,
                        "open": float(b.open),
                        "high": float(b.high),
                        "low": float(b.low),
                        "close": float(b.close),
                        "volume": float(b.volume),
                        "vwap": float(b.vwap),
                        "trade_count": int(b.trade_count),
                    })
                    latest_bar_ts = ts  # Keep the last (most recent) timestamp
                
                df_result = pl.DataFrame(data_list)
                # Cache with latest bar timestamp for invalidation
                self._bars_cache[cache_key] = (time.time(), latest_bar_ts, df_result)
                return df_result
        except Exception as e:
            logger.warning(f"Failed to fetch bars for {symbol}: {e}")

        result = pl.DataFrame()
        self._bars_cache[cache_key] = (time.time(), None, result)
        return result

    @retry(retry=retry_if_exception(_retry_on_rate_limit), stop=stop_after_attempt(5), wait=_rate_limit_wait)
    async def get_latest_bar(self, symbol: str) -> pl.DataFrame:
        """Get the latest bar for a crypto symbol."""
        if not self.data_client:
            await self.load()

        request_params = CryptoLatestBarRequest(symbol_or_symbols=symbol)
        try:
            latest_bars = await asyncio.to_thread(self.data_client.get_crypto_latest_bar, request_params)
            if symbol in latest_bars:
                b = latest_bars[symbol]
                data = {
                    "timestamp": b.timestamp.isoformat() if hasattr(b.timestamp, "isoformat") else b.timestamp,
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                    "vwap": float(b.vwap),
                    "trade_count": int(b.trade_count),
                }
                return pl.DataFrame([data])
        except Exception as e:
            logger.warning(f"Failed to fetch latest bar for {symbol}: {e}")

        return pl.DataFrame()

    def invalidate_bars_cache(self, symbol: str | None = None, timeframe: str | None = None) -> int:
        """
        Invalidate cached bars data.
        
        Args:
            symbol: If provided, only invalidate cache for this symbol. If None, invalidate all symbols.
            timeframe: If provided, only invalidate cache for this timeframe. If None, invalidate all timeframes.
            
        Returns:
            Number of cache entries invalidated.
        """
        if symbol is None and timeframe is None:
            # Clear all
            count = len(self._bars_cache)
            self._bars_cache.clear()
            logger.info(f"Invalidated all bars cache ({count} entries)")
            return count
        
        count = 0
        keys_to_delete = []
        for key in self._bars_cache:
            key_symbol, key_timeframe, key_limit = key
            if symbol is not None and key_symbol != symbol:
                continue
            if timeframe is not None and key_timeframe != timeframe:
                continue
            keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self._bars_cache[key]
            count += 1
        
        if count > 0:
            logger.info(f"Invalidated {count} bars cache entries for symbol={symbol}, timeframe={timeframe}")
        
        return count

    @retry(retry=retry_if_exception(_retry_unless_circuit_open), stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def get_positions(self, bypass_circuit_breaker: bool = False) -> list[dict[str, Any]]:
        """Get open positions.

        ``bypass_circuit_breaker``: when True, fetches positions directly
        via the trading client instead of through the circuit breaker.
        Used by emergency/killswitch liquidation paths that must read
        current exposure even during an exchange outage.
        """
        if not self.trading_client:
            await self.load()

        if bypass_circuit_breaker:
            positions = await asyncio.to_thread(self.trading_client.get_all_positions)
        else:
            positions = await self.circuit_breaker.call(asyncio.to_thread, self.trading_client.get_all_positions)
        result = []
        for p in positions:
            result.append({
                "symbol": str(p.symbol),
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            })
        return result

    @retry(retry=retry_if_exception(_retry_unless_circuit_open), stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Fetch order details by order ID."""
        if not self.trading_client:
            await self.load()

        order = await self.circuit_breaker.call(asyncio.to_thread, self.trading_client.get_order_by_id, order_id)
        return {
            "id": str(order.id),
            "symbol": str(order.symbol),
            "qty": float(order.qty) if order.qty else 0.0,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0.0,
            "status": str(order.status.value) if hasattr(order.status, "value") else str(order.status),
            "side": str(order.side.value) if hasattr(order.side, "value") else str(order.side),
            "type": str(order.type.value) if hasattr(order.type, "value") else str(order.type),
        }

    @retry(retry=retry_if_exception(_retry_unless_circuit_open), stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def get_orders(
        self,
        status: str | None = None,
        after: str | None = None,
        until: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch order history from Alpaca with optional filters.

        Args:
            status: Filter by order status (e.g., "filled", "cancelled", "expired",
                or Alpaca's own "open"/"closed"/"all" query filter directly --
                both vocabularies work, see _ORDER_STATUS_TO_QUERY_STATUS)
            after: ISO format datetime string - only orders after this time
            until: ISO format datetime string - only orders before this time
            limit: Maximum number of orders to return

        Returns:
            List of order dictionaries with fill data
        """
        if not self.trading_client:
            await self.load()

        # Parse after/until strings to datetime if provided
        after_dt = None
        until_dt = None
        if after:
            after_dt = datetime.datetime.fromisoformat(after.replace("Z", "+00:00"))
        if until:
            until_dt = datetime.datetime.fromisoformat(until.replace("Z", "+00:00"))

        # Translate to the query-filter vocabulary GetOrdersRequest actually
        # accepts -- see _ORDER_STATUS_TO_QUERY_STATUS above.
        query_status = _ORDER_STATUS_TO_QUERY_STATUS.get(status.lower(), status) if status else status

        request = GetOrdersRequest(
            status=query_status,
            after=after_dt,
            until=until_dt,
            limit=limit,
            direction="desc",
        )
        
        orders = await self.circuit_breaker.call(asyncio.to_thread, self.trading_client.get_orders, request)
        
        results = []
        for order in orders:
            results.append({
                "id": str(order.id),
                "symbol": str(order.symbol).replace("/", ""),
                "qty": float(order.qty) if order.qty else 0.0,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0.0,
                "status": str(order.status.value) if hasattr(order.status, "value") else str(order.status),
                "side": str(order.side.value) if hasattr(order.side, "value") else str(order.side),
                "type": str(order.type.value) if hasattr(order.type, "value") else str(order.type),
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                                # Alpaca orders are commission-free. alpaca-py>=0.43 Order model
                "commission": 0.0,  # SDK 0.44.0 Order has no .commission attribute
                "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                "filled_at": order.filled_at.isoformat() if order.filled_at else None,
                "client_order_id": str(order.client_order_id) if order.client_order_id else None,
            })
        return results

    async def _find_existing_order_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        """
        Look up an order by client_order_id on the exchange. Used by
        create_order to detect an already-submitted order before letting a
        retry resubmit it (see the comment there).

        Returns the formatted order dict if found, or None if not found OR
        the lookup itself failed. Both cases are treated the same
        (fail safe, not fail closed): this is only trying to catch the
        common, verifiable "it actually went through" case, not guarantee
        it always will -- returning None just means create_order falls back
        to its original behavior of re-raising and letting @retry resubmit.
        """
        try:
            # Fix #3: 404 "order not found" is expected before order exists - don't count toward circuit breaker
            order = await asyncio.to_thread(self.trading_client.get_order_by_client_id, client_order_id)
        except Exception as e:
            # 404 means order hasn't landed yet, not an exchange outage
            if "404" in str(e) or "not found" in str(e).lower():
                logger.debug(f"Lookup by client_order_id={client_order_id!r}: not found yet (expected)")
            else:
                logger.warning(f"Lookup by client_order_id={client_order_id!r} failed: {e}")
            return None
        return {
            "id": str(order.id),
            "symbol": str(order.symbol),
            "qty": float(order.qty) if order.qty else 0.0,
            "status": str(order.status.value) if hasattr(order.status, "value") else str(order.status),
        }

    @retry(retry=retry_if_exception(_retry_unless_circuit_open), stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def create_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        type: str = "market",
        time_in_force: str = "ioc",
        confirm: bool = True,
        confirm_timeout: float = 10.0,
        client_order_id: str | None = None,
        bypass_circuit_breaker: bool = False,
        limit_price: float | None = None,
        post_only: bool = False,
    ) -> dict[str, Any]:
        """Create a new order using alpaca-py.

        Supports both market and limit orders.

        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            qty: Order quantity
            side: "buy" or "sell"
            type: "market" or "limit"
            time_in_force: "ioc", "gtc", "fok", "day"
            confirm: Whether to wait for fill confirmation
            confirm_timeout: Seconds to wait for confirmation
            client_order_id: Optional idempotency key
            bypass_circuit_breaker: For exit orders during circuit open
            limit_price: Limit price for limit orders (required if type="limit")
            post_only: If True, order is post-only (maker only, no taker)

        Returns:
            Order info dict with id, symbol, qty, status, filled_avg_price, filled_qty, commission
        """
        if not self.trading_client:
            await self.load()

        # --- Idempotency check ---
        if client_order_id is not None:
            now = time.time()
            if len(self._order_cache) > 128:
                self._order_cache = {
                    k: v for k, v in self._order_cache.items()
                    if now - v[0] < self._order_cache_ttl
                }
            cached = self._order_cache.get(client_order_id)
            if cached is not None:
                ts, order_info = cached
                if now - ts < self._order_cache_ttl:
                    if order_info is _ORDER_PENDING:
                        # A concurrent call already claimed this id and is
                        # still submitting/confirming it. Wait briefly for
                        # it to resolve instead of racing a second
                        # submission past this check. Bounded: if the other
                        # call never resolves (e.g. it crashed after
                        # claiming but before writing the real result), fail
                        # open and submit for real rather than block a
                        # legitimate trade for the rest of the TTL -- at that
                        # point Alpaca's own client_order_id uniqueness
                        # constraint (already relied on by the except-path
                        # below) is the remaining safety net.
                        wait_start = time.monotonic()
                        while time.monotonic() - wait_start < min(confirm_timeout, 10.0):
                            await asyncio.sleep(0.5)
                            cached = self._order_cache.get(client_order_id)
                            if cached is None or cached[1] is not _ORDER_PENDING:
                                break
                        if cached is not None and cached[1] is not _ORDER_PENDING:
                            ts2, order_info2 = cached
                            if time.time() - ts2 < self._order_cache_ttl:
                                logger.info(
                                    f"Order {client_order_id} resolved by a concurrent "
                                    f"call while waiting, reusing its result"
                                )
                                return order_info2
                        logger.warning(
                            f"Order {client_order_id} still pending from a concurrent "
                            f"call after waiting -- proceeding to submit"
                        )
                    else:
                        logger.info(
                            f"Order {client_order_id} already placed (idempotent), "
                            f"returning cached result"
                        )
                        return order_info
            # Claim this id immediately, before any `await` below, so a
            # concurrent call for the same client_order_id sees the pending
            # marker above instead of independently passing this same check.
            self._order_cache[client_order_id] = (now, _ORDER_PENDING)

        # --- Duplicate-submission detection (distinct symbol+side arriving
        # close together, e.g. from a concurrent-scan-cycle race) ---
        # getattr-guarded: some tests construct AlpacaExchange with __init__
        # patched out, so these attributes may not exist.
        if not hasattr(self, "_recent_order_submissions"):
            self._recent_order_submissions = {}
        if not hasattr(self, "_duplicate_order_window_sec"):
            self._duplicate_order_window_sec = 10.0
        now = time.time()
        dup_key = (symbol, side.lower())
        recent = [t for t in self._recent_order_submissions.get(dup_key, []) if now - t < self._duplicate_order_window_sec]
        if recent:
            logger.warning(
                f"Possible duplicate order: {symbol} {side} submitted {now - recent[-1]:.1f}s "
                f"after a prior {symbol} {side} submission (client_order_id={client_order_id})"
            )
            try:
                from src.alerting import get_alerting_engine
                task = asyncio.create_task(get_alerting_engine().alert_duplicate_order(
                    symbol, client_order_id or "none",
                    f"{len(recent)} prior {side} submission(s) for {symbol} in last {self._duplicate_order_window_sec:.0f}s",
                ))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except Exception as alert_err:
                logger.debug(f"Duplicate-order alert skipped (non-fatal): {alert_err}")
        recent.append(now)
        self._recent_order_submissions[dup_key] = recent[-10:]  # bounded

        # Parse enums
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        
        # Parse time in force
        tif_map = {
            "ioc": TimeInForce.IOC,
            "gtc": TimeInForce.GTC,
            "fok": TimeInForce.FOK,
            "day": TimeInForce.DAY,
        }
        tif = tif_map.get(time_in_force.lower(), TimeInForce.IOC)
        
        # Build order request based on type
        if type.lower() == "limit":
            if limit_price is None:
                raise ValueError("limit_price is required for limit orders")
            order_kwargs = {
                "symbol": symbol,
                "qty": qty,
                "side": order_side,
                "time_in_force": tif,
                "limit_price": limit_price,
            }
            if post_only:
                order_kwargs["order_class"] = "post_only"
            if client_order_id is not None:
                order_kwargs["client_order_id"] = client_order_id
            request = LimitOrderRequest(**order_kwargs)
        else:
            # Market order (default)
            order_kwargs = {
                "symbol": symbol,
                "qty": qty,
                "side": order_side,
                "time_in_force": tif,
            }
            if client_order_id is not None:
                order_kwargs["client_order_id"] = client_order_id
            request = MarketOrderRequest(**order_kwargs)

        try:
            if bypass_circuit_breaker:
                order = await asyncio.to_thread(self.trading_client.submit_order, request)
            else:
                order = await self.circuit_breaker.call(asyncio.to_thread, self.trading_client.submit_order, request)
            order_id = str(order.id)
            order_info = {
                "id": order_id,
                "symbol": str(order.symbol),
                "qty": float(order.qty) if order.qty else 0.0,
                "status": str(order.status.value) if hasattr(order.status, "value") else str(order.status),
            }
        except Exception as submit_err:
            existing = None
            if client_order_id is not None and not _is_circuit_open_error(submit_err):
                existing = await self._find_existing_order_by_client_id(client_order_id)
            if existing is None:
                # Always log the REAL rejection reason here, not just in the
                # "duplicate found" branch below. Previously this raised
                # silently -- by the time it reached the caller (after
                # tenacity's retries and/or the circuit tripping from
                # repeated failures), the original Alpaca error text (bad
                # qty/precision, insufficient buying power, invalid symbol,
                # etc.) was gone, replaced by a generic "Circuit is OPEN" or
                # RetryError message. Found 2026-09-21 diagnosing a live
                # incident where this made 3 failed buy orders' actual cause
                # unrecoverable from the logs.
                status = getattr(submit_err, "status_code", None) if isinstance(submit_err, APIError) else None
                logger.error(
                    f"submit_order REJECTED for {symbol} {side} qty={qty} "
                    f"(client_order_id={client_order_id!r}, status={status}): {submit_err!r}"
                )
                # Clear the pending claim so a legitimate retry with this
                # same client_order_id isn't blocked for the rest of the
                # cache TTL -- this submission never resolved to a real
                # order, so nothing should be idempotency-cached for it.
                if client_order_id is not None:
                    pending = self._order_cache.get(client_order_id)
                    if pending is not None and pending[1] is _ORDER_PENDING:
                        del self._order_cache[client_order_id]
                raise
            logger.warning(
                f"submit_order raised ({submit_err!r}) but an order with "
                f"client_order_id={client_order_id!r} already exists on the "
                f"exchange (id={existing['id']}) -- using it instead of "
                f"letting the retry resubmit a duplicate."
            )
            order_id = existing["id"]
            order_info = existing

        # Cache for idempotency
        if client_order_id is not None:
            self._order_cache[client_order_id] = (time.time(), order_info)

        if not confirm or not order_id:
            return order_info

        # Confirmation loop
        start_time = time.monotonic()
        while time.monotonic() - start_time < confirm_timeout:
            await asyncio.sleep(0.5)
            try:
                poll_info = await self.get_order(order_id)
                status = poll_info.get("status")
                if status in (OrderStatus.FILLED.value, OrderStatus.CANCELED.value, OrderStatus.EXPIRED.value, OrderStatus.REJECTED.value, "filled", "canceled", "expired", "rejected"):
                    logger.info(f"Order {order_id} reached final status: {status}")
                    if status == OrderStatus.FILLED.value or status == "filled":
                        try:
                            filled_order = await self.circuit_breaker.call(asyncio.to_thread, self.trading_client.get_order_by_id, order_id)
                            order_info["filled_avg_price"] = float(filled_order.filled_avg_price) if filled_order.filled_avg_price else 0.0
                            order_info["filled_qty"] = float(filled_order.filled_qty) if filled_order.filled_qty else 0.0
                            order_info["commission"] = 0.0  # SDK 0.44.0: Order has no .commission
                            order_info["slippage"] = 0.0
                        except Exception as fetch_err:
                            logger.warning(
                                f"Order {order_id} filled but fetching fill details failed "
                                f"({fetch_err!r}); retrying once before falling back."
                            )
                            try:
                                await asyncio.sleep(0.5)
                                filled_order = await self.circuit_breaker.call(asyncio.to_thread, self.trading_client.get_order_by_id, order_id)
                                order_info["filled_avg_price"] = float(filled_order.filled_avg_price) if filled_order.filled_avg_price else 0.0
                                order_info["filled_qty"] = float(filled_order.filled_qty) if filled_order.filled_qty else 0.0
                                order_info["commission"] = 0.0  # SDK 0.44.0: Order has no .commission
                                order_info["slippage"] = 0.0
                            except Exception as retry_err:
                                logger.error(
                                    f"Order {order_id} filled but fill details could not be "
                                    f"retrieved after retry ({retry_err!r}). Falling back to "
                                    f"poll_info (may be less precise than the true fill price)."
                                )
                                order_info["filled_avg_price"] = float(poll_info.get("filled_avg_price", 0.0) or 0.0)
                                order_info["filled_qty"] = float(poll_info.get("qty", order_info.get("qty", 0.0)) or 0.0)
                                order_info["commission"] = 0.0  # SDK 0.44.0
                                order_info["slippage"] = 0.0
                                order_info["fill_data_incomplete"] = True
                    order_info["status"] = status
                    return order_info
            except Exception as e:
                logger.warning(f"Error polling order {order_id}: {e}")

        logger.warning(f"Order {order_id} confirmation timed out after {confirm_timeout}s")

        # Fallback: if we have a client_order_id but no confirmed fill yet,
        # try looking the order up by client_order_id via get_orders()
        if client_order_id is not None and order_id and "filled_avg_price" not in order_info:
            try:
                # status="filled" now maps to Alpaca's broader "closed" query
                # filter (see _ORDER_STATUS_TO_QUERY_STATUS), which also
                # includes cancelled/expired/rejected orders -- filter to the
                # actual fill explicitly rather than trusting the query alone,
                # or a cancelled order with this client_order_id would get
                # mislabeled "filled" below.
                recent_orders = await self.get_orders(limit=50, status="filled")
                for o in recent_orders:
                    if o.get("client_order_id") == client_order_id and o.get("status") == "filled":
                        order_info["status"] = "filled"
                        order_info["filled_avg_price"] = float(o.get("filled_avg_price", 0.0) or 0.0)
                        order_info["filled_qty"] = float(o.get("filled_qty", 0.0) or 0.0)
                        break
            except Exception as e:
                logger.debug(f"Fallback order lookup failed: {e}")

        return order_info
