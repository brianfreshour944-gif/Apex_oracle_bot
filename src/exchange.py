"""Modern Alpaca exchange client using alpaca-py with rate limiting and circuit breakers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import asyncio
import time
import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed, retry_if_exception, RetryCallState

import polars as pl

from src.config import settings
from src.logging_config import get_logger
from src.circuit_breaker import CircuitBreaker

from typing import Protocol, runtime_checkable

# Import alpaca-py components
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestBarRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.common.exceptions import APIError

logger = get_logger(__name__)


def _is_rate_limit_error(exception: Exception) -> bool:
    """Return True if the exception is an Alpaca 429 rate-limit error."""
    if isinstance(exception, APIError):
        return getattr(exception, "status_code", 0) == 429
    return False


def _retry_on_rate_limit(exception: Exception) -> bool:
    """Retry predicate: retry on rate-limit errors with longer backoff."""
    return _is_rate_limit_error(exception)


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
    async def get_account(self) -> Dict[str, Any]: ...
    async def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100) -> pl.DataFrame: ...
    async def get_positions(self) -> List[Dict[str, Any]]: ...
    async def create_order(self, symbol: str, qty: float, side: str, type: str = "market", time_in_force: str = "ioc", client_order_id: Optional[str] = None) -> Dict[str, Any]: ...


class AlpacaExchange:
    """Modern Alpaca exchange client using alpaca-py."""

    def __init__(self):
        self.api_key = settings.ALPACA_API_KEY
        self.secret_key = settings.ALPACA_SECRET_KEY
        base_url = settings.ALPACA_BASE_URL or "https://paper-api.alpaca.markets"
        self.paper = "paper" in base_url.lower()
        
        self.trading_client: Optional[TradingClient] = None
        self.data_client: Optional[CryptoHistoricalDataClient] = None
        
        self.circuit_breaker = CircuitBreaker("alpaca_exchange")
        # Internal lock for mimicking rate limiter if needed, though SDK handles some
        self._lock = asyncio.Lock()
        # Rate-limit tracking
        self._rate_limit_remaining = 200
        self._rate_limit_reset = 0.0
        self._rate_limit_hits = 0
        # Bar cache: (symbol, timeframe, limit) -> (timestamp, DataFrame)
        self._bars_cache: Dict[tuple, Tuple[float, pl.DataFrame]] = {}
        self._bars_cache_ttl: float = 60.0  # cache bars for 60 seconds
        # Idempotency cache: client_order_id -> (timestamp, order_info)
        self._order_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._order_cache_ttl: float = 300.0  # cache orders for 5 minutes

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
        except Exception as e:
            self.trading_client = None
            self.data_client = None
            raise RuntimeError(f"Failed to initialize Alpaca clients: {e}")
            
        logger.info(f"Alpaca client initialized (Paper={self.paper})")

    async def close(self) -> None:
        """Close the exchange client."""
        self.trading_client = None
        self.data_client = None
        logger.info("Alpaca client closed")

    @retry(retry=retry_if_exception(_retry_on_rate_limit), stop=stop_after_attempt(5), wait=_rate_limit_wait)
    async def get_account(self) -> Dict[str, Any]:
        """Get account information (formatted to dict for compatibility)."""
        if not self.trading_client:
            await self.load()
            
        account = await asyncio.to_thread(self.trading_client.get_account)
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
    async def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100) -> pl.DataFrame:
        """Get crypto market data using alpaca-py and convert to Polars."""
        if not self.data_client:
            await self.load()

        cache_key = (symbol, timeframe, limit)
        now = time.time()
        cached = self._bars_cache.get(cache_key)
        if cached is not None:
            cached_ts, cached_df = cached
            if now - cached_ts < self._bars_cache_ttl and len(cached_df) >= limit:
                return cached_df

        tf = self._parse_timeframe(timeframe)
        
        # Calculate start time heuristically based on limit
        now = datetime.datetime.now(datetime.timezone.utc)
        if tf.unit == TimeFrameUnit.Minute:
            delta = datetime.timedelta(minutes=tf.amount * limit * 1.5)
        elif tf.unit == TimeFrameUnit.Hour:
            delta = datetime.timedelta(hours=tf.amount * limit * 1.5)
        else:
            delta = datetime.timedelta(days=tf.amount * limit * 1.5)
            
        start_time = now - max(delta, datetime.timedelta(hours=1))
        
        request_params = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start_time,
            limit=limit
        )

        try:
            bars_df = await asyncio.to_thread(self.data_client.get_crypto_bars, request_params)
            if bars_df.data and symbol in bars_df.data:
                # Get the list of Bar objects
                bars = bars_df.data[symbol]
                
                # Convert to dict format expected by downstream
                data_list = []
                for b in bars:
                    data_list.append({
                        "timestamp": b.timestamp.isoformat() if hasattr(b.timestamp, "isoformat") else b.timestamp,
                        "open": float(b.open),
                        "high": float(b.high),
                        "low": float(b.low),
                        "close": float(b.close),
                        "volume": float(b.volume),
                        "vwap": float(b.vwap),
                        "trade_count": int(b.trade_count),
                    })
                
                df_result = pl.DataFrame(data_list)
                self._bars_cache[cache_key] = (time.time(), df_result)
                return df_result
        except Exception as e:
            logger.warning(f"Failed to fetch bars for {symbol}: {e}")

        result = pl.DataFrame()
        self._bars_cache[cache_key] = (time.time(), result)
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

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get open positions."""
        if not self.trading_client:
            await self.load()
            
        positions = await asyncio.to_thread(self.trading_client.get_all_positions)
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

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Fetch order details by order ID."""
        if not self.trading_client:
            await self.load()
            
        order = await asyncio.to_thread(self.trading_client.get_order_by_id, order_id)
        return {
            "id": str(order.id),
            "symbol": str(order.symbol),
            "qty": float(order.qty) if order.qty else 0.0,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0.0,
            "status": str(order.status.value) if hasattr(order.status, "value") else str(order.status),
            "side": str(order.side.value) if hasattr(order.side, "value") else str(order.side),
            "type": str(order.type.value) if hasattr(order.type, "value") else str(order.type),
        }

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def create_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        type: str = "market",
        time_in_force: str = "ioc",
        confirm: bool = True,
        confirm_timeout: float = 10.0,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new order using alpaca-py.

        If ``client_order_id`` is provided and a matching order was already
        placed recently (within the idempotency window), the cached result
        is returned instead of re-submitting — preventing double-submits
        from confirmation timeouts.
        """
        if not self.trading_client:
            await self.load()

        # --- Idempotency check ---
        if client_order_id is not None:
            now = time.time()
            cached = self._order_cache.get(client_order_id)
            if cached is not None:
                ts, order_info = cached
                if now - ts < self._order_cache_ttl:
                    logger.info(
                        f"Order {client_order_id} already placed (idempotent), "
                        f"returning cached result"
                    )
                    return order_info

        # Parse enums
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        
        # We enforce market orders in this bot, but it can be expanded.
        order_kwargs = dict(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.IOC if time_in_force.lower() == "ioc" else TimeInForce.GTC,
        )
        if client_order_id is not None:
            order_kwargs["client_order_id"] = client_order_id

        request = MarketOrderRequest(**order_kwargs)

        order = await asyncio.to_thread(self.trading_client.submit_order, request)
        order_id = str(order.id)

        order_info = {
            "id": order_id,
            "symbol": str(order.symbol),
            "qty": float(order.qty) if order.qty else 0.0,
            "status": str(order.status.value) if hasattr(order.status, "value") else str(order.status),
        }

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
                            filled_order = await asyncio.to_thread(self.trading_client.get_order_by_id, order_id)
                            order_info["filled_avg_price"] = float(filled_order.filled_avg_price) if filled_order.filled_avg_price else 0.0
                            order_info["filled_qty"] = float(filled_order.filled_qty) if filled_order.filled_qty else 0.0
                            order_info["commission"] = float(filled_order.commission) if filled_order.commission else 0.0
                            order_info["slippage"] = 0.0
                        except Exception:
                            order_info["filled_avg_price"] = 0.0
                            order_info["filled_qty"] = 0.0
                            order_info["commission"] = 0.0
                            order_info["slippage"] = 0.0
                    return poll_info
            except Exception as e:
                logger.warning(f"Error polling order {order_id}: {e}")

        logger.warning(f"Order {order_id} confirmation timed out after {confirm_timeout}s")
        return order_info
