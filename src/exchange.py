"""Modern Alpaca exchange client using alpaca-py with rate limiting and circuit breakers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import asyncio
import time
import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

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

logger = get_logger(__name__)


@runtime_checkable
class BaseExchange(Protocol):
    """Abstract Base Exchange Protocol for multi-exchange adapters."""
    async def load(self) -> None: ...
    async def close(self) -> None: ...
    async def get_account(self) -> Dict[str, Any]: ...
    async def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100) -> pl.DataFrame: ...
    async def get_positions(self) -> List[Dict[str, Any]]: ...
    async def create_order(self, symbol: str, qty: float, side: str, type: str = "market", time_in_force: str = "ioc") -> Dict[str, Any]: ...


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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100) -> pl.DataFrame:
        """Get crypto market data using alpaca-py and convert to Polars."""
        if not self.data_client:
            await self.load()

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
                
                return pl.DataFrame(data_list)
        except Exception as e:
            logger.warning(f"Failed to fetch bars for {symbol}: {e}")
            
        return pl.DataFrame()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def create_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        type: str = "market",
        time_in_force: str = "ioc",
        confirm: bool = True,
        confirm_timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Create a new order using alpaca-py."""
        if not self.trading_client:
            await self.load()
            
        # Parse enums
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        
        # We enforce market orders in this bot, but it can be expanded.
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.IOC if time_in_force.lower() == "ioc" else TimeInForce.GTC
        )

        order = await asyncio.to_thread(self.trading_client.submit_order, request)
        order_id = str(order.id)
        
        order_info = {
            "id": order_id,
            "symbol": str(order.symbol),
            "qty": float(order.qty) if order.qty else 0.0,
            "status": str(order.status.value) if hasattr(order.status, "value") else str(order.status),
        }

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
