"""Modern Alpaca exchange client using httpx and Polars with rate limiting, circuit breakers, and order confirmation."""

from __future__ import annotations

from typing import Any, Dict, List, AsyncGenerator, Optional
import json
import asyncio
import random
import time
import websockets

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

import polars as pl

from src.config import settings
from src.logging_config import get_logger
from src.circuit_breaker import CircuitBreaker

from typing import Protocol, runtime_checkable

logger = get_logger(__name__)


@runtime_checkable
class BaseExchange(Protocol):
    """Abstract Base Exchange Protocol for multi-exchange adapters (Alpaca, OKX, Binance, Bybit)."""

    async def load(self) -> None: ...
    async def close(self) -> None: ...
    async def get_account(self) -> Dict[str, Any]: ...
    async def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100) -> pl.DataFrame: ...
    async def get_positions(self) -> List[Dict[str, Any]]: ...
    async def create_order(self, symbol: str, qty: float, side: str, type: str = "market", time_in_force: str = "ioc") -> Dict[str, Any]: ...



class RateLimiter:
    """Token-bucket rate limiter for API calls."""

    def __init__(self, max_rate: float = 200.0, time_period: float = 60.0):
        self.max_tokens = max_rate
        self.tokens = max_rate
        self.time_period = time_period
        self.fill_rate = max_rate / time_period
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token from the bucket, pausing if rate limit is reached."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.fill_rate)

            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.fill_rate
                logger.debug(f"Rate limit hit. Waiting {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0


class AlpacaExchange:
    """Modern Alpaca exchange client using httpx and Polars."""

    def __init__(self):
        self.client = None
        self.data_client = None
        base = settings.ALPACA_BASE_URL or "https://paper-api.alpaca.markets"
        if not base.startswith(("http://", "https://")):
            base = "https://" + base
        self.base_url = base
        self.data_base_url = "https://data.alpaca.markets"
        self.api_key = settings.ALPACA_API_KEY
        self.secret_key = settings.ALPACA_SECRET_KEY
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        self.rate_limiter = RateLimiter(max_rate=180.0, time_period=60.0)
        self.circuit_breaker = CircuitBreaker("alpaca_exchange")

    async def load(self) -> None:
        """Initialize the exchange client and verify credentials."""
        if not (self.api_key and self.secret_key):
            raise RuntimeError(
                "Alpaca credentials missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "(e.g. in Coolify environment variables)."
            )
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=30.0,
        )
        try:
            await self.rate_limiter.acquire()
            resp = await self.client.get("/v2/account")
            if resp.status_code == 401:
                raise RuntimeError(
                    "Alpaca returned 401 Unauthorized. Check ALPACA_API_KEY/ALPACA_SECRET_KEY "
                    "are valid and match the environment. base_url=" + self.base_url
                )
            resp.raise_for_status()
        except Exception:
            await self.client.aclose()
            self.client = None
            raise
        logger.info(f"Alpaca client initialized for {self.base_url}")
        self.data_client = httpx.AsyncClient(
            base_url=self.data_base_url,
            headers=self.headers,
            timeout=30.0,
        )
        logger.info(f"Alpaca data client initialized for {self.data_base_url}")

    async def close(self) -> None:
        """Close the exchange client."""
        if self.client:
            await self.client.aclose()
            logger.info("Alpaca client closed")
        if self.data_client:
            await self.data_client.aclose()
            logger.info("Alpaca data client closed")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        if not self.client:
            await self.load()
        await self.rate_limiter.acquire()
        response = await self.client.get("/v2/account")
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100) -> pl.DataFrame:
        """Get crypto market data from Alpaca's data API using Polars."""
        if not self.data_client:
            await self.load()

        import datetime
        import re

        now = datetime.datetime.now(datetime.timezone.utc)
        match = re.match(r'(\d+)\s*([a-zA-Z]+)?', timeframe)
        if match:
            val = int(match.group(1))
            unit = match.group(2)
            if unit:
                unit_lower = unit.lower()
                if 'min' in unit_lower or unit_lower == 'm':
                    delta = datetime.timedelta(minutes=val * limit * 1.5)
                elif 'hour' in unit_lower or unit_lower == 'h':
                    delta = datetime.timedelta(hours=val * limit * 1.5)
                elif 'day' in unit_lower or unit_lower == 'd':
                    delta = datetime.timedelta(days=val * limit * 1.5)
                else:
                    delta = datetime.timedelta(days=limit * 1.5)
            else:
                delta = datetime.timedelta(days=limit * 1.5)
        else:
            delta = datetime.timedelta(days=limit * 1.5)

        min_delta = datetime.timedelta(hours=1)
        if delta < min_delta:
            delta = min_delta

        start_time = now - delta

        params = {
            "symbols": symbol,
            "timeframe": timeframe,
            "limit": limit,
            "start": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        await self.rate_limiter.acquire()
        response = await self.data_client.get("/v1beta3/crypto/us/bars", params=params)
        if response.status_code in (401, 403, 404):
            body = response.text[:300]
            raise RuntimeError(
                f"Crypto market data unavailable for {symbol} (HTTP {response.status_code}): {body}"
            )
        response.raise_for_status()

        data = response.json()
        bars = data.get("bars", {})
        if not bars or symbol not in bars:
            return pl.DataFrame()
        df = pl.DataFrame(bars[symbol])
        rename_map = {
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "trade_count",
        }
        df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})
        return df

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_latest_bar(self, symbol: str) -> pl.DataFrame:
        """Get the latest bar for a crypto symbol using the dedicated latest/bars endpoint."""
        if not self.data_client:
            await self.load()

        params = {"symbols": symbol}
        await self.rate_limiter.acquire()
        response = await self.data_client.get("/v1beta3/crypto/us/latest/bars", params=params)
        response.raise_for_status()

        data = response.json()
        bars = data.get("bars", {})
        if not bars or symbol not in bars:
            return pl.DataFrame()

        bar = bars[symbol]
        if isinstance(bar, dict):
            bar = [bar]
        df = pl.DataFrame(bar)
        rename_map = {
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "trade_count",
        }
        df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})
        return df

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get open positions."""
        if not self.client:
            await self.load()
        await self.rate_limiter.acquire()
        response = await self.client.get("/v2/positions")
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Fetch order details by order ID."""
        if not self.client:
            await self.load()
        await self.rate_limiter.acquire()
        response = await self.client.get(f"/v2/orders/{order_id}")
        response.raise_for_status()
        return response.json()

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
        """Create a new order with optional confirmation polling."""
        if not self.client:
            await self.load()

        order_data = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": type,
            "time_in_force": time_in_force,
        }

        await self.rate_limiter.acquire()
        response = await self.client.post("/v2/orders", json=order_data)
        response.raise_for_status()
        order_info = response.json()
        order_id = order_info.get("id")

        if not confirm or not order_id:
            return order_info

        # Confirmation loop: poll order until filled, canceled, or expired
        start_time = time.monotonic()
        while time.monotonic() - start_time < confirm_timeout:
            await asyncio.sleep(0.5)
            try:
                poll_info = await self.get_order(order_id)
                status = poll_info.get("status")
                if status in ("filled", "canceled", "expired", "rejected"):
                    logger.info(f"Order {order_id} reached final status: {status}")
                    return poll_info
            except Exception as e:
                logger.warning(f"Error polling order {order_id}: {e}")

        logger.warning(f"Order {order_id} confirmation timed out after {confirm_timeout}s")
        return order_info

    async def listen_crypto_bars(self, symbols: List[str]) -> AsyncGenerator[Dict[str, Any], None]:
        """Listen to real-time crypto bars with exponential backoff, jitter, and circuit breaker."""
        wss_url = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
        attempt = 0
        min_delay = 1.0
        max_delay = 60.0

        while True:
            try:
                async with websockets.connect(wss_url) as ws:
                    logger.info("Connected to Alpaca Crypto WebSocket")
                    attempt = 0  # reset attempt count on successful connect

                    connected_msg = json.loads(await ws.recv())
                    logger.debug(f"WebSocket connected: {connected_msg}")

                    auth_message = {
                        "action": "auth",
                        "key": self.api_key,
                        "secret": self.secret_key
                    }
                    await ws.send(json.dumps(auth_message))
                    auth_response = json.loads(await ws.recv())
                    logger.info(f"WebSocket Auth Response: {auth_response}")
                    if isinstance(auth_response, list) and auth_response[0].get("T") == "error":
                        err_msg = auth_response[0].get("msg", "")
                        if "connection limit" in err_msg.lower():
                            logger.warning("⚠️ Alpaca WebSocket connection limit exceeded (another bot instance is using the API key). Backing off 15s...")
                            await asyncio.sleep(15.0)
                        raise ValueError(f"WebSocket auth failed: {err_msg}")


                    sub_message = {
                        "action": "subscribe",
                        "bars": symbols
                    }
                    await ws.send(json.dumps(sub_message))
                    sub_response = json.loads(await ws.recv())
                    logger.info(f"WebSocket Subscription Response: {sub_response}")
                    if isinstance(sub_response, list) and sub_response[0].get("T") == "error":
                        logger.error(f"WebSocket subscription failed: {sub_response[0].get('msg')}")

                    while True:
                        message = await ws.recv()
                        data = json.loads(message)
                        for item in data:
                            if item.get("T") == "b":
                                yield item

            except Exception as e:
                attempt += 1
                backoff = min(max_delay, min_delay * (2 ** (attempt - 1))) + random.uniform(0, 1.0)
                logger.error(f"WebSocket connection error (attempt {attempt}): {e}. Reconnecting in {backoff:.2f}s...")
                await asyncio.sleep(backoff)
