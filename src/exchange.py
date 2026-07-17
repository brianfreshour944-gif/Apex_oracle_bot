"""Modern exchange integration using httpx and Polars for data processing."""

import asyncio
import httpx
import polars as pl
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

class AlpacaExchange:
    """Modern Alpaca exchange client using httpx and Polars."""

    def __init__(self):
        self.client = None
        self.base_url = settings.ALPACA_BASE_URL
        self.api_key = settings.ALPACA_API_KEY
        self.secret_key = settings.ALPACA_SECRET_KEY
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    async def load(self) -> None:
        """Initialize the exchange client."""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=30.0,
        )
        logger.info(f"Alpaca client initialized for {self.base_url}")

    async def close(self) -> None:
        """Close the exchange client."""
        if self.client:
            await self.client.aclose()
            logger.info("Alpaca client closed")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        if not self.client:
            await self.load()

        response = await self.client.get("/v2/account")
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100) -> pl.DataFrame:
        """Get market data using Polars for efficient processing."""
        if not self.client:
            await self.load()

        params = {
            "symbols": symbol,
            "timeframe": timeframe,
            "limit": limit,
        }

        response = await self.client.get("/v2/stocks/bars", params=params)
        response.raise_for_status()

        # Convert to Polars DataFrame for modern data processing
        data = response.json()
        if symbol in data:
            return pl.DataFrame(data[symbol])
        return pl.DataFrame()

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get open positions."""
        if not self.client:
            await self.load()

        response = await self.client.get("/v2/positions")
        response.raise_for_status()
        return response.json()

    async def create_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        type: str = "market",
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """Create a new order."""
        if not self.client:
            await self.load()

        order_data = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": type,
            "time_in_force": time_in_force,
        }

        response = await self.client.post("/v2/orders", json=order_data)
        response.raise_for_status()
        return response.json()