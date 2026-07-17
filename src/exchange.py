  
"""Modern Alpaca exchange client using httpx and Polars."""  
  
from __future__ import annotations  
  
from typing import Any, Dict, List  
  
import httpx  
from tenacity import retry, stop_after_attempt, wait_exponential  
  
import polars as pl  
  
from src.config import settings  
from src.logging_config import get_logger  
  
logger = get_logger(__name__)  
  
class AlpacaExchange:  
    """Modern Alpaca exchange client using httpx and Polars."""  
  
    def __init__(self):
        self.client = None
        self.data_client = None
        self.base_url = settings.ALPACA_BASE_URL
        self.data_base_url = "https://data.alpaca.markets"
        self.api_key = settings.ALPACA_API_KEY
        self.secret_key = settings.ALPACA_SECRET_KEY
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
  
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
            resp = await self.client.get("/v2/account")  
            if resp.status_code == 401:  
                raise RuntimeError(  
                    "Alpaca returned 401 Unauthorized. Check ALPACA_API_KEY/ALPACA_SECRET_KEY "  
                    "are valid and match the environment (paper keys -^> paper-api.alpaca.markets, "  
                    "live keys -^> api.alpaca.markets). base_url=" + self.base_url  
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

        response = await self.client.get("/v2/account")
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100) -> pl.DataFrame:
        """Get crypto market data from Alpaca's data API using Polars."""
        if not self.data_client:
            await self.load()

        params = {
            "symbols": symbol,
            "timeframe": timeframe,
            "limit": limit,
        }

        response = await self.data_client.get("/v2/crypto/bars", params=params)
        response.raise_for_status()

        # Convert to Polars DataFrame for modern data processing
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
