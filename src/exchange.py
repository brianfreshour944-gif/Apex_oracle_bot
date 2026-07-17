"""Modern exchange module using httpx for async HTTP and Polars for data processing."""

import asyncio
import logging
from typing import Dict, Any, Optional
import polars as pl
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = logging.getLogger("exchange")

class AlpacaExchange:
    """Modern Alpaca exchange client using httpx and Polars."""

    def __init__(self) -> None:
        self.client: Optional[httpx.AsyncClient] = None
        self._loaded: bool = False

    async def load(self) -> None:
        """Initialize the HTTP client and validate credentials."""
        self.client = httpx.AsyncClient(
            base_url=settings.ALPACA_BASE_URL,
            headers={
                "APCA-API-KEY-ID": settings.ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY,
            },
            timeout=30.0,
        )

        try:
            # Test connection
            response = await self.client.get("/v2/account")
            response.raise_for_status()
            self._loaded = True
            logger.info("Alpaca account connected.")
        except Exception as e:
            logger.critical(f"Alpaca connection failure on startup: {e}")
            await self.close()
            raise

    async def close(self) -> None:
        """Clean up resources."""
        if self.client:
            await self.client.aclose()
            self.client = None

    # ---------- Market data ----------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_ohlcv_df(self, symbol: str, timeframe: str = "5m", limit: int = 200) -> pl.DataFrame:
        """Fetch OHLCV data using Polars instead of pandas."""
        try:
            params = {
                "symbols": symbol,
                "timeframe": timeframe,
                "limit": limit,
            }

            response = await self.client.get("/v2/stocks/{symbol}/bars", params=params)
            response.raise_for_status()

            data = response.json()
            if not data or not data.get("bars", []):
                return pl.DataFrame()

            # Convert to Polars DataFrame
            bars = data["bars"][symbol]
            df = pl.DataFrame({
                "timestamp": [bar["t"] for bar in bars],
                "open": [bar["o"] for bar in bars],
                "high": [bar["h"] for bar in bars],
                "low": [bar["l"] for bar in bars],
                "close": [bar["c"] for bar in bars],
                "volume": [bar["v"] for bar in bars],
            })

            # Set timestamp as datetime and sort
            df = df.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ")
            ).sort("timestamp")

            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return pl.DataFrame()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            df = await self.fetch_ohlcv_df(symbol, timeframe="1m", limit=1)
            if not df.is_empty():
                return float(df["close"].item())
            return 0.0
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    # ---------- Account / balances ----------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_account_snapshot(self, symbols: list[str], quote: str) -> tuple[float, float, Dict[str, Any]]:
        """Get account snapshot with positions.

        Returns:
            (equity, buying_power, positions)
        """
        try:
            # Get account info
            account_response = await self.client.get("/v2/account")
            account_response.raise_for_status()
            account_data = account_response.json()

            equity = float(account_data["equity"])
            buying_power = float(account_data["buying_power"])

            # Get positions
            positions: Dict[str, Any] = {}
            for symbol in symbols:
                try:
                    pos_response = await self.client.get(f"/v2/positions/{symbol}")
                    pos_response.raise_for_status()
                    pos_data = pos_response.json()

                    qty = float(pos_data["qty"])
                    price = float(pos_data["current_price"])
                    market_value = qty * price

                    positions[symbol] = {
                        "qty": qty,
                        "price": price,
                        "market_value": market_value
                    }
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        # No position for this symbol
                        continue
                    raise

            return equity, buying_power, positions

        except Exception as e:
            logger.error(f"Error getting account snapshot: {e}")
            raise

    # ---------- Order helpers ----------
    def amount_to_precision(self, symbol: str, amount: float) -> float:
        """Round amount to exchange precision."""
        try:
            return round(float(amount), 8)
        except Exception:
            return float(amount)

    def get_min_qty(self, symbol: str) -> float:
        """Get minimum order quantity for symbol."""
        return 0.0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def market_buy(self, symbol: str, base_qty: float) -> Dict[str, Any]:
        """Submit market buy order."""
        try:
            order_data = {
                "symbol": symbol,
                "qty": str(base_qty),
                "side": "buy",
                "type": "market",
                "time_in_force": "gtc",
            }

            response = await self.client.post("/v2/orders", json=order_data)
            response.raise_for_status()

            order = response.json()
            return {"id": str(order["id"])}

        except Exception as e:
            logger.error(f"Market buy failed for {symbol}: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def market_sell(self, symbol: str, base_qty: float) -> Dict[str, Any]:
        """Submit market sell order."""
        try:
            order_data = {
                "symbol": symbol,
                "qty": str(base_qty),
                "side": "sell",
                "type": "market",
                "time_in_force": "gtc",
            }

            response = await self.client.post("/v2/orders", json=order_data)
            response.raise_for_status()

            order = response.json()
            return {"id": str(order["id"])}

        except Exception as e:
            logger.error(f"Market sell failed for {symbol}: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_filled_price(self, order_id: str, symbol: str, default_price: float) -> float:
        """Poll for the average fill price of a market order."""
        try:
            for _ in range(10):
                response = await self.client.get(f"/v2/orders/{order_id}")
                response.raise_for_status()
                order = response.json()

                if "filled_avg_price" in order and order["filled_avg_price"]:
                    return float(order["filled_avg_price"])

                if order["status"] == "filled" and "filled_avg_price" in order:
                    return float(order["filled_avg_price"])

                await asyncio.sleep(1)

            logger.warning(f"Order {order_id} fill price unresolved in 10s; using {default_price}")
            return default_price

        except Exception as e:
            logger.warning(f"Error fetching order {order_id} fill price: {e}")
            return default_price