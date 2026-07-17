# exchange.py
"""Async wrapper around the Alpaca trading + data API (crypto focus).

Alpaca Crypto uses symbols like 'BTC/USD' and trades 24/7, which fits this
bot's always-on regime loop. Alpaca has real broker-side positions, so
get_account_snapshot reads them directly.

The interface matches what bot.py expects: market_buy/market_sell return a dict
with an "id" key, and get_account_snapshot returns
(equity, buying_power, positions).
"""
import asyncio
import logging
import os

import alpaca_trade_api as tradeapi
import pandas as pd

import config

logger = logging.getLogger("exchange")

# Alpaca paper trading endpoint. Override with ALPACA_BASE_URL for live.
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


class AlpacaExchange:
    def __init__(self):
        self.api = tradeapi.REST(
            key_id=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            base_url=ALPACA_BASE_URL,
        )
        self._loaded = False

    async def load(self):
        # Alpaca REST is lazy; a lightweight call validates credentials.
        try:
            await asyncio.to_thread(self.api.get_account)
            self._loaded = True
            logger.info("Alpaca account connected.")
        except Exception as e:
            logger.critical(f"Alpaca connection failure on startup: {e}")
            raise

    async def close(self):
        # REST client is stateless; nothing to tear down.
        pass

    # ---------- Market data ----------
    async def fetch_ohlcv_df(self, symbol: str, timeframe: str = "5m", limit: int = 200) -> pd.DataFrame:
        bars = await asyncio.to_thread(
            self.api.get_crypto_bars, symbol, timeframe, limit=limit
        )
        if bars is None or bars.df is None or bars.df.empty:
            return pd.DataFrame()
        df = bars.df
        # Crypto bars come back with a (symbol, timestamp) MultiIndex.
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
        df = df[["open", "high", "low", "close", "volume"]]
        return df

    async def get_price(self, symbol: str) -> float:
        bars = await asyncio.to_thread(self.api.get_crypto_bars, symbol, "1m", limit=1)
        if bars is not None and bars.df is not None and not bars.df.empty:
            return float(bars.df["close"].iloc[-1])
        return 0.0

    # ---------- Account / balances ----------
    async def get_account_snapshot(self, symbols, quote: str):
        """Returns (equity, buying_power, positions).

        positions: {symbol: {'qty', 'price', 'market_value'}} for held assets.
        equity / buying_power are reported by Alpaca directly in USD.
        """
        account = await asyncio.to_thread(self.api.get_account)
        equity = float(account.equity)
        buying_power = float(account.buying_power)

        positions = {}
        for symbol in symbols:
            try:
                pos = await asyncio.to_thread(self.api.get_position, symbol)
            except Exception:
                # No open position for this symbol is normal, not an error.
                continue
            qty = float(pos.qty)            # crypto qty is in base units
            price = float(pos.current_price)
            market_value = qty * price
            positions[symbol] = {"qty": qty, "price": price, "market_value": market_value}

        return equity, buying_power, positions

    # ---------- Order helpers ----------
    def amount_to_precision(self, symbol: str, amount: float) -> float:
        # Alpaca accepts fractional crypto quantities; 8 dp is a safe default.
        try:
            return round(float(amount), 8)
        except Exception:
            return float(amount)

    def get_min_qty(self, symbol: str) -> float:
        # Most Alpaca crypto pairs have no hard min qty; return a tiny floor.
        return 0.0

    async def market_buy(self, symbol: str, base_qty: float):
        order = await asyncio.to_thread(
            self.api.submit_order,
            symbol=symbol, qty=base_qty, side="buy",
            type="market", time_in_force="gtc",
        )
        # Normalize to the dict shape bot.py expects (order.get("id")).
        return {"id": str(order.id)}

    async def market_sell(self, symbol: str, base_qty: float):
        order = await asyncio.to_thread(
            self.api.submit_order,
            symbol=symbol, qty=base_qty, side="sell",
            type="market", time_in_force="gtc",
        )
        return {"id": str(order.id)}

    async def get_filled_price(self, order_id: str, symbol: str, default_price: float) -> float:
        """Polls for the average fill price of a market order."""
        for _ in range(10):
            try:
                order = await asyncio.to_thread(self.api.get_order, order_id)
                if getattr(order, "filled_avg_price", None):
                    return float(order.filled_avg_price)
                if order.status == "filled" and getattr(order, "filled_avg_price", None):
                    return float(order.filled_avg_price)
            except Exception as e:
                logger.warning(f"Error fetching order {order_id}: {e}")
            await asyncio.sleep(1)
        logger.warning(f"Order {order_id} fill price unresolved in 10s; using {default_price}")
        return default_price