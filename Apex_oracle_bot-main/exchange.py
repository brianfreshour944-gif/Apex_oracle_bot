# exchange.py
"""Thin async wrapper around ccxt's OKX client.

Encapsulates everything the trading loop needs so the rest of the bot does not
depend on ccxt details. OKX spot has no broker-side "position" object, so a
"position" here is simply a non-dust balance of a base currency; entry price /
PnL are tracked by the bot via its own trade log.
"""
import asyncio
import logging
import os

import ccxt.async_support as ccxt
import pandas as pd

import config

logger = logging.getLogger("exchange")

class OKXExchange:
    def __init__(self):
        hostname = os.getenv("OKX_HOSTNAME", "app.okx.com")
        self.client = ccxt.okx({
            "apiKey": config.OKX_API_KEY,
            "secret": config.OKX_SECRET_KEY,
            "password": config.OKX_PASSPHRASE,
            "enableRateLimit": True,
            "hostname": hostname,
            "options": {
                "createMarketBuyOrderRequiresPrice": False,
                "defaultType": "spot",
                "x-simulated-trading": "1"
            },
        })
        self._markets_loaded = False

        self.client.set_sandbox_mode(True)

    async def load(self):
        # Load instrument definitions from the production endpoint. OKX demo and
        # live share identical instruments, and the demo market-load path in
        # ccxt returns malformed entries, so we load live then flip to demo for
        # trading via the `x-simulated-trading` header below.
        await self.client.load_markets()
        if config.OKX_USE_DEMO:
            # OKX unified demo/paper trading: same host, just this header.
            self.client.headers["x-simulated-trading"] = "1"
        self._markets_loaded = True
        logger.info(f"OKX markets loaded (demo={config.OKX_USE_DEMO}).")

    async def close(self):
        try:
            await self.client.close()
        except Exception as e:
            logger.warning(f"Error closing OKX client: {e}")

    # ---------- Market data ----------
    async def fetch_ohlcv_df(self, symbol: str, timeframe: str = "5m", limit: int = 200) -> pd.DataFrame:
        raw = await self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df = df.set_index("time")
        return df

    async def get_price(self, symbol: str) -> float:
        ticker = await self.client.fetch_ticker(symbol)
        return float(ticker["last"]) if ticker and ticker.get("last") else 0.0

    # ---------- Account / balances ----------
    async def get_account_snapshot(self, symbols, quote: str):
        """Returns (equity, buying_power, positions).

        positions: {symbol: {'qty', 'price', 'market_value'}} for non-dust holdings.
        equity = free+used quote balance + market value of held base assets.
        buying_power = free quote balance.
        """
        balance = await self.client.fetch_balance()
        totals = balance.get("total", {}) or {}
        free = balance.get("free", {}) or {}

        quote_total = float(totals.get(quote, 0.0) or 0.0)
        quote_free = float(free.get(quote, 0.0) or 0.0)

        positions = {}
        holdings_value = 0.0
        for symbol in symbols:
            base = symbol.split("/")[0]
            qty = float(totals.get(base, 0.0) or 0.0)
            if qty <= 0:
                continue
            try:
                price = await self.get_price(symbol)
            except Exception as e:
                logger.warning(f"Price fetch failed for {symbol}: {e}")
                continue
            market_value = qty * price
            holdings_value += market_value
            positions[symbol] = {"qty": qty, "price": price, "market_value": market_value}

        equity = quote_total + holdings_value
        return equity, quote_free, positions

    # ---------- Order helpers ----------
    def amount_to_precision(self, symbol: str, amount: float) -> float:
        try:
            return float(self.client.amount_to_precision(symbol, amount))
        except Exception:
            return float(amount)

    def min_amount(self, symbol: str) -> float:
        try:
            m = self.client.market(symbol)
            return float(m["limits"]["amount"]["min"] or 0.0)
        except Exception:
            return 0.0

    def get_min_qty(self, symbol: str) -> float:
        """
        Return the minimum tradable quantity (minSz) for a given symbol on OKX.
        Returns 0.0 if the instrument info is not available.
        """
        # Ensure instrument data is loaded; load() populates self.client.markets
        if not hasattr(self, "client") or not hasattr(self.client, "markets"):
            logger.warning(f"Exchange markets not loaded; defaulting min qty for {symbol} to 0")
            return 0.0

        market = self.client.market(symbol)
        if not market:
            logger.warning(f"Market data missing for {symbol}, defaulting min qty to 0")
            return 0.0

        # OKX uses 'minSz' for the minimum order size (quantity)
        min_sz = market.get("info", {}).get("minSz")
        if min_sz is None:
            logger.warning(f"No minSz found for {symbol}, defaulting to 0")
            return 0.0

        try:
            return float(min_sz)
        except Exception:
            logger.warning(f"Unable to parse minSz '{min_sz}' for {symbol}, defaulting to 0")
            return 0.0

    async def market_buy(self, symbol: str, base_qty: float):
        amt = self.amount_to_precision(symbol, base_qty)
        return await self.client.create_order(
            symbol, "market", "buy", amt, None, {"tgtCcy": "base_ccy"}
        )

    async def market_sell(self, symbol: str, base_qty: float):
        amt = self.amount_to_precision(symbol, base_qty)
        return await self.client.create_order(
            symbol, "market", "sell", amt, None, {"tgtCcy": "base_ccy"}
        )

    async def get_filled_price(self, order_id: str, symbol: str, default_price: float) -> float:
        """Polls for the average fill price of a market order."""
        for _ in range(10):
            try:
                order = await self.client.fetch_order(order_id, symbol)
                if order.get("average"):
                    return float(order["average"])
                if order.get("status") == "closed" and order.get("price"):
                    return float(order["price"])
            except Exception as e:
                logger.warning(f"Error fetching order {order_id}: {e}")
            await asyncio.sleep(1)
        logger.warning(f"Order {order_id} fill price unresolved in 10s; using {default_price}")
        return default_price
