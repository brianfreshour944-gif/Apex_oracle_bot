"""
src/data_fetcher.py — Live market data fetching via Alpaca.
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

log = logging.getLogger(__name__)

BAR_TIMEFRAME = TimeFrame(15, TimeFrameUnit.Minute)


def fetch_bars(client: CryptoHistoricalDataClient, symbol: str, days: int) -> pd.DataFrame | None:
    try:
        start = datetime.now(tz=timezone.utc) - timedelta(days=days)
        req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=BAR_TIMEFRAME, start=start)
        raw_bars = client.get_crypto_bars(req).data.get(symbol, [])
        if not raw_bars:
            return None
        df = pd.DataFrame([{
            "timestamp": b.timestamp, "open": float(b.open or 0), "high": float(b.high or 0),
            "low": float(b.low or 0), "close": float(b.close or 0), "volume": float(b.volume or 0),
            "vwap": float(b.vwap or 0), "trade_count": float(b.trade_count or 0),
        } for b in raw_bars])
        df.set_index("timestamp", inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df["vwap"] = df["vwap"].where(df["vwap"] > 0, df["close"])
        return df[df["close"] > 0]
    except Exception as exc:
        log.error(f"  {symbol}: fetch failed — {exc}")
        return None
