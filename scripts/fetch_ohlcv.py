"""Fetch and cache hourly OHLCV bars from Alpaca's market-data API.

Read-only: downloads public market data and writes JSON caches under data/ohlcv/.
Never contacts the trading endpoint, never places orders. Stdlib only so it runs
without the project venv.

Usage:
    python3 scripts/fetch_ohlcv.py --symbols BTC/USD ETH/USD SOL/USD \
        --start 2025-10-01 --end 2026-10-24 --timeframe 1Hour
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "ohlcv"


def _headers() -> dict[str, str]:
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        sys.exit("ALPACA_API_KEY / ALPACA_SECRET_KEY must be set (data-only use).")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def fetch_symbol(symbol: str, start: str, end: str, timeframe: str) -> list[dict]:
    bars: list[dict] = []
    token: str | None = None
    while True:
        params = {
            "symbols": symbol,
            "timeframe": timeframe,
            "start": f"{start}T00:00:00Z",
            "end": f"{end}T23:59:59Z",
            "limit": "10000",
        }
        if token:
            params["page_token"] = token
        url = f"{BASE}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=_headers())
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    payload = json.loads(resp.read())
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 or 500 <= exc.code < 600:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        else:
            raise RuntimeError(f"repeated failures for {symbol}")
        bars.extend(payload.get("bars", {}).get(symbol, []))
        token = payload.get("next_page_token")
        if not token:
            break
        time.sleep(0.3)
    return bars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTC/USD", "ETH/USD", "SOL/USD"])
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-10-24")
    ap.add_argument("--timeframe", default="1Hour")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in args.symbols:
        bars = fetch_symbol(symbol, args.start, args.end, args.timeframe)
        safe = symbol.replace("/", "_")
        out = CACHE_DIR / f"{safe}_{args.timeframe}.json"
        out.write_text(json.dumps({"symbol": symbol, "timeframe": args.timeframe,
                                   "bars": bars}))
        first = bars[0]["t"] if bars else "n/a"
        last = bars[-1]["t"] if bars else "n/a"
        print(f"{symbol}: {len(bars)} bars {first} -> {last}  -> {out}")


if __name__ == "__main__":
    main()
