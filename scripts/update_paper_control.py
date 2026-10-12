#!/usr/bin/env python3
"""Mark the $100 buy-and-hold control account to market. Read-only except for
its own ledger under data/. Places no orders.

Usage:
    ALPACA_API_KEY=x ALPACA_SECRET_KEY=y python scripts/update_paper_control.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from src.paper_control import CONTROL_SYMBOLS, summary, update


def _last_prices() -> dict[str, float]:
    """Latest close per symbol. Uses yfinance (public, no auth)."""
    import yfinance as yf
    prices = {}
    for s in CONTROL_SYMBOLS:
        try:
            df = yf.Ticker(s.replace("/", "-")).history(period="5d", interval="1d")
            if df is not None and not df.empty:
                prices[s] = float(df["Close"].iloc[-1])
        except Exception as e:
            print(f"  {s}: price fetch failed ({e})")
    return prices


def main() -> int:
    prices = _last_prices()
    if not prices:
        print("no prices fetched; control unchanged")
        return 1
    print("prices:", {k: round(v, 2) for k, v in prices.items()})
    s = update(prices)
    print(f"control equity: ${s['equity_usd']:.2f}  ({s['return_pct']:+.2f}%)  "
          f"[entry fee {s['entry_fees_usd']:.4f} USD]")
    for sym, d in s["per_symbol"].items():
        print(f"  {sym:8} entry ${d['entry_price']:.2f} -> ${d['mark_price']:.2f}  "
              f"{d['return_pct']:+.2f}%")
    print("ledger:", summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
