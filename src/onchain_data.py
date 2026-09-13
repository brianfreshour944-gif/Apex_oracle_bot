"""Derivatives & On-Chain Data Fetcher.

Pulls macro crypto data (Funding Rates, Open Interest, Long/Short Ratios)
from public Binance Futures endpoints to augment technical indicators.

Adds Z-scored versions for regime-invariant comparison across time.
"""

import asyncio
import time
from collections import deque

import httpx

from src.logging_config import get_logger

logger = get_logger("onchain")

# Rolling windows for Z-scoring (per symbol)
_DERIVATIVES_HISTORY: dict[str, dict[str, deque]] = {}
_HISTORY_MAXLEN = 200  # ~200 5-min intervals = ~16 hours


def _get_symbol_history(symbol: str) -> dict[str, deque]:
    """Get or create rolling history buffers for a symbol."""
    if symbol not in _DERIVATIVES_HISTORY:
        _DERIVATIVES_HISTORY[symbol] = {
            "funding_rate": deque(maxlen=_HISTORY_MAXLEN),
            "open_interest": deque(maxlen=_HISTORY_MAXLEN),
            "long_short_ratio": deque(maxlen=_HISTORY_MAXLEN),
            "bid_ask_imbalance": deque(maxlen=_HISTORY_MAXLEN),
        }
    return _DERIVATIVES_HISTORY[symbol]


def _z_score_latest(history: deque, latest: float) -> float:
    """Compute Z-score of latest value against rolling history."""
    if len(history) < 20:
        return 0.0
    arr = np.array(history)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-8:
        return 0.0
    return float((latest - mean) / std)


# Import numpy for Z-scoring
import numpy as np


def fetch_derivatives_data_sync(symbol: str) -> dict[str, float]:
    """
    Synchronous version: Fetches Open Interest, Funding Rate, and Long/Short ratio from Binance Futures.
    Alpaca symbols are usually 'BTC/USD', so we convert to 'BTCUSDT' for Binance.
    
    Returns both raw and Z-scored values for regime-invariant analysis.
    """
    base_asset = symbol.split("/")[0] if "/" in symbol else symbol.replace("USD", "")
    binance_symbol = f"{base_asset}USDT"
    
    # Get rolling history for Z-scoring
    history = _get_symbol_history(symbol)
    
    raw_data = {
        "funding_rate": 0.0,
        "open_interest": 0.0,
        "long_short_ratio": 1.0,
        "bid_ask_imbalance": 0.0
    }
    
    try:
        with httpx.Client(timeout=5.0) as client:
            # 1. Funding Rate (Premium Index)
            funding_url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={binance_symbol}"
            
            # 2. Open Interest
            oi_url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={binance_symbol}"
            
            # 3. Global Long/Short Ratio (top traders, 5m timeframe)
            ls_url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={binance_symbol}&period=5m&limit=1"
            # 4. L2 Order Book Depth
            depth_url = f"https://fapi.binance.com/fapi/v1/depth?symbol={binance_symbol}&limit=50"
            
            responses = [
                client.get(funding_url),
                client.get(oi_url),
                client.get(ls_url),
                client.get(depth_url),
            ]
            
            # Parse Funding Rate
            if responses[0].status_code == 200:
                payload = responses[0].json()
                raw_data["funding_rate"] = float(payload.get("lastFundingRate", 0.0))
                
            # Parse Open Interest
            if responses[1].status_code == 200:
                payload = responses[1].json()
                raw_data["open_interest"] = float(payload.get("openInterest", 0.0))
                
            # Parse Long/Short Ratio
            if responses[2].status_code == 200:
                payload = responses[2].json()
                if len(payload) > 0:
                    raw_data["long_short_ratio"] = float(payload[0].get("longShortRatio", 1.0))
                    
            # Parse L2 Depth Imbalance
            if responses[3].status_code == 200:
                payload = responses[3].json()
                bids = payload.get("bids", [])
                asks = payload.get("asks", [])
                
                bid_vol = sum(float(b[1]) for b in bids)
                ask_vol = sum(float(a[1]) for a in asks)
                
                total_vol = bid_vol + ask_vol
                if total_vol > 0:
                    raw_data["bid_ask_imbalance"] = (bid_vol - ask_vol) / total_vol
                    
    except Exception as e:
        logger.warning(f"Failed to fetch derivatives data for {binance_symbol}: {e}")
    
    # Update history and compute Z-scores
    for key in raw_data:
        history[key].append(raw_data[key])
    
    # Build result with both raw and Z-scored values
    result = {}
    result["funding_rate"] = raw_data["funding_rate"]
    result["funding_rate_z"] = _z_score_latest(history["funding_rate"], raw_data["funding_rate"])
    
    result["open_interest"] = raw_data["open_interest"]
    result["open_interest_z"] = _z_score_latest(history["open_interest"], raw_data["open_interest"])
    
    result["long_short_ratio"] = raw_data["long_short_ratio"]
    result["long_short_ratio_z"] = _z_score_latest(history["long_short_ratio"], raw_data["long_short_ratio"])
    
    result["bid_ask_imbalance"] = raw_data["bid_ask_imbalance"]
    result["bid_ask_imbalance_z"] = _z_score_latest(history["bid_ask_imbalance"], raw_data["bid_ask_imbalance"])
    
    return result


async def fetch_derivatives_data(symbol: str) -> dict[str, float]:
    """
    Async wrapper for backward compatibility.
    """
    # Run sync version in thread pool to avoid blocking event loop
    return await asyncio.to_thread(fetch_derivatives_data_sync, symbol)
