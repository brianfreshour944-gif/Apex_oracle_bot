"""Derivatives & On-Chain Data Fetcher.

Pulls macro crypto data (Funding Rates, Open Interest, Long/Short Ratios)
from public Binance Futures endpoints to augment technical indicators.

Adds Z-scored versions for regime-invariant comparison across time.
"""

import asyncio
import time
from collections import deque
from typing import Optional

import httpx
import numpy as np

from src.logging_config import get_logger

logger = get_logger("onchain")

# Rolling windows for Z-scoring (per symbol)
_DERIVATIVES_HISTORY: dict[str, dict[str, deque]] = {}
_HISTORY_MAXLEN = 200  # ~200 5-min intervals = ~16 hours

# Shared async client for connection pooling
_derivatives_client: Optional[httpx.AsyncClient] = None


async def _get_client() -> httpx.AsyncClient:
    """Get or create shared async HTTP client with connection pooling."""
    global _derivatives_client
    if _derivatives_client is None or _derivatives_client.is_closed:
        _derivatives_client = httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=2.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _derivatives_client


async def close_derivatives_client() -> None:
    """Close the shared HTTP client."""
    global _derivatives_client
    if _derivatives_client is not None and not _derivatives_client.is_closed:
        await _derivatives_client.aclose()
        _derivatives_client = None


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


async def _fetch_single_symbol(client: httpx.AsyncClient, symbol: str, binance_symbol: str, history: dict[str, deque]) -> dict[str, float]:
    """Fetch derivatives data for a single symbol."""
    raw_data = {
        "funding_rate": 0.0,
        "open_interest": 0.0,
        "long_short_ratio": 1.0,
        "bid_ask_imbalance": 0.0
    }
    
    try:
        # 1. Funding Rate (Premium Index)
        funding_url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={binance_symbol}"
        
        # 2. Open Interest
        oi_url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={binance_symbol}"
        
        # 3. Global Long/Short Ratio (top traders, 5m timeframe)
        ls_url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={binance_symbol}&period=5m&limit=1"
        
        # 4. L2 Order Book Depth
        depth_url = f"https://fapi.binance.com/fapi/v1/depth?symbol={binance_symbol}&limit=50"
        
        # Fetch all 4 endpoints concurrently
        responses = await asyncio.gather(
            client.get(funding_url),
            client.get(oi_url),
            client.get(ls_url),
            client.get(depth_url),
            return_exceptions=True
        )
        
        # Parse Funding Rate
        if not isinstance(responses[0], Exception) and responses[0].status_code == 200:
            payload = responses[0].json()
            raw_data["funding_rate"] = float(payload.get("lastFundingRate", 0.0))
            
        # Parse Open Interest
        if not isinstance(responses[1], Exception) and responses[1].status_code == 200:
            payload = responses[1].json()
            raw_data["open_interest"] = float(payload.get("openInterest", 0.0))
            
        # Parse Long/Short Ratio
        if not isinstance(responses[2], Exception) and responses[2].status_code == 200:
            payload = responses[2].json()
            if len(payload) > 0:
                raw_data["long_short_ratio"] = float(payload[0].get("longShortRatio", 1.0))
                
        # Parse L2 Depth Imbalance
        if not isinstance(responses[3], Exception) and responses[3].status_code == 200:
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
    Async version: Fetches Open Interest, Funding Rate, and Long/Short ratio from Binance Futures.
    Alpaca symbols are usually 'BTC/USD', so we convert to 'BTCUSDT' for Binance.
    
    Returns both raw and Z-scored values for regime-invariant analysis.
    """
    base_asset = symbol.split("/")[0] if "/" in symbol else symbol.replace("USD", "")
    binance_symbol = f"{base_asset}USDT"
    
    # Get rolling history for Z-scoring
    history = _get_symbol_history(symbol)
    
    client = await _get_client()
    return await _fetch_single_symbol(client, symbol, binance_symbol, history)


async def fetch_derivatives_batch(symbols: list[str]) -> dict[str, dict[str, float]]:
    """
    Fetch derivatives data for multiple symbols concurrently.
    
    Args:
        symbols: List of Alpaca-style symbols (e.g., ["BTC/USD", "ETH/USD"])
        
    Returns:
        Dict mapping symbol -> derivatives data dict
    """
    if not symbols:
        return {}
    
    client = await _get_client()
    
    async def _fetch_one(symbol: str) -> tuple[str, dict[str, float]]:
        base_asset = symbol.split("/")[0] if "/" in symbol else symbol.replace("USD", "")
        binance_symbol = f"{base_asset}USDT"
        history = _get_symbol_history(symbol)
        data = await _fetch_single_symbol(client, symbol, binance_symbol, history)
        return symbol, data
    
    results = await asyncio.gather(*[_fetch_one(s) for s in symbols], return_exceptions=True)
    
    output = {}
    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Batch fetch error: {result}")
        else:
            symbol, data = result
            output[symbol] = data
    
    return output


# Backward compatibility: synchronous wrapper for any legacy callers
def fetch_derivatives_data_sync(symbol: str) -> dict[str, float]:
    """
    Synchronous wrapper for backward compatibility.
    Runs the async version in the current event loop if running, otherwise creates a new one.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop is not None and loop.is_running():
        # We're in an async context - can't use asyncio.run()
        # This is a legacy sync caller in an async context - should use async version instead
        logger.warning("fetch_derivatives_data_sync called from async context; use fetch_derivatives_data() instead")
        # Create a task and run it synchronously (not ideal but maintains compatibility)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, fetch_derivatives_data(symbol))
            return future.result()
    else:
        return asyncio.run(fetch_derivatives_data(symbol))
