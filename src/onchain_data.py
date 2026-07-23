"""Derivatives & On-Chain Data Fetcher.

Pulls macro crypto data (Funding Rates, Open Interest, Long/Short Ratios)
from public Binance Futures endpoints to augment technical indicators.
"""

import httpx
import asyncio
from typing import Dict, Any

from src.logging_config import get_logger

logger = get_logger("onchain")

async def fetch_derivatives_data(symbol: str) -> Dict[str, float]:
    """
    Fetches Open Interest, Funding Rate, and Global Long/Short ratio from Binance Futures.
    Alpaca symbols are usually 'BTC/USD', so we convert to 'BTCUSDT' for Binance.
    """
    base_asset = symbol.split("/")[0] if "/" in symbol else symbol.replace("USD", "")
    binance_symbol = f"{base_asset}USDT"
    
    data = {
        "funding_rate": 0.0,
        "open_interest": 0.0,
        "long_short_ratio": 1.0,
        "bid_ask_imbalance": 0.0
    }
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # 1. Funding Rate (Premium Index)
            funding_url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={binance_symbol}"
            
            # 2. Open Interest
            oi_url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={binance_symbol}"
            
            # 3. Global Long/Short Ratio (top traders, 5m timeframe)
            ls_url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={binance_symbol}&period=5m&limit=1"
            # 4. L2 Order Book Depth
            depth_url = f"https://fapi.binance.com/fapi/v1/depth?symbol={binance_symbol}&limit=50"
            
            results = await asyncio.gather(
                client.get(funding_url),
                client.get(oi_url),
                client.get(ls_url),
                client.get(depth_url),
                return_exceptions=True
            )
            
            # Parse Funding Rate
            if not isinstance(results[0], Exception) and results[0].status_code == 200:
                payload = results[0].json()
                data["funding_rate"] = float(payload.get("lastFundingRate", 0.0))
                
            # Parse Open Interest
            if not isinstance(results[1], Exception) and results[1].status_code == 200:
                payload = results[1].json()
                data["open_interest"] = float(payload.get("openInterest", 0.0))
                
            # Parse Long/Short Ratio
            if not isinstance(results[2], Exception) and results[2].status_code == 200:
                payload = results[2].json()
                if len(payload) > 0:
                    data["long_short_ratio"] = float(payload[0].get("longShortRatio", 1.0))
                    
            # Parse L2 Depth Imbalance
            if not isinstance(results[3], Exception) and results[3].status_code == 200:
                payload = results[3].json()
                bids = payload.get("bids", [])
                asks = payload.get("asks", [])
                
                bid_vol = sum(float(b[1]) for b in bids)
                ask_vol = sum(float(a[1]) for a in asks)
                
                total_vol = bid_vol + ask_vol
                if total_vol > 0:
                    data["bid_ask_imbalance"] = (bid_vol - ask_vol) / total_vol
                    
    except Exception as e:
        logger.warning(f"Failed to fetch derivatives data for {binance_symbol}: {e}")
        
    return data
