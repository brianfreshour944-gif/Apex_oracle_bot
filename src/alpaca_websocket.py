"""Alpaca Crypto WebSocket Client for Real-Time Bar Data.

Provides event-driven bar close notifications to replace polling-based loops.
Uses alpaca-py's WebSocket client for crypto market data.
"""

import asyncio
import datetime
import json
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set

import structlog
from alpaca.data.live.crypto import CryptoDataStream
from alpaca.data.enums import DataFeed

from src.config import settings
from src.logging_config import get_logger

logger = get_logger("alpaca_ws")


class BarCloseEvent:
    """Event emitted when a bar closes."""
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        vwap: float,
        trade_count: int,
        timestamp: str,
        is_final: bool = True,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.vwap = vwap
        self.trade_count = trade_count
        self.timestamp = timestamp
        self.is_final = is_final
        self.received_at = time.time()

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "trade_count": self.trade_count,
            "timestamp": self.timestamp,
            "is_final": self.is_final,
            "received_at": self.received_at,
        }


class AlpacaWebSocketClient:
    """
    WebSocket client for Alpaca Crypto data.
    
    Subscribes to bar streams for specified symbols and timeframes,
    and emits BarCloseEvent callbacks when bars are finalized.
    
    Usage:
        ws = AlpacaWebSocketClient()
        ws.on_bar_close(symbols=["BTC/USD", "ETH/USD"], 
                        timeframes=["1Min", "5Min", "1Hour"],
                        callback=my_handler)
        await ws.start()
    """
    
    def __init__(self):
        self.api_key = settings.ALPACA_API_KEY
        self.secret_key = settings.ALPACA_SECRET_KEY
        self.paper = "paper" in (settings.ALPACA_BASE_URL or "").lower()
        
        self._stream: Optional[CryptoDataStream] = None
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)  # timeframe -> symbols
        self._running = False
        self._start_time = 0.0
        self._bars_received = 0
        self._bars_finalized = 0
        self._errors = 0
        self._last_bar_ts: Dict[str, str] = {}  # symbol|timeframe -> timestamp
        
    def on_bar_close(
        self,
        symbols: List[str],
        timeframes: List[str],
        callback: Callable[[BarCloseEvent], Any],
    ) -> None:
        """
        Register a callback for bar close events.
        
        Args:
            symbols: List of symbols to subscribe (e.g., ["BTC/USD", "ETH/USD"])
            timeframes: List of timeframes (e.g., ["1Min", "5Min", "1Hour"])
            callback: Async function(BarCloseEvent) -> None
        """
        for tf in timeframes:
            for symbol in symbols:
                key = f"{symbol}|{tf}"
                self._callbacks[key].append(callback)
                self._subscriptions[tf].add(symbol)
                logger.info(f"Registered callback for {symbol} {tf} bars")
    
    def _parse_timeframe(self, tf: str) -> str:
        """Convert our timeframe format to Alpaca's."""
        # Alpaca expects: "1Min", "5Min", "15Min", "1Hour", "1Day"
        return tf
    
    async def start(self) -> None:
        """Start the WebSocket connection and subscriptions."""
        if self._running:
            logger.warning("WebSocket already running")
            return
            
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Alpaca credentials required for WebSocket")
        
        self._stream = CryptoDataStream(
            api_key=self.api_key,
            secret_key=self.secret_key,
            feed=DataFeed.IEX if self.paper else DataFeed.SIP,
        )
        
        # Subscribe to bars for each timeframe
        for tf, symbols in self._subscriptions.items():
            alpaca_tf = self._parse_timeframe(tf)
            for symbol in symbols:
                # Convert BTC/USD -> BTCUSD for Alpaca
                alpaca_symbol = symbol.replace("/", "")
                self._stream.subscribe_bars(
                    self._handle_bar,
                    alpaca_symbol,
                    timeframe=alpaca_tf,
                )
                logger.info(f"Subscribed to {symbol} {tf} bars")
        
        self._running = True
        self._start_time = time.time()
        
        # Run the stream in background
        self._stream_task = asyncio.create_task(self._run_stream())
        logger.info("Alpaca WebSocket started")
    
    async def _run_stream(self) -> None:
        """Run the stream with reconnection logic."""
        while self._running:
            try:
                await self._stream._run_forever()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors += 1
                logger.error(f"WebSocket error: {e}, reconnecting in 5s...")
                if not self._running:
                    break
                await asyncio.sleep(5)
                # Re-subscribe on reconnect
                if self._running:
                    await self._resubscribe()
    
    async def _resubscribe(self) -> None:
        """Re-subscribe after reconnection."""
        if not self._stream:
            return
        for tf, symbols in self._subscriptions.items():
            alpaca_tf = self._parse_timeframe(tf)
            for symbol in symbols:
                alpaca_symbol = symbol.replace("/", "")
                try:
                    self._stream.subscribe_bars(
                        self._handle_bar,
                        alpaca_symbol,
                        timeframe=alpaca_tf,
                    )
                except Exception as e:
                    logger.error(f"Resubscribe failed for {symbol} {tf}: {e}")
    
    async def _handle_bar(self, bar) -> None:
        """Handle incoming bar from Alpaca WebSocket."""
        self._bars_received += 1
        
        # Convert Alpaca bar to our format
        symbol = bar.symbol
        # Convert BTCUSD -> BTC/USD
        if len(symbol) >= 6 and symbol.endswith("USD"):
            formatted_symbol = f"{symbol[:-3]}/{symbol[-3:]}"
        else:
            formatted_symbol = symbol
        
        # Determine timeframe from the bar (Alpaca includes it in the stream)
        # For now, we need to track which timeframe this bar belongs to
        # Alpaca sends separate streams per timeframe, so we infer from subscription
        timeframe = "1Min"  # Default, will be overridden by subscription tracking
        
        # Try to find matching subscription
        for tf, symbols in self._subscriptions.items():
            if formatted_symbol in symbols:
                timeframe = tf
                break
        
        # Check if this is a new bar (timestamp changed)
        key = f"{formatted_symbol}|{timeframe}"
        bar_ts = bar.timestamp.isoformat() if hasattr(bar.timestamp, 'isoformat') else str(bar.timestamp)
        
        is_new_bar = self._last_bar_ts.get(key) != bar_ts
        if is_new_bar:
            self._last_bar_ts[key] = bar_ts
            self._bars_finalized += 1
            
            event = BarCloseEvent(
                symbol=formatted_symbol,
                timeframe=timeframe,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
                vwap=float(bar.vwap) if hasattr(bar, 'vwap') and bar.vwap else float(bar.close),
                trade_count=int(bar.trade_count) if hasattr(bar, 'trade_count') and bar.trade_count else 0,
                timestamp=bar_ts,
                is_final=True,
            )
            
            # Fire callbacks
            callbacks = self._callbacks.get(key, [])
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(event))
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Callback error for {key}: {e}")
    
    async def stop(self) -> None:
        """Stop the WebSocket connection."""
        self._running = False
        if hasattr(self, '_stream_task'):
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        if self._stream:
            await self._stream.stop_ws()
        logger.info(f"WebSocket stopped. Bars received: {self._bars_received}, finalized: {self._bars_finalized}, errors: {self._errors}")
    
    def get_stats(self) -> dict:
        """Get connection statistics."""
        return {
            "running": self._running,
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "bars_received": self._bars_received,
            "bars_finalized": self._bars_finalized,
            "errors": self._errors,
            "subscriptions": {tf: list(syms) for tf, syms in self._subscriptions.items()},
        }


# Global instance for easy access
_ws_client: Optional[AlpacaWebSocketClient] = None


def get_ws_client() -> AlpacaWebSocketClient:
    """Get the global WebSocket client instance."""
    global _ws_client
    if _ws_client is None:
        _ws_client = AlpacaWebSocketClient()
    return _ws_client


async def start_ws_client(symbols: List[str], timeframes: List[str]) -> AlpacaWebSocketClient:
    """Start the global WebSocket client with subscriptions."""
    client = get_ws_client()
    client.on_bar_close(symbols, timeframes, lambda e: None)  # Placeholder, real callbacks registered by bot
    await client.start()
    return client