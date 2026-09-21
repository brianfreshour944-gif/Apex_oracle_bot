#!/usr/bin/env python3
"""
Concurrent performance measurement - simulates the actual main loop
with concurrent symbol processing.
"""

import asyncio
import time
import os
import sys
import json
from datetime import UTC, datetime
from typing import Dict, List, Any
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import key modules
from src.config import settings
from src.exchange import AlpacaExchange
from src.strategies import TradingStrategy
from src.risk import RiskManager
from src.bot import _state, process_signal_for_symbol
from src.committee.transformer_brain import get_ml_predictor
from src.db import init_db
from src.shadow_arena import evaluate_candidates

# Timing storage
timings: Dict[str, List[float]] = defaultdict(list)


def record_timing(operation: str, duration: float, metadata: Dict = None):
    timings[operation].append(duration)


class Timer:
    def __init__(self, operation: str, metadata: Dict = None):
        self.operation = operation
        self.metadata = metadata or {}
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        duration = time.perf_counter() - self.start
        record_timing(self.operation, duration, self.metadata)


def format_stats(measurements: List[float]) -> Dict:
    if not measurements:
        return {"count": 0}
    sorted_ms = sorted([m * 1000 for m in measurements])
    n = len(sorted_ms)
    return {
        "count": n,
        "min_ms": sorted_ms[0],
        "max_ms": sorted_ms[-1],
        "mean_ms": sum(sorted_ms) / n,
        "median_ms": sorted_ms[n // 2],
        "p95_ms": sorted_ms[int(n * 0.95)] if n > 1 else sorted_ms[-1],
    }


async def run_concurrent_test():
    """Run concurrent symbol processing like the real main loop."""
    
    print("=" * 60)
    print("CONCURRENT PROCESSING TEST")
    print("=" * 60)
    
    # Initialize with mock exchange
    ex = AlpacaExchange()
    ex.trading_client = None
    ex.data_client = None
    
    class MockTradingClient:
        def get_account(self):
            from types import SimpleNamespace
            return SimpleNamespace(
                id="test", status="ACTIVE", currency="USD",
                buying_power=10000.0, equity=10000.0, portfolio_value=10000.0,
                account_blocked=False
            )
        def get_all_positions(self):
            return []
        def submit_order(self, request):
            from types import SimpleNamespace
            return SimpleNamespace(
                id="test_order_123", symbol=request.symbol, qty=request.qty,
                status=SimpleNamespace(value="filled"),
                side=request.side, type=request.type,
                filled_avg_price=50000.0, filled_qty=request.qty,
                client_order_id=request.client_order_id
            )
        def get_order_by_id(self, order_id):
            from types import SimpleNamespace
            return SimpleNamespace(
                id=order_id, symbol="BTC/USD", qty=0.1,
                filled_qty=0.1, status=SimpleNamespace(value="filled"),
                side=SimpleNamespace(value="buy"), type=SimpleNamespace(value="market"),
                filled_avg_price=50000.0, commission=0.0,
                submitted_at=datetime.now(UTC), filled_at=datetime.now(UTC),
                client_order_id="test_client_123"
            )
        def get_order_by_client_id(self, client_order_id):
            from types import SimpleNamespace
            return SimpleNamespace(
                id="test_order_123", symbol="BTC/USD", qty=0.1,
                status=SimpleNamespace(value="filled"),
                filled_avg_price=50000.0, filled_qty=0.1
            )
        def get_orders(self, request):
            return []
    
    class MockDataClient:
        def get_crypto_bars(self, request):
            from types import SimpleNamespace
            import numpy as np
            from datetime import datetime, UTC
            
            n = request.limit if request.limit else 100
            dates = [datetime.now(UTC) for _ in range(n)]
            
            symbol = request.symbol_or_symbols if isinstance(request.symbol_or_symbols, str) else request.symbol_or_symbols[0]
            base_price = {"BTC/USD": 50050.0, "ETH/USD": 3000.0, "SOL/USD": 100.0}.get(symbol, 50050.0)
            
            data = {
                "timestamp": [d.isoformat() for d in dates],
                "open": [base_price + i for i in range(n)],
                "high": [base_price * 1.002 + i for i in range(n)],
                "low": [base_price * 0.998 + i for i in range(n)],
                "close": [base_price * 1.001 + i for i in range(n)],
                "volume": [100.0] * n,
                "vwap": [base_price * 1.001 + i for i in range(n)],
                "trade_count": [100] * n,
            }
            bars_obj = SimpleNamespace(data={symbol: [
                SimpleNamespace(
                    timestamp=datetime.fromisoformat(data["timestamp"][i].replace('Z', '+00:00')),
                    open=data["open"][i], high=data["high"][i], low=data["low"][i],
                    close=data["close"][i], volume=data["volume"][i],
                    vwap=data["vwap"][i], trade_count=data["trade_count"][i]
                ) for i in range(n)
            ]})
            return bars_obj
        
        def get_crypto_latest_bar(self, request):
            from types import SimpleNamespace
            from datetime import datetime, UTC
            symbol = request.symbol_or_symbols if isinstance(request.symbol_or_symbols, str) else request.symbol_or_symbols[0]
            base_price = {"BTC/USD": 50050.0, "ETH/USD": 3000.0, "SOL/USD": 100.0}.get(symbol, 50050.0)
            return {
                symbol: SimpleNamespace(
                    timestamp=datetime.now(UTC),
                    open=base_price, high=base_price * 1.002, low=base_price * 0.998,
                    close=base_price * 1.001, volume=100.0, vwap=base_price * 1.001, trade_count=100
                )
            }
    
    ex.trading_client = MockTradingClient()
    ex.data_client = MockDataClient()
    
    strategy = TradingStrategy(ex, cache_ttl=settings.LOOP_INTERVAL_SEC * 1.5)
    risk_manager = RiskManager(ex)
    
    _state.ex = ex
    _state.strategy = strategy
    _state.risk_manager = risk_manager
    
    # Warmup model first
    print("Warming up model...")
    await asyncio.to_thread(get_ml_predictor)
    
    regime_flag = {"pause_grok": False, "pause_oracle": False, "grok_multiplier": 1.0, "oracle_multiplier": 1.0, "regime": "normal"}
    banned_symbols = set()
    
    # Simulate 3 cycles
    for cycle in range(3):
        print(f"\n--- Cycle {cycle + 1} ---")
        
        # Fetch bars concurrently (as real main loop does)
        with Timer("main_fetch_latest_bars"):
            bar_tasks = {symbol: ex.get_latest_bar(symbol) for symbol in settings.SYMBOLS}
            bar_results = await asyncio.gather(*bar_tasks.values(), return_exceptions=True)
        
        # Fetch positions
        with Timer("main_fetch_positions"):
            positions = await ex.get_positions()
        
        # Process symbols concurrently (as real main loop does)
        tasks = []
        for symbol, bar_result in zip(settings.SYMBOLS, bar_results):
            if isinstance(bar_result, Exception):
                continue
            latest_bar_df = bar_result
            if latest_bar_df.is_empty():
                continue
            
            current_price = latest_bar_df["close"][0]
            
            task = asyncio.create_task(
                process_signal_for_symbol(
                    symbol, current_price, risk_manager, strategy, ex,
                    positions=positions,
                    regime_flag=regime_flag,
                    banned_symbols=banned_symbols,
                )
            )
            tasks.append(task)
        
        # Wait for all with timing
        with Timer("main_concurrent_process_symbols"):
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flush crash recovery state
        with Timer("main_flush_crash_recovery"):
            from src.bot import flush_crash_recovery_state
            await flush_crash_recovery_state()
    
    # Now test the full cycle with fresh data each time (no cache)
    print("\n--- Fresh cache cycle ---")
    strategy._regime_cache.clear()
    
    with Timer("main_full_cycle_fresh"):
        bar_tasks = {symbol: ex.get_latest_bar(symbol) for symbol in settings.SYMBOLS}
        bar_results = await asyncio.gather(*bar_tasks.values(), return_exceptions=True)
        positions = await ex.get_positions()
        
        tasks = []
        for symbol, bar_result in zip(settings.SYMBOLS, bar_results):
            if isinstance(bar_result, Exception):
                continue
            latest_bar_df = bar_result
            if latest_bar_df.is_empty():
                continue
            current_price = latest_bar_df["close"][0]
            task = asyncio.create_task(
                process_signal_for_symbol(
                    symbol, current_price, risk_manager, strategy, ex,
                    positions=positions,
                    regime_flag=regime_flag,
                    banned_symbols=banned_symbols,
                )
            )
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)
    
    print("\nDone.")


def print_results():
    print("\n" + "=" * 60)
    print("CONCURRENT PERFORMANCE RESULTS")
    print("=" * 60)
    
    sorted_ops = sorted(
        [(op, format_stats(measurements)) for op, measurements in timings.items()],
        key=lambda x: x[1].get("mean_ms", 0),
        reverse=True
    )
    
    print("\n{:<50} {:>8} {:>10} {:>10} {:>10} {:>10}".format(
        "OPERATION", "COUNT", "MEAN(ms)", "MEDIAN", "P95", "MAX"
    ))
    print("-" * 98)
    
    for op, stats in sorted_ops:
        if stats["count"] == 0:
            continue
        print("{:<50} {:>8} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f}".format(
            op[:48], stats["count"], stats["mean_ms"], stats["median_ms"],
            stats["p95_ms"], stats["max_ms"]
        ))
    
    print("\n" + "=" * 60)
    print("BLOCKING OPERATIONS (>100ms mean or >500ms max)")
    print("=" * 60)
    
    for op, stats in sorted_ops:
        if stats["count"] == 0:
            continue
        if stats["mean_ms"] > 100 or stats["max_ms"] > 500:
            severity = "CRITICAL" if stats["max_ms"] > 1000 else "HIGH" if stats["max_ms"] > 500 else "MEDIUM"
            print(f"\n  {severity}: {op}")
            print(f"    Mean: {stats['mean_ms']:.1f}ms, Max: {stats['max_ms']:.1f}ms, Count: {stats['count']}")


async def main():
    await run_concurrent_test()
    print_results()


if __name__ == "__main__":
    asyncio.run(main())