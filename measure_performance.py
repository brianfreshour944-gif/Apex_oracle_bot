#!/usr/bin/env python3
"""
Performance measurement script for the trading bot.
Measures actual blocking durations in a trading cycle using timers.
"""

import asyncio
import time
import os
import sys
import json
from datetime import UTC, datetime
from typing import Dict, List, Any
from contextlib import asynccontextmanager
from collections import defaultdict
from types import SimpleNamespace
import polars as pl
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import key modules
from src.config import settings
from src.exchange import AlpacaExchange
from src.strategies import TradingStrategy
from src.risk import RiskManager
from src.bot import _state, process_signal_for_symbol
from src.committee.transformer_brain import get_ml_predictor, _model_inference_lock
from src.onchain_data import fetch_derivatives_data
from src.db import (
    init_db, get_open_snapshot, close_decision_snapshot,
    save_decision_snapshot, update_decision_snapshot_position,
    get_all_open_snapshots, get_closed_decision_snapshots
)
from src.feature_engineering import add_multi_timeframe_features
from src.onchain_data import fetch_derivatives_data_sync
from src.sentiment_analyzer import extract_sentiment
from src.shadow_arena import evaluate_candidates

# Timing storage
timings: Dict[str, List[float]] = defaultdict(list)
blocking_events: List[Dict[str, Any]] = []

# Track what's running
measurement_active = True


def record_timing(operation: str, duration: float, metadata: Dict = None):
    """Record a timing measurement."""
    timings[operation].append(duration)
    blocking_events.append({
        "operation": operation,
        "duration_ms": duration * 1000,
        "timestamp": time.time(),
        "metadata": metadata or {}
    })


def format_stats(measurements: List[float]) -> Dict:
    """Format timing statistics."""
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
        "p95_ms": sorted_ms[int(n * 0.95)],
        "p99_ms": sorted_ms[int(n * 0.99)] if n > 1 else sorted_ms[-1],
    }


class Timer:
    """Context manager for timing operations."""
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


async def measure_async(operation: str, coro, metadata: Dict = None):
    """Measure an async operation."""
    start = time.perf_counter()
    try:
        result = await coro
        return result
    finally:
        duration = time.perf_counter() - start
        record_timing(operation, duration, metadata)


async def measure_to_thread(operation: str, func, *args, metadata: Dict = None, **kwargs):
    """Measure asyncio.to_thread operation."""
    start = time.perf_counter()
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    finally:
        duration = time.perf_counter() - start
        record_timing(operation, duration, metadata)


async def run_measured_cycle():
    """Run one full measured trading cycle."""
    global measurement_active
    
    print("=" * 60)
    print("STARTING PERFORMANCE MEASUREMENT")
    print("=" * 60)
    
    # Initialize components
    print("\n[1] Initializing components...")
    
    with Timer("exchange_init"):
        ex = AlpacaExchange()
        # Don't actually load - we'll mock the data fetching
        ex.trading_client = None
        ex.data_client = None
    
    # Create mock objects for exchange methods
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
            import polars as pl
            import numpy as np
            from datetime import datetime, UTC
            
            # Create mock bars data
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
    
    with Timer("strategy_init"):
        strategy = TradingStrategy(ex, cache_ttl=settings.LOOP_INTERVAL_SEC * 1.5)
    
    with Timer("risk_manager_init"):
        risk_manager = RiskManager(ex)
    
    _state.ex = ex
    _state.strategy = strategy
    _state.risk_manager = risk_manager
    
    # Warm up model
    print("\n[2] Warming up ML model...")
    with Timer("model_warmup"):
        await asyncio.to_thread(get_ml_predictor)
    
    print("\n[3] Running measured trading cycle...")
    
    # Get regime flag and banned symbols
    regime_flag = {"pause_grok": False, "pause_oracle": False, "grok_multiplier": 1.0, "oracle_multiplier": 1.0, "regime": "normal"}
    banned_symbols = set()
    
    # Fetch all latest bars concurrently
    with Timer("fetch_latest_bars_all_symbols"):
        bar_tasks = {symbol: ex.get_latest_bar(symbol) for symbol in settings.SYMBOLS}
        bar_results = await asyncio.gather(*bar_tasks.values(), return_exceptions=True)
    
    # Fetch positions once
    with Timer("fetch_positions"):
        positions = await ex.get_positions()
    
    # Process each symbol
    for symbol, bar_result in zip(settings.SYMBOLS, bar_results):
        if isinstance(bar_result, Exception):
            print(f"  Error fetching bar for {symbol}: {bar_result}")
            continue
        
        latest_bar_df = bar_result
        if latest_bar_df.is_empty():
            print(f"  No data for {symbol}")
            continue
        
        current_price = latest_bar_df["close"][0]
        print(f"\n  Processing {symbol} @ ${current_price:.2f}")
        
        # Measure regime analysis
        with Timer("analyze_market_regime", {"symbol": symbol}):
            regime_data = await strategy.analyze_market_regime(symbol)
        
        # Measure feature engineering (this is called inside analyze_market_regime)
        # But let's also measure multi-timeframe features separately
        with Timer("add_multi_timeframe_features", {"symbol": symbol}):
            from src.config import FEATURE_BASE_TIMEFRAME, TIMEFRAMES
            bars_df_raw = await ex.get_bars(symbol, "1Day", 100)
            await add_multi_timeframe_features(
                ex, symbol,
                base_timeframe=FEATURE_BASE_TIMEFRAME,
                timeframes=TIMEFRAMES,
                limit=100,
                bars_df=bars_df_raw
            )
        
        # Measure onchain data fetch
        with Timer("fetch_derivatives_data", {"symbol": symbol}):
            deriv_data = await fetch_derivatives_data(symbol)
        
        # Measure sentiment fetch
        with Timer("extract_sentiment", {"symbol": symbol}):
            await extract_sentiment(symbol)
        
        # Measure signal generation
        with Timer("generate_trading_signal", {"symbol": symbol}):
            signal = await strategy.generate_trading_signal(symbol, current_price, None)
        
        # Measure committee (only if signal is buy/sell)
        if signal.get("action") in ["buy", "sell"]:
            from src.committee.committee import run_committee
            with Timer("run_committee", {"symbol": symbol}):
                committee_result = await run_committee(symbol, current_price, signal)
            
            # Measure transformer inference specifically
            from src.committee.transformer_brain import transformer_brain
            with Timer("transformer_brain_inference", {"symbol": symbol}):
                await transformer_brain(symbol, current_price, signal)
        
        # Measure full process_signal_for_symbol
        with Timer("process_signal_for_symbol_full", {"symbol": symbol}):
            await process_signal_for_symbol(
                symbol, current_price, risk_manager, strategy, ex,
                positions=positions,
                regime_flag=regime_flag,
                banned_symbols=banned_symbols,
            )
        
        # Measure DB operations
        test_decision_id = f"perf_test_{symbol}_{int(time.time())}"
        
        with Timer("save_decision_snapshot", {"symbol": symbol}):
            await measure_to_thread(
                "save_decision_snapshot_thread",
                save_decision_snapshot,
                decision_id=test_decision_id,
                symbol=symbol,
                regime="test",
                final_action="buy",
                confidence=0.7,
                size_multiplier=1.0,
                entry_price=current_price,
                qty=0.1,
                brain_votes={"transformer": "buy", "quant": "buy"},
                feature_snapshot_json="{}",
                causal_reasoning_json="{}",
                tensor_state_json="{}"
            )
        
        with Timer("get_open_snapshot", {"symbol": symbol}):
            await measure_to_thread("get_open_snapshot_thread", get_open_snapshot, symbol)
        
        with Timer("close_decision_snapshot", {"symbol": symbol}):
            await measure_to_thread(
                "close_decision_snapshot_thread",
                close_decision_snapshot,
                test_decision_id,
                realized_pnl=10.0,
                return_pct=1.0,
                holding_period_sec=3600,
                exit_reason="test"
            )
        
        with Timer("update_decision_snapshot_position", {"symbol": symbol}):
            await measure_to_thread(
                "update_decision_snapshot_position_thread",
                update_decision_snapshot_position,
                test_decision_id,
                entry_price=current_price * 1.01,
                qty=0.15
            )
    
    # Measure get_all_open_snapshots
    with Timer("get_all_open_snapshots"):
        await measure_to_thread("get_all_open_snapshots_thread", get_all_open_snapshots)
    
    # Measure get_closed_decision_snapshots
    with Timer("get_closed_decision_snapshots"):
        await measure_to_thread("get_closed_decision_snapshots_thread", get_closed_decision_snapshots, limit=100)
    
    # Measure account/position refresh
    with Timer("get_account"):
        await ex.get_account()
    
    with Timer("get_positions"):
        await ex.get_positions()
    
# Measure shadow arena
        with Timer("evaluate_candidates_shadow_arena", {"symbol": "BTC/USD"}):
            evaluate_candidates("BTC/USD", 50000.0, None)
    
    # Measure risk manager operations
    with Timer("update_account_status"):
        await risk_manager.update_account_status(positions=positions)
    
    with Timer("calculate_position_size"):
        risk_manager.calculate_position_size(
            "BTC/USD", 50000.0, "trending", atr=1000.0,
            confidence=0.8, expected_return_pct=0.03,
            current_equity=10000.0, drawdown_pct=0.0, side="buy"
        )
    
    with Timer("check_trailing_stop"):
        risk_manager.check_trailing_stop("BTC/USD", 51000.0, 50000.0, 0.1, regime="trending")
    
    with Timer("check_and_reserve_exposure"):
        await risk_manager.check_and_reserve_exposure(5000.0, current_exposure=1000.0)
    
    with Timer("reserve_position_slot"):
        await risk_manager.reserve_position_slot("BTC/USD", 1, 0, positions)
    
    # Measure alerting
    from src.alerting import get_alerting_engine
    alerting = get_alerting_engine()
    
    with Timer("alerting_run_monitoring_cycle"):
        await alerting.run_monitoring_cycle()
    
    print("\n[4] Cycle complete. Collecting results...")
    
    measurement_active = False


def print_results():
    """Print formatted timing results."""
    print("\n" + "=" * 60)
    print("PERFORMANCE MEASUREMENT RESULTS")
    print("=" * 60)
    
    # Sort operations by mean duration
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
    
    # Identify blocking operations (>100ms)
    print("\n" + "=" * 60)
    print("BLOCKING OPERATIONS (>100ms mean or >500ms max)")
    print("=" * 60)
    
    blocking_found = False
    for op, stats in sorted_ops:
        if stats["count"] == 0:
            continue
        if stats["mean_ms"] > 100 or stats["max_ms"] > 500:
            blocking_found = True
            severity = "CRITICAL" if stats["max_ms"] > 1000 else "HIGH" if stats["max_ms"] > 500 else "MEDIUM"
            print(f"\n  {severity}: {op}")
            print(f"    Mean: {stats['mean_ms']:.1f}ms, Max: {stats['max_ms']:.1f}ms, Count: {stats['count']}")
    
    if not blocking_found:
        print("  No significantly blocking operations detected.")
    
    # Check for specific issues
    print("\n" + "=" * 60)
    print("SPECIFIC AUDIT CHECKS")
    print("=" * 60)
    
    # 1. Check for synchronous calls that should be async
    sync_blocking = []
    for op, stats in sorted_ops:
        if stats["mean_ms"] > 50 and "thread" not in op.lower() and "to_thread" not in op.lower():
            sync_blocking.append((op, stats))
    
    if sync_blocking:
        print("\n1. SYNCHRONOUS/BLOCKING CALLS (should be async or to_thread):")
        for op, stats in sync_blocking:
            print(f"   {op}: mean={stats['mean_ms']:.1f}ms, max={stats['max_ms']:.1f}ms")
    else:
        print("\n1. SYNCHRONOUS/BLOCKING CALLS: None detected above 50ms")
    
    # 2. Check for redundant API calls
    api_calls = [(op, stats) for op, stats in sorted_ops if "fetch" in op.lower() or "get_" in op.lower()]
    print("\n2. API CALL PATTERNS:")
    for op, stats in api_calls:
        print(f"   {op}: count={stats['count']}, mean={stats['mean_ms']:.1f}ms")
    
    # 3. DB operations
    db_ops = [(op, stats) for op, stats in sorted_ops if "snapshot" in op.lower() or "db" in op.lower() or "decision" in op.lower()]
    print("\n3. DATABASE OPERATIONS:")
    for op, stats in db_ops:
        print(f"   {op}: count={stats['count']}, mean={stats['mean_ms']:.1f}ms, max={stats['max_ms']:.1f}ms")
    
    # 4. Model inference
    model_ops = [(op, stats) for op, stats in sorted_ops if "model" in op.lower() or "transformer" in op.lower() or "committee" in op.lower() or "inference" in op.lower()]
    print("\n4. MODEL INFERENCE:")
    for op, stats in model_ops:
        print(f"   {op}: count={stats['count']}, mean={stats['mean_ms']:.1f}ms, max={stats['max_ms']:.1f}ms")
    
    # 5. Log/alert I/O
    log_ops = [(op, stats) for op, stats in sorted_ops if "alert" in op.lower() or "telegram" in op.lower() or "log" in op.lower()]
    print("\n5. ALERT/LOGGING I/O:")
    for op, stats in log_ops:
        print(f"   {op}: count={stats['count']}, mean={stats['mean_ms']:.1f}ms, max={stats['max_ms']:.1f}ms")
    
    # Save raw data
    output = {
        "timings": {op: format_stats(ms) for op, ms in timings.items()},
        "blocking_events": blocking_events,
        "summary": {
            "total_operations": sum(len(ms) for ms in timings.values()),
            "unique_operations": len(timings),
            "timestamp": datetime.now(UTC).isoformat()
        }
    }
    
    with open("performance_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nFull results saved to performance_results.json")


async def main():
    """Main entry point."""
    try:
        await run_measured_cycle()
    except Exception as e:
        print(f"Error during measurement: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print_results()
        # Cleanup
        if _state.ex:
            await _state.ex.close()


if __name__ == "__main__":
    asyncio.run(main())