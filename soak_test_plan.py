#!/usr/bin/env python3
"""
Memory/Soak Test Plan for Apex Oracle Bot

This document outlines the plan for verifying memory stability and 
resource usage over extended periods (24-48 hours).
"""

print("=== Memory/Soak Test Plan ===")
print()

print("OBJECTIVE:")
print("  Verify memory usage stays flat (within normal fluctuation) across")
print("  a 24-48 hour observation window post-redeploy for all bot containers.")
print()

print("BASELINE MEASUREMENTS (capture immediately after deploy):")
print("  docker stats --no-stream > baseline_$(date +%Y%m%d_%H%M).txt")
print("  Capture for each container: bot, redis, postgres (if used)")
print()

print("ONGOING MONITORING (every 1-2 hours for 24-48 hours):")
print("  docker stats --no-stream >> memory_log_$(date +%Y%m%d).txt")
print("  Include timestamp in each capture")
print()

print("KEY METRICS TO TRACK:")
print("  - Memory Usage (RSS): Should stay flat or grow very slowly")
print("  - Memory %: Should not exceed 80% of container limit")
print("  - CPU %: Should stay low (<10% avg) during idle periods")
print("  - Net I/O: Should be periodic (API calls), not continuous")
print("  - Block I/O: Should be minimal (DB writes)")
print("  - PIDs: Should stay constant (no thread leaks)")
print()

print("CANDIDATE LEAK SOURCES TO WATCH:")
print("1. _state.latest_scan_results (dict) - grows if scan results not cleaned")
print("2. _state.trade_timestamps (list) - grows if cleanup fails")
print("3. _state.cooldowns (dict) - grows if cleanup loop fails")
print("4. _state.position_adds (dict) - grows if positions not closed properly")
print("5. _state._background_tasks (set) - grows if tasks not cleaned up")
print("6. RiskManager._reserved_exposure (list) - grows if release fails")
print("7. RiskManager._realized_tx_costs (dict) - grows per symbol over time")
print("8. RiskManager.peak_prices (dict) - grows if trailing stops not cleaned")
print("9. _state._symbol_locks (dict) - grows if locks not cleaned")
print("10. Exchange._bars_cache (dict) - TTL 60s, should self-limit")
print("11. Exchange._order_cache (dict) - TTL 300s, should self-limit")
print("12. DB connection pool - check for connection leaks")
print("13. Thread pool (asyncio.to_thread) - check for thread leaks")
print("14. Background tasks (_state._background_tasks) - check for zombie tasks")
print()

print("RED FLAGS (indicates leak):")
print("  - Memory grows linearly over time (>10MB/hour sustained)")
print("  - Memory spikes and never returns to baseline")
print("  - Container OOM kills after N hours")
print("  - Connection pool exhaustion errors in logs")
print("  - 'Too many open files' errors")
print("  - Thread count continuously increasing")
print()

print("SOAK TEST PROCEDURE:")
print("1. Deploy bot to staging/production")
print("2. Immediately capture baseline: docker stats --no-stream > baseline.txt")
print("3. Set up cron job: */2 * * * * docker stats --no-stream >> /var/log/bot_memory.log")
print("4. Let run for 24-48 hours with normal trading activity")
print("5. Analyze memory_log.txt for trends:")
print("   - Plot RSS over time")
print("   - Calculate slope (MB/hour)")
print("   - Check for sawtooth pattern (GC) vs linear growth")
print("6. If leak detected:")
print("   - Identify growing data structure via heap profiling")
print("   - Check logs for error patterns")
print("   - Run targeted audit on suspected component")
print()

print("HEAP PROFILING (if leak suspected):")
print("  - Install: pip install objgraph")
print("  - Add to bot: import objgraph; objgraph.show_growth()")
print("  - Or use: python -m tracemalloc")
print()

print("EXPECTED RESULTS:")
print("- Memory should stay within 200-500MB range for bot container")
print("- RSS fluctuation < 50MB over 24h is normal")
print("- No sustained growth trend (slope ~ 0 MB/hour)")
print("- GC cycles visible as sawtooth pattern")