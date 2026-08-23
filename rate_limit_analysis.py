print("=== API Rate-Limit Analysis ===")
print()

# Alpaca Paper Trading Rate Limits (from docs):
# - Market Data (crypto): 200 requests/minute per endpoint
# - Trading (orders/positions/account): 200 requests/minute
# - WebSocket: 50 connections, 100 messages/second

# Bot's API usage per cycle (LOOP_INTERVAL_SEC = 60s default):

# Per symbol per cycle:
# 1. get_latest_bar() - 1 request (cached for 60s, but called each cycle)
# 2. get_positions() - 1 request (fetched ONCE per cycle in main loop, line 1681)
# 3. get_account() - 1 request (in update_account_status, but positions passed in)

# Per symbol for BUY path:
# 4. check_and_reserve_exposure -> update_account_status (if current_exposure not provided)
#    But in bot.py line 686-687, current_exposure IS provided from risk_status
#    So NO additional get_account/get_positions calls for buy path!

# Order placement:
# 5. create_order() - 1 request per trade
# 6. get_order() polling - up to 20 requests (10s timeout / 0.5s interval) per order

# Concurrent symbols: 3 (BTC/USD, ETH/USD, SOL/USD)

# REQUESTS PER MINUTE (steady state, no trades):
# - get_latest_bar: 3 symbols * 1 = 3 req/min
# 2. get_positions: 1 request (shared across symbols)
# Total: ~4 req/min (well under 200 limit)

# REQUESTS PER MINUTE (during active trading, 3 orders):
# - create_order: 3 req/min
# - get_order polling: 3 orders * 20 polls = 60 req/min
# - get_latest_bar: 3 req/min
# - get_positions: 1 req/min
# Total: ~67 req/min (well under 200 limit)

print("=== Alpaca Rate Limit Analysis ===")
print()
print("Alpaca Paper Trading Limits:")
print("  - Market Data (crypto): 200 req/min per endpoint")
print("  - Trading (orders/positions/account): 200 req/min")
print("  - WebSocket: 50 connections, 100 msg/sec")
print()

# Steady state
steady_state = 3 + 1  # 3 bars + 1 positions
print("Steady state (no trades): " + str(steady_state) + " req/min")
print("  get_latest_bar: 3 req/min (3 symbols)")
print("  get_positions: 1 req/min (shared)")
print()

# Active trading
active = 3 + 60 + 3 + 1  # 3 orders + 60 polls + 3 bars + 1 positions
print("Active trading (3 orders): " + str(active) + " req/min")
print("  create_order: 3 req/min")
print("  get_order polling: 3 * 20 = 60 req/min")
print("  get_latest_bar: 3 req/min")
print("  get_positions: 1 req/min")
print()

print("Both well under 200 req/min limit!")
print()

# Burst analysis
print("BURST ANALYSIS:")
print("  All 3 symbols evaluated CONCURRENTLY via asyncio.gather (line 1660)")
print("  3 get_latest_bar calls fire simultaneously")
print("  Then 3 process_signal_for_symbol tasks created CONCURRENTLY")
print("  Each may call create_order concurrently")
print("  Burst: up to 3 create_order + 60 get_order calls in ~10 seconds")
print("  = 63 requests in 10 seconds = 378 req/min equivalent burst")
print("  BUT: Alpaca limits are per-minute rolling window")
print("  Risk: If cycle runs at 60s intervals, burst at t=0, next at t=60s")
print("  Could hit limit if previous minute's requests still count")
print()

print("RATE LIMIT HANDLING:")
print("  exchange.py has retry logic with exponential backoff (lines 37-69)")
print("  Respects Retry-After and RateLimit-Reset headers (lines 43-69)")
print("  Max 5 retries with exponential backoff (min=4s, max=30s)")
print("  CircuitBreaker in exchange.py (line 100) for repeated failures")
print()
print("POTENTIAL ISSUES:")
print("  1. No proactive rate limiting - relies on reactive retries")
print("  2. Concurrent bursts could trigger 429s before backoff kicks in")
print("  3. No request batching (e.g., get_positions called once but could be batched)")
print("  4. get_bars called per-symbol - could batch multi-symbol requests")
print("  5. No monitoring of rate limit headers to proactively throttle")