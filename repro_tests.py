"""
Reproduction tests for identified bugs.
Run each independently to see actual output.
"""
import asyncio
import sys

# ---- BUG 1 REPRO: bot.py line 281-295 - 'dashboard' used before assignment ----
# If signal["action"] == "buy" AND committee was NOT run (action was "close"),
# then 'dashboard' and 'committee_result' are unbound when the buy block executes.
# This happens when signal["action"] == "close" first passes committee bypass (line 212),
# but then ALSO hits signal["action"] in ["buy", "sell"] at line 276.
# Actually: if action == "close", line 212 passes directly, and line 276 is NOT hit.
# So that specific path is fine. BUT: if action == "buy", dashboard is built at line 221.
# Let's verify: at line 281 `dashboard.append(...)` - is dashboard always defined?
# The buy/sell path ALWAYS goes through committee (line 217 blocks close), so dashboard 
# is always built before line 281. HOWEVER: there is still a subtle scoping issue -
# if action changes to "buy" via committee_result, regime_flag is used at line 340 -
# but regime_flag is only set at line 279 (inside the "if signal['action'] == 'buy':' block).
# Line 340: `multiplier = regime_flag.get(...)` - regime_flag could be unbound if action was "sell" initially,
# then committee changed it to "buy". No wait, at line 273: signal["action"] = committee_result.action.
# So if committee says buy, we enter line 278-286 (regime_flag check only for buy).
# Then line 339: `if signal["action"] == "buy":` - yes, regime_flag assigned at line 279 so it's fine.
# BUT if signal["action"] was "sell" and committee returned "buy", 
# then line 278: `if signal["action"] == "buy":` -- at this point signal["action"] 
# has already been overwritten at line 273! So if committee returns buy, regime_flag IS set.
# Actually the logic seems fine here.

print("BUG 1: Checking unbound variable scenario in process_signal_for_symbol")

# ---- BUG 2 REPRO: attribution.py - model.generate_content is called WITHOUT await ----
# Line 114 in attribution.py: `response = model.generate_content(prompt)` 
# google.generativeai's generate_content is SYNCHRONOUS, not async.
# This is INSIDE an async function but called without await - that's correct for sync.
# But the whole function is called with await - that's fine.
# HOWEVER: the function does blocking I/O without asyncio.to_thread(), 
# which blocks the event loop.

print("\nBUG 2: Gemini generate_content in attribution.py")
print("  attribution.py:114 - model.generate_content() is a BLOCKING call in an async function")
print("  This blocks the entire event loop during LLM inference")
print("  STATUS: Read-only finding (not reproduced, but architecturally certain)")

# ---- BUG 3 REPRO: transformer_brain.py lines 245-248 ----
# Inside `_do_inference()` (a SYNC function called via asyncio.to_thread):
# It creates a NEW event loop and runs `fetch_derivatives_data` in it.
# This is fine for async usage from a thread. BUT: 
# The issue is it calls `asyncio.new_event_loop()` / `asyncio.set_event_loop()` 
# in a thread that has no event loop, while the MAIN loop is running.
# This can cause issues on some platforms. Let's verify the actual behavior.

print("\nBUG 3: New event loop in asyncio.to_thread")
print("  transformer_brain.py:245-248 creates a new event loop inside a thread")

async def reproduce_bug3():
    """Test that creating a new event loop inside asyncio.to_thread works."""
    import asyncio
    
    async def dummy_coro():
        return {"funding_rate": 0.1}
    
    def sync_func_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(dummy_coro())
            return result
        finally:
            loop.close()
    
    result = await asyncio.to_thread(sync_func_in_thread)
    print(f"  Result from thread with new event loop: {result}")
    print("  OK: This pattern works correctly")

asyncio.run(reproduce_bug3())

# ---- BUG 4 REPRO: bot.py - global 'strategy' and 'risk_manager' scoping ----
# In run_trading_bot(), line 722-723:
#   strategy = TradingStrategy(ex)
#   risk_manager = RiskManager(ex)
# These are LOCAL variables in run_trading_bot(). They shadow the module-level globals.
# But in _record_committee_outcome() at line 60-66:
#   global strategy
#   if strategy is not None and hasattr(strategy, '_trailing_peaks') ...
# The `global strategy` at line 60 refers to the MODULE-LEVEL `strategy` (initialized to None at line 138).
# The LOCAL `strategy` in run_trading_bot() is a different variable.
# The module-level `strategy` is NEVER updated to the actual TradingStrategy instance!
# So `strategy` in _record_committee_outcome() is ALWAYS None, and _trailing_peaks is never accessed.

print("\nBUG 4: Global 'strategy' never assigned (module-level)")
print("  bot.py:722 - `strategy = TradingStrategy(ex)` creates a LOCAL variable")
print("  bot.py:138 - module-level `strategy: Optional[TradingStrategy] = None` is never updated")
print("  bot.py:60  - `global strategy` in _record_committee_outcome() reads the module-level None")
print("  Result: max_fav_pct/max_adv_pct are always 0.0 (silently wrong, non-fatal)")

# Let's reproduce:
strategy_module_level = None  # module global

def run_bot_simulation():
    """Simulates run_trading_bot's variable scoping."""
    strategy = "real_strategy_object"  # this is LOCAL, NOT the module global
    # strategy_module_level is unchanged
    
def record_outcome_simulation():
    global strategy_module_level
    if strategy_module_level is not None:
        print("  WOULD ACCESS _trailing_peaks")
    else:
        print(f"  strategy_module_level is None -> max_fav_pct always 0.0 (confirmed bug)")

run_bot_simulation()
record_outcome_simulation()

# ---- BUG 5 REPRO: risk.py - daily_pnl is negative by default at startup ----
# start_of_day_equity is set to equity on first call (line 77).
# But daily_pnl is checked at line 82: `if self.daily_pnl < daily_loss_limit_abs`
# On the very first call: daily_pnl = equity - start_of_day_equity = 0.0 (correct).
# But the day-boundary detection at line 60: `if now.day != self.last_check_time.day`
# On first call, last_check_time is set in __init__ to now, so .day matches.
# OK, day boundary logic seems fine on first call.
# BUT: there's an ordering issue - line 62 sets `self.start_of_day_equity = equity`
# AND line 73 sets `self.last_check_time = now` - both happen on day rollover.
# Then line 76-78 runs UNCONDITIONALLY: `if not hasattr(self, 'start_of_day_equity'):` 
# This branch runs once ever (first call), setting baseline.
# On subsequent calls: daily_pnl = equity - start_of_day_equity (set at first call).
# On day rollover: start_of_day_equity is RESET (good - the comment says this was a bug fix).
# This seems correct. The fix comment in the code at lines 62-72 says this was already fixed.

print("\nBUG 5: risk.py daily_pnl baseline - Already fixed per code comments")

# ---- BUG 6 REPRO: strategy_selector.py - record_strategy_outcome always uses "buy" ----
# Lines 92-100: regardless of actual action (buy or sell), mock_votes[strategy] = "buy"
# and final_action = "buy". So if the bot made a "sell" trade and lost money,
# the learner is told "buy vote" + negative PnL -> penalizes the sell strategy incorrectly.
# This means the strategy learner NEVER properly learns from sell trades.

print("\nBUG 6: strategy_selector.py - record_strategy_outcome always uses 'buy'")
print("  strategy_selector.py:92-100 - regardless of actual side, always mocks as 'buy'")
print("  A losing 'sell' trade is recorded as: buy-voted-wrong -> penalizes sell strategy for selling")
print("  A winning 'sell' trade is recorded as: buy-voted-right -> rewards sell strategy for buying")  
print("  This makes the strategy learner miscalibrated for sell-side trades")
print("  STATUS: Read-only finding")

# ---- BUG 7 REPRO: NaN comparison in risk.py / strategies.py ----
# In strategies.py _check_price_based_exits line 396:
# if pnl_pct >= settings.PROFIT_TARGET_PCT * 100:
# pnl_pct is a float from arithmetic. If entry_price is 0 (guard is at line 386-387),
# so NaN shouldn't appear here normally. Good.
# BUT: in risk.py calculate_position_size line 165:
# conf_weight = np.clip(confidence, 0.5, 1.5)
# If confidence is NaN, np.clip(NaN, 0.5, 1.5) returns NaN! -> position_size becomes NaN.
# Then line 191: position_size = min(position_size, ...) - min(NaN, x) = NaN in Python!
# Then line 192: round(NaN, 6) raises ValueError! This gets caught by line 196.
# Result: position_size = 0.0, status = "error: ..." -> order vetoed silently.

print("\nBUG 7: NaN propagation through position sizing")
import numpy as np
confidence = float('nan')
conf_weight = np.clip(confidence, 0.5, 1.5)
print(f"  np.clip(NaN, 0.5, 1.5) = {conf_weight}")
print(f"  Is NaN: {np.isnan(conf_weight)}")
try:
    result = min(conf_weight, 100.0)
    print(f"  min(NaN, 100.0) = {result}")
    print(f"  round(NaN, 6) would raise: ", end="")
    print(round(result, 6))
except Exception as e:
    print(f"  round(NaN, 6) raises: {type(e).__name__}: {e}")

# ---- BUG 8 REPRO: feature_engineering.py line 268 - tail(32) before length check ----
# In transformer_brain.py _do_inference():
# Line 268: data = df_feat[cols].tail(32).values.astype(np.float32)
# Line 269: if len(data) < 32: return None
# The length check is AFTER tail(), so it catches short DataFrames correctly.
# But BEFORE that at line 231: `df_raw = fetch_bars(client, alpaca_symbol, days=2)`
# Line 232: `if df_raw is None or len(df_raw) < 32: return None`
# Then add_features() is called. If add_features() returns fewer rows for some reason,
# that's caught at line 269. OK, this logic is correct.

print("\nBUG 8: Length check after tail() - OK in transformer_brain.py")

# ---- BUG 9: shadow_arena.py - torch imported at module level without try/except ----
print("\nBUG 9: shadow_arena.py imports torch at module level (line 6)")
print("  `import torch` at line 6 - if torch is not installed, importing src.db will fail")
print("  because shadow_arena.py is imported from db.py (via ShadowTrade)")
print("  WAIT: shadow_arena.py is NOT imported by db.py. Let's check who imports it.")

print("\nAll reproduction tests completed.")
