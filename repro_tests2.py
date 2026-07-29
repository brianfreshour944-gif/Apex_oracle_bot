"""
Final targeted reproduction tests for critical confirmed bugs.
"""
import asyncio
import numpy as np

print("=" * 60)
print("TEST 1: Global 'strategy' variable scoping bug in bot.py")
print("=" * 60)

# Simulating the actual code structure
strategy_global = None  # Module level: bot.py line 138

async def _record_committee_outcome_sim(symbol, exit_price):
    global strategy_global
    max_fav_pct = 0.0
    # bot.py lines 61-66: check global strategy
    if strategy_global is not None and hasattr(strategy_global, '_trailing_peaks') and symbol in strategy_global._trailing_peaks:
        # This branch is NEVER reached because strategy_global is always None
        max_fav_pct = 99.9  # would be set here
    print(f"  max_fav_pct = {max_fav_pct} (always 0.0 because strategy_global is None)")

async def run_trading_bot_sim():
    global strategy_global  # NOTE: only 'ex' is declared global in real code, NOT 'strategy'
    # bot.py line 722: this assignment is LOCAL, not to the global
    class FakeStrategy:
        _trailing_peaks = {"BTC/USD": 50000.0}
    
    # In real code, this is NOT declared global:
    strategy_local = FakeStrategy()
    # strategy_global is still None
    
    await _record_committee_outcome_sim("BTC/USD", 49000.0)
    print(f"  strategy_global after bot init: {strategy_global}")
    print(f"  strategy_local._trailing_peaks: {strategy_local._trailing_peaks}")
    print("  CONFIRMED: max_favorable_pct will always be 0.0 in DB records")

asyncio.run(run_trading_bot_sim())

print()
print("=" * 60)
print("TEST 2: strategy_selector.py always records 'buy' action")
print("=" * 60)

# Simulating record_strategy_outcome with a sell trade
def record_strategy_outcome_sim(regime, strategy_name, pnl, return_pct):
    """Simulation of actual code in strategy_selector.py lines 79-113"""
    STRATEGIES = {"trend_following": None, "mean_reversion": None, "momentum": None}
    mock_votes = {}
    for strat in STRATEGIES.keys():
        if strat == strategy_name:
            mock_votes[strat] = "buy"   # BUG: always "buy" regardless of actual trade direction
        else:
            mock_votes[strat] = "stand_aside"
    
    decision_snapshot = {
        "regime": regime,
        "final_action": "buy",  # BUG: always "buy"
        "brain_votes": mock_votes
    }
    
    outcome = {"net_pnl": pnl, "return_pct": return_pct}
    return decision_snapshot, outcome

# Test: a winning SELL trade
snap, out = record_strategy_outcome_sim("bear", "trend_following", -100.0, -2.0)
print(f"  Sell trade loses $100. Recorded as: action='{snap['final_action']}', pnl={out['net_pnl']}")
print(f"  -> AdaptiveMetaLearner sees: buy-voted by trend_following, net_pnl=-100")
print(f"  -> Penalizes trend_following for 'buy' vote (wrong) - CORRECT PENALTY for a sell trade!")
print(f"  Actually this specific case is accidentally correct for losses.")
print()

snap2, out2 = record_strategy_outcome_sim("bear", "trend_following", 100.0, 2.0)
print(f"  Sell trade wins $100. Recorded as: action='{snap2['final_action']}', pnl={out2['net_pnl']}")
print(f"  -> AdaptiveMetaLearner sees: buy-voted by trend_following, net_pnl=+100")
print(f"  -> REWARDS trend_following for 'buy' vote when it actually placed a SELL trade")
print(f"  -> This is WRONG: sell strategy is being rewarded as if it voted buy and won")

print()
print("=" * 60)
print("TEST 3: NaN confidence propagation in position sizing")
print("=" * 60)

# committee_result.score could theoretically be NaN if scoring math fails
# Let's trace through calculate_position_size
confidence = float('nan')  
try:
    conf_weight = np.clip(confidence, 0.5, 1.5)
    print(f"  np.clip(NaN, 0.5, 1.5) = {conf_weight}")  # NaN
    position_size = 0.001 * conf_weight  # propagates NaN
    print(f"  position_size after NaN confidence: {position_size}")
    # min() in Python with NaN:
    result = min(position_size, 2500.0 / 50000.0)
    print(f"  min(NaN, cap) = {result}")
    rounded = round(result, 6)
    print(f"  round(NaN, 6) = {rounded}")
except (ValueError, TypeError) as e:
    print(f"  Exception: {type(e).__name__}: {e}")
    print(f"  -> Caught by except block, returns (0.0, 'error: ...')")
    print(f"  -> Trade is silently vetoed with cryptic error message")

print()
print("=" * 60)
print("TEST 4: sentinel_brain soft veto threshold off-by-one")
print("=" * 60)
# sentinel_brain.py line 33-34: ATR > 10% -> confidence = 0.80 (NOT veto)
# committee.py line 237: sentinel veto if `v.name == "sentinel" and v.action == "stand_aside" and v.confidence >= 0.85`
# ATR 10% case: confidence = 0.80, threshold is 0.85, so NOT vetoed
# But the docstring says "Severe penalty, just below the 0.85 hard veto threshold"
# So this is INTENTIONAL - ATR>10% gives a very high stand_aside confidence but not a veto.
# This works as designed.
print("  sentinel_brain ATR>10% -> confidence=0.80, veto threshold is 0.85")
print("  0.80 < 0.85 -> NOT vetoed (intentional per comment)")
print("  This is working as designed.")

print()
print("=" * 60)
print("TEST 5: shadow_arena.py - unchecked torch import")
print("=" * 60)
# shadow_arena.py line 6: `import torch` - no try/except
# It's imported lazily from transformer_brain.py at line 279 (inside _do_inference, inside try/except)
# So if torch is missing, transformer_brain.py gracefully handles it.
# But shadow_arena.py itself would fail to import if imported directly.
# Since it's only imported from within a try/except in _do_inference, this is safe.
print("  shadow_arena.py: `import torch` at line 6, no try/except guard")
print("  BUT: it's only imported inside a try/except in transformer_brain.py line 279")
print("  So a missing torch won't crash the bot (caught by outer try/except)")
print("  However: shadow_arena.py has no independent guard for import-time failures")
print("  Severity: low (defensive issue, not a crash)")

print()
print("=" * 60)
print("TEST 6: attribution.py - blocking call in async function")
print("=" * 60)
# _call_gemini_attribution at line 108:
# model.generate_content(prompt) - this is a SYNCHRONOUS call
# It's inside an async function but NOT wrapped in asyncio.to_thread
# This blocks the entire asyncio event loop for the duration of the LLM call
print("  attribution.py:114 - model.generate_content(prompt)")
print("  google.generativeai GenerativeModel.generate_content() is SYNCHRONOUS")
print("  Called directly in async function WITHOUT asyncio.to_thread()")
print("  This blocks the event loop during LLM inference (could be 1-10 seconds)")
print("  Impact: ALL other async tasks (trailing stops, new signals) are FROZEN")
print("  during attribution analysis. Attribution runs async from a fire-and-forget task.")
print("  Severity: performance/correctness - not a crash but affects responsiveness")

print()
print("All tests complete.")
