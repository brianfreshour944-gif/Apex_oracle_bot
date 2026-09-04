"""Crash-recovery simulations: what happens after a hard kill with positions open.
Run: python audit_crash_recovery.py
"""
import asyncio, os, math, time
from datetime import datetime, timezone, timedelta

PASS = []
def report(name, severity, reproduced, detail):
    PASS.append((name, reproduced))
    print(f"[{'SIMULATED' if reproduced else 'CODE-TRACE ONLY'}] ({severity}) {name}\n    {detail}\n")

# ── SIM 1: trailing-stop state loss on cold restart ──────────────────────────
# risk.peak_prices and strategy._trailing_peaks live only in memory. Simulate:
# long BTC entered at 100, rallied to peak 120 (activation crossed, peak
# recorded), then crashed down to 108 -- pre-crash the trailing stop fires.
# After a restart the peak is forgotten.
from src.risk import RiskManager
from src.config import settings

class FakeEx:
    async def get_positions(self): return []
    async def get_account(self): return {"equity": 10000.0, "portfolio_value": 10000.0, "buying_power": 5000.0}
    async def create_order(self, **kw): return {"id": "x", "status": "filled", "filled_avg_price": 1.0, "filled_qty": 1.0, "commission": 0.0}

rm = RiskManager(FakeEx())
# pre-crash: price rallied through activation -> peak recorded at 120
rm.check_trailing_stop("BTC/USD", 120.0, 100.0, 1.0, regime="trending")
pre = rm.check_trailing_stop("BTC/USD", 108.0, 100.0, 1.0, regime="trending")
peak_pre = rm.peak_prices.get("BTC/USD")

rm_restarted = RiskManager(FakeEx())  # cold restart: peak_prices == {}
post = rm_restarted.check_trailing_stop("BTC/USD", 108.0, 100.0, 1.0, regime="trending")
peak_post = rm_restarted.peak_prices.get("BTC/USD", "<absent>")
report("Trailing-stop peak lost on restart; stop that should have fired does not",
       "money-risk (delayed/lost trailing exit)",
       pre == "close" and post == "hold" and peak_post == 108.0,
       f"pre-crash: peak={peak_pre}, price 120->108 => '{pre}' (correctly closes). "
       f"after cold restart: same price 108 => '{post}', and the peak silently RE-ANCHORS at the "
       f"current, already-fallen price ({peak_post}). The old 120 peak is gone, so the drawdown that "
       f"should have closed the position produces no exit; a new stop can only trigger on a further "
       f"~{settings.TRAILING_DISTANCE_PCT*100:.0f}% drop from 108.")

# ── SIM 2: cooldown loss -> immediate re-entry after restart ─────────────────
# _state.cooldowns is in-memory. Scenario: SL fired seconds before the crash.
import src.bot as bot_mod
pre_state = bot_mod.BotState()
pre_state.cooldowns["BTC/USD"] = time.time() + settings.COOLDOWN_SECONDS_BUY
pre_blocked = time.time() < pre_state.cooldowns["BTC/USD"]
post_state = bot_mod.BotState()  # cold restart
post_blocked = time.time() < post_state.cooldowns.get("BTC/USD", 0)
report("Entry cooldown forgotten on restart (whipsaw re-entry immediately after SL)",
       "money-risk (immediate re-entry into a position that just stopped out)",
       pre_blocked and not post_blocked,
       f"pre-crash: BTC/USD locked out for another {settings.COOLDOWN_SECONDS_BUY}s = {pre_blocked}; "
       f"after restart: cooldowns dict empty => blocked = {post_blocked}. "
       f"process_signal_for_symbol's cooldown gate reads only this in-memory dict.")

# ── SIM 3: scale-in cap + size decay forgotten on restart ────────────────────
from src.config import MAX_POSITION_ADDS, POSITION_ADD_SIZE_DECAY
pre_adds = {"count": MAX_POSITION_ADDS, "last_add_time": time.time(), "last_add_score": 0.9}
pre_blocked = pre_adds["count"] >= MAX_POSITION_ADDS
post_adds = bot_mod.BotState().position_adds.get("BTC/USD", {"count": 0, "last_add_time": 0.0, "last_add_score": 0.0})
post_blocked = post_adds["count"] >= MAX_POSITION_ADDS
pre_decay = POSITION_ADD_SIZE_DECAY ** pre_adds["count"]
post_decay = POSITION_ADD_SIZE_DECAY ** post_adds["count"]
report("Scale-in cap and size decay reset by restart -> re-adds beyond MAX_POSITION_ADDS at full size",
       "money-risk (oversized add to an existing position)",
       pre_blocked and not post_blocked,
       f"pre-crash: adds={pre_adds['count']}/{MAX_POSITION_ADDS} -> blocked={pre_blocked}, next add {pre_decay:.2f}x. "
       f"post-restart: adds=0 -> blocked={post_blocked}, next add {post_decay:.2f}x (full size). "
       f"The exchange still reports the same open position, so process_signal_for_symbol takes the scale-in branch.")

# ── SIM 4: stale 'open' decision snapshot after external close ───────────────
import src.db as db
settings.DATABASE_URL = "sqlite:///data/audit_crash.db"
db._open_snapshot_cache.clear()
db._tables_ensured = False
try:
    os.remove("data/audit_crash.db")
except OSError:
    pass
db.init_db()
db.save_decision_snapshot(decision_id="crash-1", symbol="ETH/USD", regime="bull",
                          final_action="buy", confidence=0.8, size_multiplier=1.0,
                          brain_votes={"quant": "buy"}, entry_price=3000.0, qty=0.5)
snap = db.get_open_snapshot("ETH/USD")
ok_open = snap is not None and snap["decision_id"] == "crash-1"
db.save_decision_snapshot(decision_id="crash-2", symbol="ETH/USD", regime="bull",
                          final_action="buy", confidence=0.8, size_multiplier=1.0,
                          brain_votes={"quant": "buy"}, entry_price=3200.0, qty=0.4)
served = (db._open_snapshot_cache.clear(), db.get_open_snapshot("ETH/USD"))[1]
report("Ghost 'open' decision snapshots after crash/external close; never reconciled at startup",
       "benign-to-moderate (polluted 'open' state; adaptive learner misses the lost trade's outcome)",
       ok_open and served["decision_id"] == "crash-2",
       f"snapshot crash-1 left open across 'restart': {ok_open}. After a new position opens (crash-2), "
       f"get_open_snapshot serves '{served['decision_id']}' -- a later exit closes the RIGHT snapshot, "
       f"but crash-1 stays status='open' forever (no startup reconciliation of snapshots vs positions).")

# ── SIM 5: corrupted DB at startup -> crash-loop or graceful degradation? ────
with open("data/audit_corrupt.db", "wb") as f:
    f.write(b"SQLite format 3\x00" + os.urandom(4096))
settings.DATABASE_URL = "sqlite:///data/audit_corrupt.db"
db._engine = None
db._tables_ensured = False
try:
    db.init_db()
    outcome = "init_db succeeded (unexpected)"
except Exception as e:
    outcome = f"init_db raised: {type(e).__name__}: {e}"
report("Corrupted SQLite DB does NOT crash-loop the bot; it silently degrades to offline mode",
       "moderate (trading continues with NO persistence: no snapshots, no meta-learner updates, no max-hold)",
       "init_db raised" in outcome,
       f"{outcome}. run_trading_bot catches this and keeps trading; all save_*/snapshot reads then fail "
       f"as non-fatal warnings -- no learning, no max-hold, no decision records.")
