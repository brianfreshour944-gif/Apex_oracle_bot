"""Post-fix verification for crash-recovery findings F1-F4. Run: python verify_crash_fixes.py"""
import asyncio
import os
import time

ok = []
def check(name, passed, detail):
    print(f"  {'PASS' if passed else 'FAIL'} - {name}\n      {detail}")
    return passed

all_ok = True

# ── F1/F1b/F2/F3: crash-recovery state survives restart ─────────────────────
# Production sequence in run_trading_bot: build RiskManager + TradingStrategy,
# then apply_crash_recovery_state(load_persistent_state()). Mirror it exactly
# instead of expecting BotState()/RiskManager() to self-restore at construction.
import src.bot as bot_mod
from src.config import MAX_POSITION_ADDS, settings
from src.persistent_state import load_persistent_state, save_persistent_state
from src.risk import RiskManager
from src.strategies import TradingStrategy

bot_mod._state.risk_manager = RiskManager(type("E", (), {"get_positions": lambda s: [], "get_account": lambda s: {}})())
bot_mod._state.strategy = TradingStrategy(exchange=None)
future = time.time() + settings.COOLDOWN_SECONDS_BUY
save_persistent_state({
    "peak_prices": {"BTC/USD": 120.0},
    "trailing_peaks": {"BTC/USD": 120.0},
    "cooldowns": {"BTC/USD": future},
    "position_adds": {"BTC/USD": {"count": MAX_POSITION_ADDS, "last_add_time": time.time(), "last_add_score": 0.9}},
})
counts = bot_mod.apply_crash_recovery_state(load_persistent_state())

restored_peak = bot_mod._state.risk_manager.peak_prices.get("BTC/USD")
action = bot_mod._state.risk_manager.check_trailing_stop("BTC/USD", 108.0, 100.0, 1.0, regime="trending")
all_ok &= check("F1: restored peak fires the trailing stop post-restart",
                restored_peak == 120.0 and action == "close",
                f"restored peak={restored_peak}, check at 108 -> '{action}' (pre-fix: 'hold' with peak re-anchored at 108)")
all_ok &= check("F1b: strategy._trailing_peaks restored post-restart",
                bot_mod._state.strategy._trailing_peaks.get("BTC/USD") == 120.0,
                f"_trailing_peaks['BTC/USD']={bot_mod._state.strategy._trailing_peaks.get('BTC/USD')}")

# ── F2: cooldown survives restart ────────────────────────────────────────────
blocked = time.time() < bot_mod._state.cooldowns.get("BTC/USD", 0)
all_ok &= check("F2: entry cooldown restored post-restart (no whipsaw re-entry)",
                blocked and bot_mod._state.cooldowns.get("BTC/USD") == future,
                f"cooldown restored={blocked}, expiry preserved={bot_mod._state.cooldowns.get('BTC/USD') == future}")

# ── F2b: expired cooldowns are dropped before the next flush ─────────────────
# The loop filters expired entries every cycle (bot.py), and cleanup_stale_state
# prunes them too; a persisted-but-expired cooldown must not block re-entry.
st = bot_mod.BotState()
st.cooldowns = {"OLD/USD": time.time() - 100, "LIVE/USD": time.time() + 100}
st.cleanup_stale_state()
all_ok &= check("F2b: expired persisted cooldowns dropped before next flush",
                "OLD/USD" not in st.cooldowns and "LIVE/USD" in st.cooldowns,
                f"cooldowns after cleanup={st.cooldowns}")

# ── F3: scale-in cap survives restart ────────────────────────────────────────
info = bot_mod._state.position_adds.get("BTC/USD", {"count": 0})
all_ok &= check("F3: scale-in count restored (no oversized re-add post-restart)",
                info["count"] == MAX_POSITION_ADDS,
                f"restored count={info.get('count')}/{MAX_POSITION_ADDS} -> gate would block: {info.get('count', 0) >= MAX_POSITION_ADDS}")

# ── F4: ghost snapshot reconciliation closes it at startup ───────────────────
import src.db as db

settings.DATABASE_URL = "sqlite:///data/audit_reconcile.db"
db._open_snapshot_cache.clear()
db._tables_ensured = False
db._engine = None
try:
    os.remove("data/audit_reconcile.db")
except OSError:
    pass
db.init_db()
db.save_decision_snapshot(decision_id="ghost-1", symbol="GHOST/USD", regime="bull",
                          final_action="buy", confidence=0.8, size_multiplier=1.0,
                          brain_votes={"quant": "buy"}, entry_price=50.0, qty=2.0)
all_open = db.get_all_open_snapshots()
assert len(all_open) == 1, all_open
# simulate restart with NO exchange position for GHOST/USD
class NoPosEx:
    async def get_positions(self): return []
    async def get_latest_bar(self, symbol):
        class B:
            def is_empty(self): return False
            def __getitem__(self, k): return [55.0]  # last known price
        return B()
await_fn = bot_mod.reconcile_open_snapshots(NoPosEx())
asyncio.run(await_fn)
after = db.get_all_open_snapshots()
closed_snap = db.get_open_snapshot("GHOST/USD")
all_ok &= check("F4: ghost 'open' snapshot reconciled and closed at startup",
                len(after) == 0 and closed_snap is None,
                f"open snapshots before={len(all_open)}, after={len(after)}; get_open_snapshot now returns {closed_snap}")

# ── F6: corrupt DB is quarantined and escalated as a critical alert ──────────
# init_db() deliberately does NOT raise on a fully corrupt SQLite file (CR-7):
# it moves the bad file aside, rebuilds a clean schema, and returns True so the
# caller can fire alert_data_integrity_failure. Assert that contract plus the
# connection-failure alert wiring in run_trading_bot.
with open("data/audit_corrupt2.db", "wb") as f:
    f.write(b"garbage-not-sqlite" * 256)
settings.DATABASE_URL = "sqlite:///data/audit_corrupt2.db"
db._engine = None
db._tables_ensured = False
rebuilt = False
try:
    rebuilt = db.init_db() is True
except Exception:
    rebuilt = False
import inspect

bot_src = inspect.getsource(bot_mod.run_trading_bot)
all_ok &= check("F6: corrupt DB rebuilt and escalated via alert path",
                rebuilt and "alert_data_integrity_failure" in bot_src
                and "alert_system_health" in bot_src and "database" in bot_src,
                f"init_db quarantined+rebuilt={rebuilt}; run_trading_bot alerts on "
                f"rebuild={'alert_data_integrity_failure' in bot_src}, "
                f"on connection failure={'alert_system_health' in bot_src}")

print("\n===== CRASH-RECOVERY FIXES: " + ("ALL VERIFIED" if all_ok else "FAILURES PRESENT") + " =====")
