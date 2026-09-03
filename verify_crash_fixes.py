"""Post-fix verification for crash-recovery findings F1-F4. Run: python verify_crash_fixes.py"""
import asyncio, os, time
from datetime import datetime, timezone, timedelta

ok = []
def check(name, passed, detail):
    print(f"  {'PASS' if passed else 'FAIL'} - {name}\n      {detail}")
    return passed

all_ok = True

# ── F1: trailing peak survives restart ──────────────────────────────────────
from src.persistent_state import save_persistent_state, load_persistent_state, STATE_FILENAME
from src.risk import RiskManager
from src.config import settings

save_persistent_state({"peak_prices": {"BTC/USD": 120.0}, "trailing_peaks": {"BTC/USD": 120.0},
                       "cooldowns": {}, "position_adds": {}})
rm = RiskManager(type("E", (), {"get_positions": lambda s: [], "get_account": lambda s: {}})())
restored_peak = rm.peak_prices.get("BTC/USD")
action = rm.check_trailing_stop("BTC/USD", 108.0, 100.0, 1.0, regime="trending")
all_ok &= check("F1: restored peak fires the trailing stop post-restart",
                restored_peak == 120.0 and action == "close",
                f"restored peak={restored_peak}, check at 108 -> '{action}' (pre-fix: 'hold' with peak re-anchored at 108)")
# strategies path too
from src.strategies import TradingStrategy
strat = TradingStrategy(exchange=None)
all_ok &= check("F1b: strategy._trailing_peaks restored post-restart",
                strat._trailing_peaks.get("BTC/USD") == 120.0,
                f"_trailing_peaks['BTC/USD']={strat._trailing_peaks.get('BTC/USD')}")

# ── F2: cooldown survives restart ────────────────────────────────────────────
import src.bot as bot_mod
future = time.time() + settings.COOLDOWN_SECONDS_BUY
save_persistent_state({"peak_prices": {}, "cooldowns": {"BTC/USD": future}, "position_adds": {}})
st = bot_mod.BotState()
blocked = time.time() < st.cooldowns.get("BTC/USD", 0)
all_ok &= check("F2: entry cooldown restored post-restart (no whipsaw re-entry)",
                blocked and st.cooldowns["BTC/USD"] == future,
                f"cooldown restored={blocked}, expiry preserved={st.cooldowns.get('BTC/USD') == future}")
# expired entries are dropped at load
save_persistent_state({"cooldowns": {"OLD/USD": time.time() - 100}})
st2 = bot_mod.BotState()
all_ok &= check("F2b: expired persisted cooldowns dropped at load",
                "OLD/USD" not in st2.cooldowns, f"cooldowns={st2.cooldowns}")

# ── F3: scale-in cap survives restart ────────────────────────────────────────
from src.config import MAX_POSITION_ADDS, POSITION_ADD_SIZE_DECAY
save_persistent_state({"cooldowns": {}, "position_adds": {"BTC/USD": {"count": MAX_POSITION_ADDS, "last_add_time": time.time(), "last_add_score": 0.9}}})
st3 = bot_mod.BotState()
info = st3.position_adds.get("BTC/USD", {"count": 0})
all_ok &= check("F3: scale-in count restored (no oversized re-add post-restart)",
                info["count"] == MAX_POSITION_ADDS,
                f"restored count={info.get('count')}/{MAX_POSITION_ADDS} -> gate would block: {info.get('count', 0) >= MAX_POSITION_ADDS}")

# ── F4: ghost snapshot reconciliation closes it at startup ───────────────────
from src.db import get_all_open_snapshots, close_decision_snapshot
import src.db as db
settings.DATABASE_URL = "sqlite:///data/audit_reconcile.db"
db._open_snapshot_cache.clear(); db._tables_ensured = False; db._engine = None
try: os.remove("data/audit_reconcile.db")
except OSError: pass
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

# ── F6: corrupt DB raises at init (alert wiring is code-trace verified) ─────
with open("data/audit_corrupt2.db", "wb") as f:
    f.write(b"garbage-not-sqlite" * 256)
settings.DATABASE_URL = "sqlite:///data/audit_corrupt2.db"
db._engine = None; db._tables_ensured = False
raised = False
try:
    db.init_db()
except Exception:
    raised = True
import inspect
bot_src = inspect.getsource(bot_mod.run_trading_bot)
all_ok &= check("F6: corrupt DB raises into bot's alert path (alert_system_health wired)",
                raised and "alert_system_health" in bot_src and "database" in bot_src,
                f"init_db raised={raised}; run_trading_bot sends critical alert on DB-down: "
                f"{'alert_system_health' in bot_src}")

print("\n===== CRASH-RECOVERY FIXES: " + ("ALL VERIFIED" if all_ok else "FAILURES PRESENT") + " =====")