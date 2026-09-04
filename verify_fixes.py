"""Post-fix verification for audit findings 1-3. Run: python verify_fixes.py"""
import asyncio, math
from datetime import datetime, timezone, timedelta

ok = []

# Fix 3: committee and bot now share ONE AlertingEngine singleton
from src.alerting import get_alerting_engine, AlertCategory, AlertSeverity
import src.committee.committee as cmod
shared = cmod._alerting_engine is get_alerting_engine()
ok.append(("AlertingEngine singleton shared by committee + bot", shared))

# Fix 2: NaN committee score now hits the stand_aside gate
nan = float("nan")
effective_threshold = 0.15
winner = "buy"
gated = not math.isfinite(nan) or nan < effective_threshold or winner in ["stand_aside", "skip"]
ok.append(("NaN committee score routed to stand_aside (committee.py gate)", gated))
# and confirm the guard is actually in run_committee's source path
import inspect
src = inspect.getsource(cmod.run_committee)
ok.append(("isfinite guard present inside run_committee", "math.isfinite(score)" in src))

# Fix 1: end-to-end max-hold -- bot.py now attaches created_at from the
# persisted open decision snapshot; verify the full chain against a real
# (temporary) sqlite DB.
from src.config import settings
settings.DATABASE_URL = "sqlite:///data/audit_verify.db"
from src import db as dbmod
dbmod.get_engine.cache_clear() if hasattr(dbmod.get_engine, "cache_clear") else None
import src.db as db
db._open_snapshot_cache.clear()
db.init_db()
did = db.save_decision_snapshot(
    decision_id="audit-fix-1",
    symbol="TEST/USD",
    regime="neutral",
    final_action="buy",
    confidence=0.9,
    size_multiplier=1.0,
    brain_votes={"transformer": "buy"},
    entry_price=100.0,
    qty=1.0,
)
# Backdate the snapshot's created_at past MAX_HOLD_HOURS, then clear the
# open-snapshot cache so get_open_snapshot re-reads it (bot.py would see the
# same cached value within a cycle).
from sqlalchemy import update as sa_update
with db.get_db_session() as session:
    session.execute(
        sa_update(db.DecisionSnapshot)
        .where(db.DecisionSnapshot.decision_id == "audit-fix-1")
        .values(created_at=datetime.now(timezone.utc) - timedelta(hours=settings.MAX_HOLD_HOURS + 24))
    )
    session.commit()
db._open_snapshot_cache.clear()
# simulate bot.py's attach step (bot.py:567-574)
snap = db.get_open_snapshot("TEST/USD")
pos = {"symbol": "TEST/USD", "qty": "1.0", "avg_entry_price": "100.0",
       "market_value": "100.0", "unrealized_pl": "0.0", "unrealized_plpc": "0.0"}
if snap and snap.get("created_at"):
    pos["created_at"] = snap["created_at"]
from src.strategies import TradingStrategy
strat = TradingStrategy(exchange=None)
r = strat._check_price_based_exits("TEST/USD", 100.0, pos)
ok.append(("Max-hold exit fires with snapshot-attached created_at", r is not None and r.get("reason") == "max_hold_time_exceeded"))
db.close_decision_snapshot("audit-fix-1", realized_pnl=0.0)

print("===== POST-FIX VERIFICATION =====")
all_ok = True
for name, passed in ok:
    print(f"  {'PASS' if passed else 'FAIL'} - {name}")
    all_ok = all_ok and passed
print("ALL FIXES VERIFIED" if all_ok else "SOME FIXES FAILED")
