"""Reproduction tests for audit findings. Run: python audit_repro.py"""
import asyncio, inspect, sys, math

results = []

def report(name, ok, detail):
    results.append((name, ok, detail))
    print(f"[{'REPRODUCED' if ok else 'NOT REPRODUCED'}] {name}\n    {detail}\n")

# ── Repro 1: max-hold-time exit is dead for live positions ──────────────────
# AlpacaExchange.get_positions() builds dicts WITHOUT "created_at"
# (src/exchange.py:300-309). strategies._check_price_based_exits gates the
# max-hold check on `if "created_at" in position` (strategies.py:537).
from src.strategies import TradingStrategy
from src.config import settings
from datetime import datetime, timezone, timedelta

strat = TradingStrategy(exchange=None)  # exchange unused by _check_price_based_exits
# scenario A: position shaped like the real exchange dict (no created_at),
# held far past MAX_HOLD_HOURS
pos_live = {"symbol": "BTC/USD", "qty": "1.0", "avg_entry_price": "100.0",
            "market_value": "100.0", "unrealized_pl": "0.0", "unrealized_plpc": "0.0"}
# scenario B: same position but with created_at long past the limit
pos_db = dict(pos_live, created_at=(datetime.now(timezone.utc) - timedelta(hours=settings.MAX_HOLD_HOURS + 100)).isoformat())
r_live = strat._check_price_based_exits("BTC/USD", 100.0, pos_live)   # price flat -> only max-hold could fire
r_db   = strat._check_price_based_exits("BTC/USD", 100.0, pos_db)
report("Max-hold exit dead with exchange-shaped position dict",
       r_live is None and r_db is not None and r_db.get("reason") == "max_hold_time_exceeded",
       f"live-shape -> {r_live} (max-hold never fires); with created_at -> {r_db.get('reason')}")

# ── Repro 2: NaN committee score passes threshold gate & yields 1.75x sizing ─
from src.committee.committee import calculate_confidence_size_multiplier
nan = float("nan")
mult = calculate_confidence_size_multiplier(nan, 0.0, 0.15)
report("NaN score bypasses threshold gate and yields max size multiplier",
       math.isnan(nan) and (nan < 0.15) is False and mult == 1.75,
       f"score=NaN: (nan < threshold) is {nan < 0.15}; size_multiplier = {mult} (should be 0/rejected)")

# ── Repro 3: dual AlertingEngine instances split cooldown state ──────────────
from src.alerting import AlertingEngine
from src.alerting import AlertCategory, AlertSeverity
e1, e2 = AlertingEngine(), AlertingEngine()
async def dual():
    a = await e1.fire(AlertCategory.SYSTEM, AlertSeverity.CRITICAL, "T", "msg", key="k")
    b = await e2.fire(AlertCategory.SYSTEM, AlertSeverity.CRITICAL, "T", "msg", key="k")
    return a, b
a, b = asyncio.run(dual())
report("Committee's private AlertingEngine bypasses bot engine's cooldown dedup",
       a is True and b is True,
       f"same alert key fired via engine1 -> sent={a}, via committee's separate engine2 -> sent={b} "
       f"(two live engines exist: src/committee/committee.py:35 and src/bot.py:1664)")

# ── Repro 4: bayesian_transformer.py:329 undefined BrainVote annotation ──────
import src.committee.bayesian_transformer as bt
try:
    import typing
    typing.get_type_hints(bt.bayesian_transformer_brain)
    ok, detail = False, "get_type_hints resolved fine"
except NameError as e:
    ok, detail = True, f"typing.get_type_hints(bayesian_transformer_brain) raises NameError: {e} (string annotation at line 329 references a name never imported at module scope)"
report("Undefined 'BrainVote' in bayesian_transformer signature (latent, annotation-only)", ok, detail)

# ── Repro 5: await-on-sync sweep on the live call graph ──────────────────────
from src.exchange import AlpacaExchange
from src.risk import RiskManager
from src.committee import committee as cmod
ex_methods = ["load","close","get_account","get_bars","get_positions","get_order","get_orders","create_order","get_latest_bar"]
bad = [m for m in ex_methods if not inspect.iscoroutinefunction(getattr(AlpacaExchange, m))]
risk_async = ["update_account_status","check_killswitch_conditions","check_and_reserve_exposure",
              "reserve_position_slot","reduce_exposure_to_cap","liquidate_all_positions"]
bad += [m for m in risk_async if not inspect.iscoroutinefunction(getattr(RiskManager, m))]
bad += [m for m in ["run_committee","run_decision_transformer","run_hierarchical_skills"]
        if not inspect.iscoroutinefunction(getattr(cmod, m, None))]
report("All hot-path awaited methods are genuinely async", not bad,
       f"checked {len(ex_methods)} exchange + {len(risk_async)} risk + 3 committee coroutine defs; non-async: {bad or 'none'}")

print("\n===== SUMMARY =====")
for name, ok, _ in results:
    print(f"  {'BUG CONFIRMED' if ok else 'no repro   '} — {name}")
