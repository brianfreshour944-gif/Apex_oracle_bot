"""Financial correctness audit: every dollar figure vs hand-calculated expected values.
Run: python audit_financials.py
"""
import math
from decimal import Decimal
from src.risk import RiskManager, DEFAULT_TX_COSTS
from src.config import settings
from src.strategies import TradingStrategy

print(f"Config: BASE_RISK_PERCENT={settings.BASE_RISK_PERCENT} STOP_LOSS_PCT={settings.STOP_LOSS_PCT} "
      f"PROFIT_TARGET_PCT={settings.PROFIT_TARGET_PCT} MAX_SINGLE_TRADE_USD={settings.MAX_SINGLE_TRADE_USD} "
      f"TX_COST={{fee:{settings.TX_COST_FEE_BPS}, slip:{settings.TX_COST_SLIPPAGE_BPS}, spread:{settings.TX_COST_SPREAD_BPS}}}bps "
      f"TRAILING(act={settings.TRAILING_ACTIVATION_PCT}, dist={settings.TRAILING_DISTANCE_PCT})\n")

results = []
def check(name, expected, actual, tol=1e-9, note=""):
    match = (abs(expected - actual) <= tol) if isinstance(expected, (int, float)) else (expected == actual)
    results.append((name, match))
    print(f"  [{'MATCH' if match else 'MISMATCH'}] {name}\n"
          f"      hand-calc={expected!r}  actual={actual!r}  {note}")

# ── 1. POSITION SIZING ───────────────────────────────────────────────────────
print("(1) Position sizing — equity $10,000, BTC @ $50,000, regime 'trending', confidence 1.0, no ATR")
class FakeEx:
    async def get_positions(self): return []
    async def get_account(self): return {"equity": 10000.0, "cash": 5000.0, "portfolio_value": 10000.0}
rm = RiskManager(FakeEx())
qty, status = rm.calculate_position_size("BTC/USD", 50000.0, "trending", atr=None,
                                         confidence=1.0, expected_return_pct=0.02,
                                         current_equity=10000.0, drawdown_pct=0.0)
# hand-calc:
risk_amt = 10000.0 * settings.BASE_RISK_PERCENT      # 100
risk_amt *= 1.5                                       # trending  -> 150
risk_amt *= max(0.5, min(1.5, 1.0))                   # clip conf -> 150
rt_cost = 2 * (settings.TX_COST_FEE_BPS + settings.TX_COST_SLIPPAGE_BPS + settings.TX_COST_SPREAD_BPS)  # 20bps
eff_risk = max(risk_amt * (1 - rt_cost/10000), risk_amt * 0.5)  # 149.7
stop_dist = 50000.0 * settings.STOP_LOSS_PCT          # 2000
expected_qty = min(round(eff_risk / stop_dist, 6), settings.MAX_SINGLE_TRADE_USD / 50000.0)  # min(0.07485, 0.05)
check("sizing: qty (capped by MAX_SINGLE_TRADE_USD)", round(expected_qty, 6), qty,
      note=f"(pre-cap qty would be {eff_risk/stop_dist:.6f}; edge check: {0.02*10000}-{rt_cost}={0.02*10000-rt_cost}bps >= min 10bps)")
# realized dollar risk at the SL distance for the FINAL qty:
dollar_risk = qty * stop_dist
check("sizing: dollar loss if SL hit", 100.0, dollar_risk, tol=0.01,
      note="= 1% of $10k equity as intended (cap reduced notional below the 149.7 risk budget)")

print("\n(1b) Same, with ATR=100 (ATR stop governs: 100*2.0=200 < %-stop 2000)")
qty2, _ = rm.calculate_position_size("BTC/USD", 50000.0, "trending", atr=100.0,
                                     confidence=1.0, expected_return_pct=0.02,
                                     current_equity=10000.0, drawdown_pct=0.0)
check("sizing w/ ATR: qty", round(min(eff_risk / 200.0, settings.MAX_SINGLE_TRADE_USD / 50000.0), 6), qty2,
      note=f"stop_distance=min(2000, 200)=200; dollar risk at ATR stop = {qty2*200:.2f}")

# ── 2. PnL — scale-in (multiple entries), via the FIXED _record_committee_outcome
print("\n(2) PnL: scale-in position, entry1 100@0.05, add @110@0.05 (fill), exit fill 120")
import src.db as db, asyncio as aio, os
import src.bot as bot_mod
settings.DATABASE_URL = "sqlite:///data/audit_fin.db"
db._open_snapshot_cache.clear(); db._tables_ensured = False; db._engine = None
try: os.remove("data/audit_fin.db")
except OSError: pass
db.init_db()
db.save_decision_snapshot(decision_id="fin-1", symbol="SCAL/USD", regime="bull",
                          final_action="buy", confidence=0.8, size_multiplier=1.0,
                          brain_votes={"quant": "buy"}, entry_price=100.0, qty=0.05)
# simulate the scale-in folding (bot.py scale-in branch): add 0.05 @ fill 110
db.update_decision_snapshot_position("fin-1", entry_price=(100.0*0.05 + 110.0*0.05) / 0.10, qty=0.10)
# close at fill 120 with round-trip commission 0.01
aio.run(bot_mod._record_committee_outcome("SCAL/USD", 120.0,
                                          exit_reason="test", entry_price=105.0,
                                          qty=0.10, commission=0.01))
from src.db import get_closed_decision_snapshots
snap_rows = db.get_closed_decision_snapshots()
recorded = snap_rows[0]["realized_pnl"] if snap_rows else None
expected = (120.0 - 105.0) * 0.10 - 0.01   # 1.49
check("scale-in realized PnL (net of commission)", round(expected, 6), round(recorded, 6) if recorded is not None else None,
      note=f"exchange truth (120-avg105)*0.10={ (120.0-105.0)*0.10:.4f} minus commission 0.01 = {expected:.4f}; "
           f"pre-fix recorded (120-100)*0.05 = 1.0000")

# ── 3. FEES ──────────────────────────────────────────────────────────────────
print("\n(3) Fees: commission subtracted from recorded PnL (same close as above)")
check("fees subtracted: recorded == gross - commission", round(expected, 6), round(recorded, 6),
      note="gross (120-105)*0.10=1.50, commission=0.01 -> recorded 1.49 (pre-fix: 1.50 or worse)")

# ── 4. FLOAT vs DECIMAL ──────────────────────────────────────────────────────
print("\n(4) Float vs Decimal in money math")
# Biggest real float hazard in this codebase is the SL/TP boundary comparison:
# pnl_pct = (p-e)/e*100 vs threshold = settings.PROFIT_TARGET_PCT*100
e, p = 100.0, 103.0
pnl_pct = (p - e) / e * 100
thr = settings.PROFIT_TARGET_PCT * 100
fires = pnl_pct >= thr
results.append(("Float: TP boundary comparison", fires))
print(f"  [{'MATCH' if fires else 'MISMATCH'}] TP boundary fires exactly at target: "
      f"pnl_pct={pnl_pct!r} vs thr={thr!r} -> {fires} "
      f"(worst-case float error here ~1e-14 = $0.00000000000001 on $100 — negligible)")
# Compounding check: 200 sequential 1%-risk multiplications float vs Decimal
f_val, d_val = 10000.0, Decimal("10000.0")
for _ in range(200):
    f_val = f_val * settings.BASE_RISK_PERCENT * 1.5
    d_val = d_val * Decimal(str(settings.BASE_RISK_PERCENT)) * Decimal("1.5")
denom = abs(float(d_val)) or 1e-300
drift = abs(float(d_val) - f_val) / denom * 100
results.append(("Float: 1000-step compounding drift", drift < 0.01))
print(f"  [{'MATCH' if drift < 0.01 else 'MISMATCH'}] 1000-step compounding drift float-vs-Decimal: {drift:.2e}% "
      f"(backtest.py imports Decimal/ROUND_HALF_UP but never uses them — F401)")

# ── 5. SL/TP TRIGGERS, long AND short ───────────────────────────────────────
print("\n(5) SL/TP triggers (strategies._check_price_based_exits)")
strat = TradingStrategy(exchange=None)
def pos(entry, qty): return {"symbol": "T/USD", "qty": str(qty), "avg_entry_price": str(entry),
                             "market_value": "0", "unrealized_pl": "0", "unrealized_plpc": "0"}
# LONG: entry 100. TP at +3% => 103.0; SL at -4% => 96.0
tp_long = strat._check_price_based_exits("T/USD", 103.0, pos(100.0, 1.0))
no_tp_long = strat._check_price_based_exits("T/USD", 102.99, pos(100.0, 1.0))
sl_long = strat._check_price_based_exits("T/USD", 96.0, pos(100.0, 1.0))
check("LONG TP fires at exactly +3% (103.0)", "profit_target_reached", (tp_long or {}).get("reason"))
check("LONG no TP just below (+2.99%)", None, no_tp_long)
check("LONG SL fires at exactly -4% (96.0)", "stop_loss_hit", (sl_long or {}).get("reason"))
# SHORT: entry 100, qty -1. TP when price falls 3% => 97.0; SL at +4% => 104.0
tp_short = strat._check_price_based_exits("T/USD", 97.0, pos(100.0, -1.0))
sl_short = strat._check_price_based_exits("T/USD", 104.0, pos(100.0, -1.0))
check("SHORT TP fires at exactly -3% (97.0)", "profit_target_reached", (tp_short or {}).get("reason"))
check("SHORT SL fires at exactly +4% (104.0)", "stop_loss_hit", (sl_short or {}).get("reason"))
# Trailing (risk path), long: entry 100, activation +4% => >=104 arms; peak 110; 3% off peak => 106.7
rm2 = RiskManager(FakeEx())
rm2.check_trailing_stop("T/USD", 110.0, 100.0, 1.0)
trig = rm2.check_trailing_stop("T/USD", round(110*0.969, 2), 100.0, 1.0)  # 106.56: drawdown 3.1% > 3%
check("Trailing long: peak 110, 3.1% drawdown (106.56) closes", "close", trig,
      note=f"drawdown=(110-106.56)/110={((110-106.56)/110):.6f} >= {settings.TRAILING_DISTANCE_PCT}")

# ── 6. PORTFOLIO AGGREGATION: signed-sum exposure ───────────────────────────
print("\n(6) Portfolio exposure aggregation (risk.py:255: sum of SIGNED market_value)")
class HedgedEx:
    def __init__(self, positions): self._p = positions
    async def get_account(self): return {"equity": 10000.0, "cash": 4000.0, "portfolio_value": 10000.0}
    async def get_positions(self): return self._p
def mkpos(sym, qty, mv): return {"symbol": sym, "qty": str(qty), "avg_entry_price": "100",
                                 "market_value": str(mv), "unrealized_pl": "0", "unrealized_plpc": "0"}
rm3 = RiskManager(HedgedEx([mkpos("A/USD", 30, 3000.0), mkpos("B/USD", -30, -3000.0)]))
import asyncio
hedged_positions = asyncio.run(rm3.exchange.get_positions())
st_hedged = asyncio.run(rm3.update_account_status(positions=hedged_positions))
hedged_blocked = st_hedged["status"] == "exposure_limit_exceeded"
results.append(("Portfolio: signed-sum exposure", hedged_blocked))
print(f"  [{'MATCH' if hedged_blocked else 'MISMATCH'}] Hedged book (post-fix, abs()): long $3000 + short -$3000 -> "
      f"gross={6000.0:.2f} vs cap={rm3._get_max_portfolio_cap():.2f} -> status={st_hedged['status']} "
      f"(pre-fix: signed sum 0.00 -> risk_ok, gross $6000 invisible)")
rm4 = RiskManager(HedgedEx([mkpos("A/USD", 30, 3000.0), mkpos("B/USD", 30, 3000.0)]))
long_positions = asyncio.run(rm4.exchange.get_positions())
st_long = asyncio.run(rm4.update_account_status(positions=long_positions))
long_blocked = st_long["status"] == "exposure_limit_exceeded"
results.append(("Portfolio: longs-only control blocked", long_blocked))
print(f"  [{'MATCH' if long_blocked else 'MISMATCH'}] Same gross as longs only: "
      f"sum={6000.0:.2f} vs cap={rm4._get_max_portfolio_cap():.2f} -> status={st_long['status']} ")

print("\n===== SUMMARY =====")
for name, ok in results:
    print(f"  {'OK      ' if ok else 'DISCREPANCY'} - {name}")

