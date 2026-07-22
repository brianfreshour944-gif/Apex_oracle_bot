#!/usr/bin/env python3
"""
Apex Oracle Bot – Macro Filter Backtest Optimizer
=================================================
Tests adding a 200-period Simple Moving Average (SMA) filter to long entries.
In a bear market (like the 2025-2026 period where BTC fell 46%), buying oversold
RSI dips in MEAN_REVERSION or NEUTRAL regimes can lead to "falling knife" losses.

Filter logic:
  - ONLY allow BUY entries if the current price is above the 200-period SMA.
"""

import sys
import io
import warnings
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

# ── Strategy ──

class KF1D:
    def __init__(self):
        self.q, self.r, self.p, self.x = 1e-5, 1e-3, 1.0, None
    def update(self, m):
        if self.x is None:
            self.x = m
            return self.x
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (m - self.x)
        self.p *= (1 - k)
        return self.x

def hurst(prices, ml=20):
    try:
        v = prices.values
        if len(v) < ml * 2:
            return 0.5
        lags = range(2, ml)
        vrs = [np.var(v[l:] - v[:-l]) for l in lags]
        vi = [i for i, x in enumerate(vrs) if x > 0]
        if len(vi) < 3:
            return 0.5
        p = np.polyfit(np.log([lags[i] for i in vi]), np.log([vrs[i] for i in vi]), 1)
        return float(np.clip(p[0]/2, 0, 1))
    except:
        return 0.5

def rsi(p, per=14):
    d = p.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.rolling(per, min_periods=per).mean()
    al = l.rolling(per, min_periods=per).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)

def atr(df, per=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(per, min_periods=per).mean().fillna(0)

def macd(p):
    ef = p.ewm(span=12, adjust=False).mean()
    es = p.ewm(span=26, adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=9, adjust=False).mean()
    return ml, sl

def regime(df, hu=0.55, hl=0.45):
    if len(df) < 50:
        return {"regime": "NEUTRAL", "hurst": 0.5, "rsi": 50.0, "vol": 0.0, "sp": 0.0, "md": 0.0}
    cs = df["close"]
    cp = cs.iloc[-1]
    a = atr(df).iloc[-1]
    vp = (a / cp) * 100 if cp > 0 else 0
    h = hurst(cs, 20)
    kf = KF1D()
    sp = cp
    for p in cs:
        sp = kf.update(p)
    r = rsi(cs).iloc[-1]
    m, ms = macd(cs)
    mv = m.iloc[-1]
    msv = ms.iloc[-1]
    if vp > 5:
        reg = "HIGH_VOL"
    elif h > hu:
        if cp > sp and mv > msv:
            reg = "TREND_BULLISH"
        elif cp < sp and mv < msv:
            reg = "TREND_BEARISH"
        else:
            reg = "TREND_NEUTRAL"
    elif h < hl:
        reg = "MEAN_REVERSION"
    else:
        reg = "NEUTRAL"
    return {"regime": reg, "hurst": round(h, 4), "rsi": round(r, 2), "vol": round(vp, 4),
            "sp": round(sp, 4), "md": round(mv - msv, 4)}

def signal(df, ri):
    reg, r, md = ri["regime"], ri["rsi"], ri["md"]
    if reg == "TREND_BULLISH":
        if r < 70 and md > 0:
            return "BUY"
    elif reg == "TREND_BEARISH":
        if r > 30 and md < 0:
            return "SELL"
    elif reg == "MEAN_REVERSION":
        if r < 30:
            return "BUY"
        elif r > 70:
            return "SELL"
    elif reg == "NEUTRAL":
        if r < 35:
            return "BUY"
        elif r > 65:
            return "SELL"
    return "HOLD"

# ── Simulation Engine ──

@dataclass
class Pos:
    sym: str
    qty: float
    ep: float
    bar: int
    time: object
    reg: str = ""
    hi: float = 0.0

def run_simulation(data, max_hold_h, stop_loss_pct, profit_target_pct, use_sma_filter=False):
    ACCOUNT_BASE = 10000.0
    BASE_RISK = 0.03
    MAX_TRADE = 750.0
    MAX_PORT = 2500.0
    MAX_POS = 3
    DD_STOP = -10.0
    DAILY_STOP = -3.0
    ATR_MULT = 2.0
    FEE = 0.001
    CD = 6
    HU = 0.55
    HL = 0.45

    times = sorted(set().union(*(d.index for d in data.values())))
    sb = 200
    if len(times) <= sb:
        return {}

    eq = ACCOUNT_BASE
    pk = eq
    cash = eq
    positions = {}
    comp = []
    eq_curve = []
    cd_until = {}
    daily_start_equity = eq
    last_day = None
    killed = False
    daily_halt = False

    for bi, t in enumerate(times):
        if bi < sb:
            continue
        if killed:
            break
        
        # Daily reset (Midnight UTC)
        dy = t.date() if hasattr(t, "date") else t
        if last_day is not None and dy != last_day:
            daily_start_equity = eq
            daily_halt = False
        last_day = dy

        pv = sum(p.qty * (data[p.sym].loc[t, "close"] if t in data[p.sym].index else p.ep) for p in positions.values())
        eq = cash + pv
        pk = max(pk, eq)
        
        # Max drawdown check
        dd = ((eq - pk) / pk) * 100 if pk > 0 else 0
        if dd <= DD_STOP:
            for s, p in list(positions.items()):
                if s in data and t in data[s].index:
                    sp = data[s].loc[t, "close"]
                    fee = sp * p.qty * FEE
                    cash += sp * p.qty - fee
                    comp.append({"pp": (sp - p.ep) / p.ep})
            positions.clear()
            killed = True
            eq = cash
            eq_curve.append({"t": t, "eq": eq})
            break

        # Daily stop check
        dpnl = ((eq - daily_start_equity) / daily_start_equity) * 100 if daily_start_equity > 0 else 0
        if dpnl <= DAILY_STOP and not daily_halt:
            for s, p in list(positions.items()):
                if s in data and t in data[s].index:
                    sp = data[s].loc[t, "close"]
                    fee = sp * p.qty * FEE
                    cash += sp * p.qty - fee
                    comp.append({"pp": (sp - p.ep) / p.ep})
            positions.clear()
            daily_halt = True
            eq = cash

        eq_curve.append({"t": t, "eq": eq})
        oc = len(positions)

        for sym, df_full in data.items():
            if t not in df_full.index:
                continue
            if bi < cd_until.get(sym, 0):
                continue

            loc = df_full.index.get_loc(t)
            if isinstance(loc, slice):
                loc = loc.stop - 1
            elif hasattr(loc, '__len__'):
                loc = loc[-1] if len(loc) > 0 else 0
            sl = max(0, loc - sb + 1)
            win = df_full.iloc[sl:loc+1].copy()
            if len(win) < 50:
                continue

            ri = regime(win, HU, HL)
            sig = signal(win, ri)
            cp = float(win["close"].iloc[-1])
            av = float(atr(win).iloc[-1])
            if cp <= 0:
                continue
            
            # SMA 200 filter check
            pass_filter = True
            if use_sma_filter:
                # Calculate 200 SMA on close prices
                sma_200 = df_full["close"].iloc[max(0, loc-200+1):loc+1].mean()
                if cp < sma_200:
                    pass_filter = False

            hp = sym in positions

            if hp:
                pos = positions[sym]
                pp = (cp - pos.ep) / pos.ep if pos.ep > 0 else 0
                hb = bi - pos.bar  # hold bars
                
                ex = False
                if pp >= profit_target_pct:
                    ex = True
                elif pp <= -stop_loss_pct:
                    ex = True
                elif hb >= max_hold_h:
                    ex = True
                elif sig == "SELL":
                    ex = True

                if ex:
                    fee = cp * pos.qty * FEE
                    cash += cp * pos.qty - fee
                    comp.append({"pp": pp})
                    del positions[sym]
                    cd_until[sym] = bi + CD
                    oc -= 1
            else:
                if sig == "BUY" and oc < MAX_POS and not daily_halt and pass_filter:
                    # Sizing: use ATR
                    risk = eq * BASE_RISK
                    sd = av * ATR_MULT
                    q = risk / sd if sd > 0 else 0
                    if q * cp > MAX_TRADE:
                        q = MAX_TRADE / cp
                    tc = q * cp
                    
                    # Portfolio exposure check
                    ce = sum(p.qty * (data[p.sym].loc[t, "close"] if t in data[p.sym].index else p.ep) for p in positions.values())
                    if ce + tc > MAX_PORT:
                        continue
                    if cash < tc:
                        continue
                    if q > 0:
                        fee = tc * FEE
                        cash -= tc + fee
                        positions[sym] = Pos(sym, q, cp, bi, t, ri["regime"], cp)
                        cd_until[sym] = bi + CD // 2
                        oc += 1

    ft = times[-1]
    for s, p in list(positions.items()):
        if s in data and ft in data[s].index:
            sp = data[s].loc[ft, "close"]
            fee = sp * p.qty * FEE
            cash += sp * p.qty - fee
            comp.append({"pp": (sp - p.ep) / p.ep})
    positions.clear()
    eq = cash
    eq_curve.append({"t": ft, "eq": eq})

    pnls = [c["pp"] for c in comp]
    w = [p for p in pnls if p > 0]
    l = [p for p in pnls if p <= 0]
    edf = pd.DataFrame(eq_curve)
    ret = (edf["eq"].iloc[-1] - ACCOUNT_BASE) / ACCOUNT_BASE * 100 if len(edf) > 0 else 0.0
    pk2 = edf["eq"].cummax()
    mdd = ((edf["eq"] - pk2) / pk2 * 100).min()
    gp = sum(w) if w else 0
    gl = abs(sum(l)) if l else 0
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0)
    n = len(comp)

    return {
        "ret": ret,
        "mdd": mdd,
        "pf": pf,
        "wr": len(w) / n * 100 if n > 0 else 0.0,
        "n": n,
        "killed": killed
    }

if __name__ == "__main__":
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    print("[*] Fetching 1y data ...")
    data = {}
    for sym in symbols:
        tk = yf.Ticker(sym)
        df = tk.history(period="1y", interval="1h")
        if not df.empty:
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            data[sym] = df

    # Compare Baseline vs SMA Filtered across different hold parameters
    test_cases = [
        # (hold_h, SL, PT)
        (8, 0.02, 0.03, "BASELINE (Current Bot Parameters)"),
        (24, 0.03, 0.05, "MODERATE (24h Hold, 3% SL, 5% PT)"),
        (48, 0.05, 0.05, "CONSERVATIVE (48h Hold, 5% SL, 5% PT)"),
    ]

    print("\nEvaluating SMA 200 Trend Filter Effectiveness...")
    print(f"{'Configuration':<42s} | {'Filter':<6s} | {'Return%':>8s} | {'MaxDD%':>7s} | {'WR%':>6s} | {'PF':>6s} | {'Trades':>6s} | {'Killed':>6s}")
    print("-" * 100)

    for h, sl, pt, label in test_cases:
        # Without Filter
        res_no = run_simulation(data, h, sl, pt, use_sma_filter=False)
        killed_no = "YES" if res_no["killed"] else "no"
        print(f"{label:<42s} | OFF    | {res_no['ret']:>+7.2f}% | {res_no['mdd']:>6.2f}% | {res_no['wr']:>5.1f}% | {res_no['pf']:>5.2f} | {res_no['n']:>6d} | {killed_no:>6s}")
        
        # With Filter
        res_yes = run_simulation(data, h, sl, pt, use_sma_filter=True)
        killed_yes = "YES" if res_yes["killed"] else "no"
        print(f"{label:<42s} | ON     | {res_yes['ret']:>+7.2f}% | {res_yes['mdd']:>6.2f}% | {res_yes['wr']:>5.1f}% | {res_yes['pf']:>5.2f} | {res_yes['n']:>6d} | {killed_yes:>6s}")
        print("-" * 100)
