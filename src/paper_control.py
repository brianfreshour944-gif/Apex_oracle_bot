"""Buy-and-hold control account: the bar the bot has to beat.

A $100 paper account that buys the bot's three symbols at first observation and
never trades again. It places NO orders, touches NO exchange, and writes only
its own JSON ledger under ``data/`` -- it exists purely to measure whether the
bot's trading beats simply holding.

Why a control is needed: a strategy that "loses less than it might have" is not
an edge. The honest bar is the return of the same capital in the same coins held
passively. If the bot cannot beat this, its activity is destroying value.

Reference from the backtest (not this ledger): over the longest daily history the
SMA(20/50) crossover returned BTC +35% vs buy&hold +18,141%, ETH +87% vs +692%.
So the control is expected to win by a wide margin over long horizons -- that is
the point, and the ledger will show it in real time.
"""

from __future__ import annotations

import datetime
import json
import os

_LEDGER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "paper_control_buyhold.json")

# The account to beat: $100, equal weight across the bot's symbols.
INITIAL_USD = 100.0
CONTROL_SYMBOLS = ("BTC/USD", "ETH/USD", "SOL/USD")

# Entry fee charged on notional, in basis points. Matches the bot's
# TX_COST_FEE_BPS (25 bps = 0.25%), the real Alpaca crypto taker rate, so the
# control pays the same cost the bot does. Buy-and-hold never sells, so this
# one entry fee is the only cost it carries.
ENTRY_FEE_BPS = 25.0


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def load_ledger() -> dict:
    try:
        with open(_LEDGER_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_ledger(ledger: dict) -> None:
    os.makedirs(os.path.dirname(_LEDGER_PATH), exist_ok=True)
    tmp = f"{_LEDGER_PATH}.tmp"
    with open(tmp, "w") as f:
        json.dump(ledger, f, indent=2)
    os.replace(tmp, _LEDGER_PATH)  # atomic; a half-written ledger never loads


def initialize(prices: dict[str, float], initial_usd: float = INITIAL_USD,
               entry_fee_bps: float = ENTRY_FEE_BPS) -> dict:
    """Create the control at first observation: equal-weight buy, charging the
    same 0.25% entry fee the bot pays (on notional). Buy-and-hold never sells,
    so this entry fee is the only cost it carries over the whole horizon."""
    syms = [s for s in CONTROL_SYMBOLS if s in prices and prices[s] > 0]
    if not syms:
        raise ValueError("no usable prices to initialize the control")
    fee_rate = entry_fee_bps / 10_000.0
    per = initial_usd / len(syms)
    # Buy notional of `per` and pay the fee on top: qty = notional / price,
    # cash out = per * (1 + fee). Track the fee so the ledger is auditable.
    positions = {
        s: {"entry_price": float(prices[s]), "qty": per / float(prices[s]),
            "entry_fee_usd": round(per * fee_rate, 6)}
        for s in syms
    }
    ledger = {
        "created_at": _now(),
        "initial_usd": float(initial_usd),
        "entry_fee_bps": entry_fee_bps,
        "entry_fees_usd": round(sum(p["entry_fee_usd"] for p in positions.values()), 6),
        "weights": {s: round(per / initial_usd, 4) for s in syms},
        "positions": positions,
        "snapshots": [],
    }
    return ledger


def value(ledger: dict, prices: dict[str, float]) -> dict:
    """Mark the control to market and return a summary.

    Equity = market value of the holdings MINUS the entry fees paid. The fees
    are real cash out (0.25% of notional), so ignoring them would overstate the
    control -- the benchmark must pay the same entry cost the bot does.
    """
    total = 0.0
    per_symbol = {}
    for s, pos in ledger.get("positions", {}).items():
        px = prices.get(s)
        if px is None or px <= 0:
            px = pos["entry_price"]  # last known: never fabricate a move
        val = pos["qty"] * float(px)
        total += val
        per_symbol[s] = {
            "entry_price": pos["entry_price"],
            "mark_price": float(px),
            "value_usd": round(val, 4),
            "return_pct": round((float(px) / pos["entry_price"] - 1) * 100, 3),
        }
    init = float(ledger.get("initial_usd", INITIAL_USD))
    fees = float(ledger.get("entry_fees_usd", 0.0))
    equity = total - fees
    return {
        "as_of": _now(),
        "equity_usd": round(equity, 4),
        "return_pct": round((equity / init - 1) * 100, 3),
        "entry_fees_usd": round(fees, 6),
        "per_symbol": per_symbol,
    }


def update(prices: dict[str, float], initial_usd: float = INITIAL_USD) -> dict:
    """Initialize on first call, then append a valuation snapshot each call."""
    ledger = load_ledger()
    if not ledger.get("positions"):
        ledger = initialize(prices, initial_usd)
    summary = value(ledger, prices)
    ledger.setdefault("snapshots", []).append({
        "t": summary["as_of"],
        "equity_usd": summary["equity_usd"],
        "return_pct": summary["return_pct"],
    })
    save_ledger(ledger)
    return summary


def summary() -> dict:
    """Current control summary from the last snapshot, or {} if never run."""
    ledger = load_ledger()
    snaps = ledger.get("snapshots") or []
    if not snaps:
        return {}
    last = snaps[-1]
    return {
        "created_at": ledger.get("created_at"),
        "initial_usd": ledger.get("initial_usd"),
        "entry_fee_bps": ledger.get("entry_fee_bps"),
        "entry_fees_usd": ledger.get("entry_fees_usd"),
        "equity_usd": last["equity_usd"],
        "return_pct": last["return_pct"],
        "n_snapshots": len(snaps),
    }
