"""Edge investigation harness (read-only, offline).

Purpose: look for a *statistically convincing* directional edge on cached hourly
crypto bars, using realistic costs and a held-out period that is not inspected
until the end. The prior study found no convincing edge; this deliberately tests
a SMALL set of pre-registered, simple hypotheses rather than adding complexity.

Design rules (set before looking at any result):
  * Pre-registered hypotheses only -- the list below is fixed in code, not chosen
    after seeing results.
  * In-sample = development. Holdout = untouched until the final section.
  * Costs: round-trip fee + slippage charged on every position change. A strategy
    that does not clear costs is reported as no edge regardless of gross signal.
  * Significance uses a block bootstrap over overlapping trades to avoid the
    false confidence that i.i.d. assumptions give on autocorrelated returns.

Read-only: reads data/ohlcv/*.json. No network, no DB writes, no orders.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "ohlcv"

# --- cost model (round-trip, in basis points of notional) ---------------------
TAKER_FEE_BPS = 25.0        # matches the bot's ESTIMATED_TAKER_FEE_BPS default
SLIPPAGE_BPS = 3.0          # conservative per side for liquid crypto majors
ROUND_TRIP_BPS = 2 * (TAKER_FEE_BPS + SLIPPAGE_BPS)


def load(symbol: str, timeframe: str = "1Hour") -> list[dict]:
    path = CACHE_DIR / f"{symbol.replace('/', '_')}_{timeframe}.json"
    data = json.loads(path.read_text())
    return data["bars"]


def closes(bars: list[dict]) -> list[float]:
    return [float(b["c"]) for b in bars]


# --- pre-registered hypotheses ------------------------------------------------
# Each returns a target position (+1 long, 0 flat) computed from data STRICTLY
# up to and including bar i, to be held over bar i+1. No look-ahead.

def h1_momentum(px: list[float], lookback: int, i: int) -> int:
    """Time-series momentum: long if the past `lookback` hours were up."""
    if i < lookback:
        return 0
    return 1 if px[i] > px[i - lookback] else 0


def h2_meanrev(px: list[float], lookback: int, i: int) -> int:
    """Mean reversion: long if the past `lookback` hours were down."""
    if i < lookback:
        return 0
    return 1 if px[i] < px[i - lookback] else 0


def h3_ma_cross(px: list[float], fast: int, slow: int, i: int) -> int:
    """Moving-average crossover: long while fast MA > slow MA."""
    if i < slow:
        return 0
    f = statistics.fmean(px[i - fast + 1: i + 1])
    s = statistics.fmean(px[i - slow + 1: i + 1])
    return 1 if f > s else 0


def h4_vol_breakout(px: list[float], lookback: int, i: int) -> int:
    """Volatility-conditioned breakout: long on a new `lookback`-high close."""
    if i < lookback:
        return 0
    return 1 if px[i] >= max(px[i - lookback + 1: i + 1]) else 0


HYPOTHESES = {
    "H1_momentum_24h": lambda px, i: h1_momentum(px, 24, i),
    "H1_momentum_72h": lambda px, i: h1_momentum(px, 72, i),
    "H2_meanrev_24h": lambda px, i: h2_meanrev(px, 24, i),
    "H2_meanrev_72h": lambda px, i: h2_meanrev(px, 72, i),
    "H3_ma_12_48": lambda px, i: h3_ma_cross(px, 12, 48, i),
    "H3_ma_24_168": lambda px, i: h3_ma_cross(px, 24, 168, i),
    "H4_breakout_48": lambda px, i: h4_vol_breakout(px, 48, i),
}


def backtest(px: list[float], signal, cost_bps: float, allow_short: bool = False) -> dict:
    """Long/flat (or long/short) backtest charging cost_bps per position change.

    Returns per-trade returns (fraction) and summary stats. A trade is a maximal
    run of consecutive bars at a non-zero position; its return is net of
    entry+exit cost. With allow_short, a 0 signal from the long/flat hypothesis
    is treated as -1 (always in the market) -- this isolates directional skill
    from market beta, which matters because the test period is a bear market.
    """
    def target(i: int) -> int:
        s = signal(px, i)          # 0 or 1 from the pre-registered rule
        if allow_short:
            return 1 if s == 1 else -1
        return s

    pos = 0
    trades: list[float] = []
    entry_px = None
    equity = 1.0
    for i in range(len(px) - 1):
        tgt = target(i)
        if tgt != pos:
            if pos != 0 and entry_px is not None:
                gross = (px[i] - entry_px) / entry_px
                if pos == -1:
                    gross = -gross
                ret = gross - cost_bps / 1e4
                trades.append(ret)
                equity *= (1 + ret)
            if tgt != 0:
                pos, entry_px = tgt, px[i]
            else:
                pos, entry_px = 0, None
    if pos != 0 and entry_px is not None:
        gross = (px[-1] - entry_px) / entry_px
        if pos == -1:
            gross = -gross
        trades.append(gross - cost_bps / 1e4)
    return {"n_trades": len(trades), "trades": trades, "equity": equity}


def buy_hold(px: list[float], cost_bps: float) -> dict:
    ret = (px[-1] - px[0]) / px[0] - cost_bps / 1e4
    return {"n_trades": 1, "trades": [ret], "equity": 1 + ret}


def block_bootstrap_p(trades: list[float], block: int = 12, iters: int = 5000,
                      seed: int = 12345) -> float:
    """One-sided p-value for mean trade return > 0 via a moving-block bootstrap.

    Overlapping hourly bars make trade returns autocorrelated, so an i.i.d.
    t-test overstates significance. Blocks of consecutive trades preserve some of
    that dependence. p = fraction of bootstrap resamples whose mean <= 0.
    """
    if not trades:
        return 1.0
    rng = random.Random(seed)
    n = len(trades)
    block = min(block, n)
    means = []
    for _ in range(iters):
        sample = []
        while len(sample) < n:
            start = rng.randrange(n)
            sample.extend(trades[start:start + block])
        sample = sample[:n]
        means.append(statistics.fmean(sample))
    return sum(1 for m in means if m <= 0) / len(means)


def summarize(name: str, bt: dict, cost_bps: float) -> dict:
    tr = bt["trades"]
    if not tr:
        return {"hypothesis": name, "n_trades": 0, "note": "no trades"}
    mean = statistics.fmean(tr)
    sd = statistics.pstdev(tr) if len(tr) > 1 else 0.0
    wins = sum(1 for t in tr if t > 0)
    gross = sum(t for t in tr if t > 0)
    loss = -sum(t for t in tr if t <= 0)
    return {
        "hypothesis": name,
        "n_trades": len(tr),
        "mean_ret_bps": round(mean * 1e4, 2),
        "sd_bps": round(sd * 1e4, 2),
        "win_rate": round(wins / len(tr), 3),
        "profit_factor": round(gross / loss, 3) if loss > 0 else float("inf"),
        "total_ret_pct": round((bt["equity"] - 1) * 100, 2),
        "p_mean_le_0": round(block_bootstrap_p(tr), 4),
    }


def split(bars: list[dict], holdout_start: str) -> tuple[list, list]:
    ins = [b for b in bars if b["t"] < holdout_start]
    hold = [b for b in bars if b["t"] >= holdout_start]
    return ins, hold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-start", default="2026-07-01T00:00:00Z")
    ap.add_argument("--symbols", nargs="+", default=["BTC/USD", "ETH/USD", "SOL/USD"])
    ap.add_argument("--holdout", action="store_true",
                    help="evaluate the holdout period (only for final validation)")
    args = ap.parse_args()

    print(f"Round-trip cost model: {ROUND_TRIP_BPS:.0f} bps "
          f"({TAKER_FEE_BPS:.0f} fee + {SLIPPAGE_BPS:.0f} slip per side)\n")
    if args.holdout:
        print(f"*** HOLDOUT MODE: bars at/after {args.holdout_start} ***\n")
    else:
        print(f"IN-SAMPLE  = bars before {args.holdout_start}")
        print(f"HOLDOUT    = bars at/after {args.holdout_start} (NOT inspected here)\n")

    for symbol in args.symbols:
        bars = load(symbol)
        ins_bars, hold_bars = split(bars, args.holdout_start)
        use_bars = hold_bars if args.holdout else ins_bars
        px = closes(use_bars)
        bh = buy_hold(px, ROUND_TRIP_BPS)
        print(f"=== {symbol} ===  {'holdout' if args.holdout else 'in-sample'} "
              f"{len(px)} bars")
        print(f"  buy&hold: {round((bh['equity']-1)*100,2)}%")
        for name, sig in HYPOTHESES.items():
            lf = summarize(name, backtest(px, sig, ROUND_TRIP_BPS, allow_short=False),
                           ROUND_TRIP_BPS)
            ls = summarize(name, backtest(px, sig, ROUND_TRIP_BPS, allow_short=True),
                           ROUND_TRIP_BPS)
            if lf.get("n_trades", 0) == 0 and ls.get("n_trades", 0) == 0:
                print(f"  {name:18s} no trades")
                continue
            print(f"  {name:18s} LONG/FLAT n={lf.get('n_trades',0):4d} "
                  f"mean={lf.get('mean_ret_bps',0):8.2f}bps "
                  f"tot={lf.get('total_ret_pct',0):8.2f}% "
                  f"p={lf.get('p_mean_le_0',1):.4f}")
            print(f"  {'':18s} L/S(always-in) n={ls.get('n_trades',0):4d} "
                  f"mean={ls.get('mean_ret_bps',0):8.2f}bps "
                  f"tot={ls.get('total_ret_pct',0):8.2f}% "
                  f"p={ls.get('p_mean_le_0',1):.4f}")
        print()


if __name__ == "__main__":
    main()
