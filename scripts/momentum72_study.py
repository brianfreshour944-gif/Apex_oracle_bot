"""Isolated research harness: volatility-scaled 72-hour momentum.

Implements reports/MOMENTUM72_SPEC_2026-10-25.md EXACTLY. Read-only with respect
to production; reads cached OHLCV JSON only. No network, no DB, no orders.

Frozen parameters (do not tune):
    MOM_LOOKBACK   = 72   (hours, log-return sign)
    VOL_LOOKBACK   = 72   (hours, sample stdev of log returns)
    VOL_TARGET     = 0.005497  (pooled median hourly vol, DEVELOPMENT period only)
    EXPOSURE_CAP   = 1.0
    COST_ONE_WAY   = 28 bps   (half of the study's 56 bps round trip)

Return timing (spec §3): the return booked at bar *i+1* is
``w_i * logret_{i+1} - |w_i - w_{i-1}| * COST_ONE_WAY``. The primary backtest
(``delay=0``) therefore earns ``w_i * lr[i]`` on bar *i*, and the reported
one-bar-delayed sensitivity (``delay=1``) earns ``w_{i-1} * lr[i]``.

END-OF-PERIOD (flagged for ratification — the spec does not state it explicitly):
spec §3 defines an exhaustive per-bar return rule with no exit event, so a run
still open at the final bar is NOT force-liquidated: it accrues its per-bar
returns and pays only the entry/resize costs it actually incurred (no exit cost).
The spec's buy-and-hold baseline (§4) is charged a round trip because it is a
long-lived position the study explicitly closes; that is a separate statement.
If the operator prefers the sibling study's force-close-at-last-bar convention,
that is a one-line change here and would be a spec clarification, not a retune.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "ohlcv"

MOM_LOOKBACK = 72
VOL_LOOKBACK = 72
VOL_TARGET = 0.005497
EXPOSURE_CAP = 1.0
COST_ONE_WAY = 28.0  # bps per unit exposure change

DEV_START, DEV_END = "2025-10-01T00:00:00Z", "2026-10-24T23:59:59Z"
VAL_START, VAL_END = "2021-01-01T00:00:00Z", "2025-09-30T23:59:59Z"


def load(symbol: str, cache: Path = CACHE_DIR) -> list[dict]:
    path = cache / f"{symbol.replace('/', '_')}_1Hour.json"
    return json.loads(path.read_text())["bars"]


def log_returns(closes: list[float]) -> list[float]:
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]


def weights(closes: list[float], long_short: bool = False) -> list[float]:
    """Target exposure for each bar, using only information available at that bar.

    Returns a list the same length as closes; w[i] is the target held over the
    move from close[i] to close[i+1]. w[i] = 0 during warmup.
    """
    n = len(closes)
    lr = log_returns(closes)          # lr[j] is return of bar j+1 over bar j
    w = [0.0] * n
    for i in range(n):
        if i < MOM_LOOKBACK or i < VOL_LOOKBACK:
            continue
        mom = math.log(closes[i] / closes[i - MOM_LOOKBACK])
        window = lr[i - VOL_LOOKBACK:i]        # returns strictly up to bar i
        if len(window) < VOL_LOOKBACK:
            continue
        sd = statistics.stdev(window)
        if sd <= 0:
            continue
        scale = min(EXPOSURE_CAP, VOL_TARGET / sd)
        if long_short:
            w[i] = scale if mom > 0 else -scale
        else:
            w[i] = scale if mom > 0 else 0.0
    return w


def run(closes: list[float], long_short: bool = False, delay: int = 0) -> dict:
    """Backtest per the frozen spec §3.

    Return booked at bar *i+1* is
        w_i * logret_{i+1} - |w_i - w_{i-1}| * COST_ONE_WAY,
    so with ``delay=0`` (the primary) bar *i* earns ``w_i * lr[i]`` where
    ``lr[i] = log(close[i+1] / close[i])``. ``delay=1`` books the spec's
    one-bar-delayed sensitivity: ``w_{i-1} * lr[i]``.

    A *trade* is a maximal run of ``w > 0`` (spec §5.1). Its gross return is the
    sum of ``w_i * lr[i]`` over the run; its net subtracts the entry cost, any
    resize costs, and — when the run closes — the exit cost, so a full entry then
    exit costs 56 bps. A run still open at the last bar is **not liquidated**
    (spec §3 defines no exit event); it accrues its per-bar returns and the
    entry/resize costs it actually incurred. See END-OF-PERIOD note in the
    module docstring.
    """
    n = len(closes)
    w = weights(closes, long_short)
    lr = log_returns(closes)
    gross = 0.0
    net = 0.0
    cost_bps = 0.0
    prev = 0.0
    n_changes = 0
    exposure_sum = 0.0
    equity = [1.0]
    peak = 1.0
    maxdd = 0.0
    trades: list[float] = []          # net return per entry->flat episode
    trade_gross: list[float] = []
    cur_net = 0.0
    cur_gross = 0.0
    in_trade = False

    for i in range(n - 1):
        src = i - delay
        if src < 0:
            tgt = 0.0
        else:
            tgt = w[src]
        dw = abs(tgt - prev)
        if dw > 0:
            n_changes += 1
        c = dw * COST_ONE_WAY / 1e4
        r_gross = tgt * lr[i]
        r_net = r_gross - c
        gross += r_gross
        net += r_net
        cost_bps += dw * COST_ONE_WAY
        exposure_sum += abs(tgt)
        equity.append(equity[-1] * math.exp(r_net))
        peak = max(peak, equity[-1])
        maxdd = min(maxdd, equity[-1] / peak - 1)

        # trade bookkeeping: a trade is a maximal run with tgt != 0. The exit
        # cost is charged on the bar the target returns to 0, so it lands in
        # cur_net before the run is closed.
        if prev == 0 and tgt != 0:
            in_trade, cur_net, cur_gross = True, 0.0, 0.0
        if in_trade:
            cur_net += r_net
            cur_gross += r_gross
            if tgt == 0:
                trades.append(cur_net)
                trade_gross.append(cur_gross)
                in_trade = False
        prev = tgt

    if in_trade:
        # Run open at the last bar: no synthetic liquidation (spec §3 defines no
        # exit event), so no exit cost is charged.
        trades.append(cur_net)
        trade_gross.append(cur_gross)

    return {
        "n_trades": len(trades),
        "gross_log": gross,
        "net_log": net,
        "gross_total_pct": (math.exp(gross) - 1) * 100,
        "net_total_pct": (math.exp(net) - 1) * 100,
        "cost_bps": cost_bps,
        "n_exposure_changes": n_changes,
        "avg_exposure": exposure_sum / (n - 1) if n > 1 else 0.0,
        "max_drawdown_pct": maxdd * 100,
        "trades_net": trades,
        "trades_gross": trade_gross,
    }


def buy_hold(closes: list[float], cost_bps: float = 56.0) -> dict:
    ret = math.log(closes[-1] / closes[0]) - cost_bps / 1e4
    eq = [closes[0] / closes[0]]
    peak = 1.0
    maxdd = 0.0
    for c in closes[1:]:
        v = c / closes[0]
        eq.append(v)
        peak = max(peak, v)
        maxdd = min(maxdd, v / peak - 1)
    return {"net_total_pct": (math.exp(ret) - 1) * 100,
            "net_log": ret, "max_drawdown_pct": maxdd * 100}


def block_bootstrap_p(returns: list[float], block: int = 12, iters: int = 5000,
                      seed: int = 12345) -> float:
    if not returns:
        return 1.0
    rng = random.Random(seed)
    n = len(returns)
    block = min(block, n)
    means = []
    for _ in range(iters):
        s: list[float] = []
        while len(s) < n:
            start = rng.randrange(n)
            s.extend(returns[start:start + block])
        means.append(statistics.fmean(s[:n]))
    return sum(1 for m in means if m <= 0) / len(means)


def information_coefficient(closes: list[float], long_short: bool = False) -> float:
    w = weights(closes, long_short)
    lr = log_returns(closes)
    xs = w[:-1]
    ys = lr
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=False))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx * dy else 0.0


def slice_bars(bars: list[dict], start: str, end: str) -> list[dict]:
    return [b for b in bars if start <= b["t"] <= end]


def evaluate(symbol: str, cache: Path, start: str, end: str,
             long_short: bool = False) -> dict:
    bars = slice_bars(load(symbol, cache), start, end)
    closes = [float(b["c"]) for b in bars]
    if len(closes) < MOM_LOOKBACK + 2:
        return {"symbol": symbol, "error": "insufficient bars"}
    r = run(closes, long_short=long_short)
    r1 = run(closes, long_short=long_short, delay=1)
    bh = buy_hold(closes)
    tg = r["trades_gross"]
    tn = r["trades_net"]
    r["symbol"] = symbol
    r["bars"] = len(closes)
    r["period"] = f"{bars[0]['t']}..{bars[-1]['t']}"
    r["gross_mean_bps"] = statistics.fmean(tg) * 1e4 if tg else 0.0
    r["net_mean_bps"] = statistics.fmean(tn) * 1e4 if tn else 0.0
    r["win_rate"] = (sum(1 for t in tn if t > 0) / len(tn)) if tn else 0.0
    r["p_bootstrap"] = block_bootstrap_p(tn)
    r["ic"] = information_coefficient(closes, long_short)
    r["bh_net_total_pct"] = bh["net_total_pct"]
    r["bh_max_drawdown_pct"] = bh["max_drawdown_pct"]
    r["delay1_net_total_pct"] = r1["net_total_pct"]
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE_DIR))
    ap.add_argument("--start", default=VAL_START)
    ap.add_argument("--end", default=VAL_END)
    ap.add_argument("--symbols", nargs="+", default=["BTC/USD", "ETH/USD", "SOL/USD"])
    ap.add_argument("--long-short", action="store_true")
    ap.add_argument("--pool", action="store_true", help="pool trades across symbols")
    args = ap.parse_args()
    cache = Path(args.cache)

    all_net: list[float] = []
    all_gross: list[float] = []
    print(f"period {args.start} .. {args.end}  mode="
          f"{'LONG/SHORT' if args.long_short else 'LONG/FLAT'}")
    print(f"frozen: mom={MOM_LOOKBACK}h vol={VOL_LOOKBACK}h "
          f"target={VOL_TARGET} cap={EXPOSURE_CAP} cost1w={COST_ONE_WAY}bps\n")
    hdr = (f"{'symbol':8s} {'bars':>5s} {'trades':>6s} {'expo':>5s} "
           f"{'gross_bps':>9s} {'net_bps':>8s} {'gross%':>8s} {'net%':>8s} "
           f"{'bh%':>8s} {'maxDD%':>7s} {'IC':>7s} {'p':>6s} {'lag1%':>8s}")
    print(hdr)
    for s in args.symbols:
        r = evaluate(s, cache, args.start, args.end, args.long_short)
        if "error" in r:
            print(f"{s:8s} ERROR {r['error']}")
            continue
        all_net.extend(r["trades_net"])
        all_gross.extend(r["trades_gross"])
        print(f"{s:8s} {r['bars']:5d} {r['n_trades']:6d} {r['avg_exposure']:5.2f} "
              f"{r['gross_mean_bps']:9.2f} {r['net_mean_bps']:8.2f} "
              f"{r['gross_total_pct']:8.2f} {r['net_total_pct']:8.2f} "
              f"{r['bh_net_total_pct']:8.2f} {r['max_drawdown_pct']:7.2f} "
              f"{r['ic']:7.4f} {r['p_bootstrap']:6.3f} {r['delay1_net_total_pct']:8.2f}")
    if args.pool and all_net:
        print(f"\nPOOLED: trades={len(all_net)} "
              f"gross_mean={statistics.fmean(all_gross)*1e4:.2f}bps "
              f"net_mean={statistics.fmean(all_net)*1e4:.2f}bps "
              f"p={block_bootstrap_p(all_net):.4f}")


if __name__ == "__main__":
    main()
