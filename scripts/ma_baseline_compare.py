"""Honest baseline comparison: committee/strategy vs a dumb SMA crossover,
across fixed exit policies chosen up front.

Purpose: answer two questions.
1. Is the committee adding anything over a simple moving-average crossover?
2. Does a different exit policy (trailing-only, or a wider target) change that?

The exit variants are FIXED and chosen before looking at results -- do not
tune them against this test window:
  current   : TP +4% / SL -4%, trailing enabled (as the live bot is configured)
  C1        : trailing-only -- no fixed TP; trail 3% once +4% is reached; SL -4%
  C2_tp6    : TP +6% / SL -4%, no trailing

Every variant uses the SAME data, the SAME fees (from settings, so it tracks
TX_COST_FEE_BPS), and the SAME long-only / one-position-per-symbol / 8h
max-hold rules. Both engines are run under each variant: the SMA baseline is
implemented here; the committee is the repository's own
``run_backtest(use_committee=True)`` with ``settings`` temporarily overridden
(in memory only -- no files or DB are written).

Limitations (stated, not hidden):
- Fills at the bar close; stops checked at bar closes only. Absolute numbers
  are optimistic vs live; the comparison between engines is fair because both
  share this.
- yfinance 1h history is ~730 days max, so "longest window" is bounded there.
- Exit-reason tallies are printed so a "trailing-only" result that is really
  a time exit (8h max hold) is visible as such.
- Read-only: fetches public bars, places no orders, writes no state.

Usage:
    ALPACA_API_KEY=x ALPACA_SECRET_KEY=y python scripts/ma_baseline_compare.py
    ... --symbols BTC/USD,ETH/USD,SOL/USD --days 700 --fast 20 --slow 50
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from collections import Counter
from dataclasses import dataclass, field

sys.path.insert(0, ".")

import polars as pl

from src.config import settings

# Exit policies, fixed up front. ``tp`` is the profit target (None = no fixed
# target); ``trail`` is (activation, distance) or None.
VARIANTS: dict[str, dict] = {
    "current": {"tp": 0.04, "sl": 0.04, "trail": None},
    "C1_trail": {"tp": None, "sl": 0.04, "trail": (0.04, 0.03)},
    "C2_tp6": {"tp": 0.06, "sl": 0.04, "trail": None},
}


@dataclass
class Trade:
    entry_i: int
    exit_i: int
    entry: float
    exit: float
    pnl_pct: float  # NET of round-trip cost, as % of entry notional
    reason: str


@dataclass
class Result:
    symbol: str
    trades: list[Trade] = field(default_factory=list)
    equity: float = 1.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    fees_paid: float = 0.0

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl_pct > 0) / len(self.trades)

    @property
    def avg_win_pct(self) -> float:
        w = [t.pnl_pct for t in self.trades if t.pnl_pct > 0]
        return sum(w) / len(w) if w else 0.0

    @property
    def avg_loss_pct(self) -> float:
        losses = [t.pnl_pct for t in self.trades if t.pnl_pct <= 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def expectancy_pct(self) -> float:
        """Mean net P&L per trade as % of notional."""
        if not self.trades:
            return 0.0
        return sum(t.pnl_pct for t in self.trades) / len(self.trades)

    @property
    def profit_factor(self) -> float:
        gains = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
        losses = -sum(t.pnl_pct for t in self.trades if t.pnl_pct < 0)
        if losses <= 0:
            return float("inf") if gains > 0 else 0.0
        return gains / losses

    def breakeven_win_rate(self) -> float:
        """Win rate at which expectancy == 0, given observed avg win/loss."""
        aw, al = self.avg_win_pct, abs(self.avg_loss_pct)
        if aw + al <= 0:
            return 0.0
        return al / (aw + al)

    def curve_stats(self) -> tuple[float, float]:
        """(compounded return %, max drawdown %) from this trade sequence."""
        eq, peak, dd = 1.0, 1.0, 0.0
        for t in self.trades:
            eq *= 1 + t.pnl_pct / 100.0
            peak = max(peak, eq)
            dd = min(dd, (eq - peak) / peak)
        return (eq - 1) * 100, dd * 100


def round_trip_bps() -> float:
    return 2.0 * (
        settings.TX_COST_FEE_BPS
        + settings.TX_COST_SLIPPAGE_BPS
        + settings.TX_COST_SPREAD_BPS
    )


def sma(vals: list[float], n: int) -> list[float | None]:
    out: list[float | None] = [None] * len(vals)
    run = 0.0
    for i, v in enumerate(vals):
        run += v
        if i >= n:
            run -= vals[i - n]
        if i >= n - 1:
            out[i] = run / n
    return out


def run_sma_backtest(
    symbol: str,
    closes: list[float],
    times: list[str],
    fast: int,
    slow: int,
    tp_pct: float | None,
    sl_pct: float,
    rt_bps: float,
    warmup: int,
    trail: tuple[float, float] | None = None,
    max_hold_bars: int | None = None,
) -> Result:
    """Long-only SMA(fast/slow) crossover with a fixed exit policy, one position.

    Entry when SMA(fast) crosses above SMA(slow). Exit on the first of:
    SL (-sl_pct), fixed TP (+tp_pct, if set), trailing stop (if ``trail`` is
    set and armed), max hold (``max_hold_bars``), or the opposite cross.
    Costs are charged on close exactly like ``backtest.py::close()``.
    """
    res = Result(symbol=symbol)
    f = sma(closes, fast)
    s = sma(closes, slow)
    cost = rt_bps / 10000.0

    in_pos = False
    entry = 0.0
    entry_i = 0
    peak = 0.0
    equity = 1.0
    eq_peak = 1.0
    max_dd = 0.0

    for i in range(max(warmup, slow), len(closes)):
        if f[i] is None or s[i] is None or f[i - 1] is None or s[i - 1] is None:
            continue
        px = closes[i]
        crossed_up = f[i - 1] <= s[i - 1] and f[i] > s[i]
        crossed_dn = f[i - 1] >= s[i - 1] and f[i] < s[i]

        if not in_pos:
            if crossed_up:
                in_pos, entry, entry_i, peak = True, px, i, px
            continue

        peak = max(peak, px)
        ret = (px - entry) / entry
        reason = None
        if ret <= -sl_pct:
            reason = "sl"
        elif tp_pct is not None and ret >= tp_pct:
            reason = "tp"
        elif trail is not None and (peak - entry) / entry >= trail[0] and (peak - px) / peak >= trail[1]:
            reason = "trail"
        elif max_hold_bars is not None and (i - entry_i) >= max_hold_bars:
            reason = "max_hold"
        elif crossed_dn:
            reason = "cross"

        if reason is None:
            continue

        net = ret - cost
        equity *= 1 + net
        res.fees_paid += cost
        res.trades.append(Trade(entry_i, i, entry, px, net * 100, reason))
        eq_peak = max(eq_peak, equity)
        max_dd = min(max_dd, (equity - eq_peak) / eq_peak)
        in_pos = False

    if in_pos:
        px = closes[-1]
        net = (px - entry) / entry - cost
        equity *= 1 + net
        res.fees_paid += cost
        res.trades.append(Trade(entry_i, len(closes) - 1, entry, px, net * 100, "eod"))

    res.equity = equity
    res.total_return_pct = (equity - 1) * 100
    res.max_drawdown_pct = max_dd * 100
    return res


def fmt(r: Result, n_bars: int) -> str:
    return (
        f"  trades={len(r.trades):>3}  win={r.win_rate*100:5.1f}%  "
        f"avgW={r.avg_win_pct:+5.2f}% avgL={r.avg_loss_pct:+6.2f}%  "
        f"PF={r.profit_factor:5.2f}  exp={r.expectancy_pct:+6.3f}%/trade  "
        f"BE_WR={r.breakeven_win_rate()*100:5.1f}%  ret={r.total_return_pct:+7.2f}%  "
        f"maxDD={r.max_drawdown_pct:6.2f}%"
    )


def reasons(trades: list[Trade]) -> str:
    c = Counter(t.reason for t in trades)
    return " ".join(f"{k}={v}" for k, v in sorted(c.items()))


def _fetch_bars_yf(symbol: str, days: int, timeframe: str,
                   period: str | None = None) -> pl.DataFrame | None:
    """Fetch OHLCV bars via yfinance, with the columns the engine expects.

    ``period`` (e.g. "max") overrides ``days`` for the longest available history.
    """
    try:
        import yfinance as yf
    except Exception as e:
        print(f"  yfinance unavailable: {e}")
        return None
    tk = yf.Ticker(symbol.replace("/", "-"))
    df = tk.history(period=period, interval=timeframe) if period else \
        tk.history(period=f"{days}d", interval=timeframe)
    if df is None or df.empty:
        return None
    df = df.reset_index()
    ren = {}
    for src, dst in (("Datetime", "t"), ("Date", "t"), ("Open", "open"),
                     ("High", "high"), ("Low", "low"), ("Close", "close"),
                     ("Volume", "volume")):
        if src in df.columns:
            ren[src] = dst
    df = df.rename(columns=ren)
    if "t" not in df.columns:
        return None
    df["t"] = df["t"].astype(str)
    keep = [c for c in ("t", "open", "high", "low", "close", "volume") if c in df.columns]
    return pl.from_pandas(df[keep])


@contextlib.contextmanager
def override_settings(**kw):
    """Temporarily set settings attrs (in memory) and restore them after."""
    old = {k: getattr(settings, k) for k in kw}
    try:
        for k, v in kw.items():
            setattr(settings, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(settings, k, v)


# Bar length in hours, for converting MAX_HOLD_HOURS to a bar count.
_BAR_HOURS = {"1h": 1, "4h": 4, "1d": 24, "1D": 24}


def buy_and_hold(closes: list[float], start_i: int, rt_bps: float) -> float:
    """% return of holding from closes[start_i] to the last bar, net of one
    round-trip cost (one entry + one exit)."""
    if start_i >= len(closes) - 1 or closes[start_i] <= 0:
        return 0.0
    return (closes[-1] / closes[start_i] - 1) * 100 - rt_bps / 100.0


async def run_maxhold_sweep(args) -> None:
    """Vary ONLY the max-hold (exit policy fixed at TP4/SL4, no trailing),
    across timeframes, with a buy-and-hold benchmark. Variants fixed up front."""
    tp, sl = 0.04, 0.04
    rt = round_trip_bps()
    max_holds = [float(x) for x in args.max_holds.split(",") if x.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    print("=" * 108)
    print(f"MAX-HOLD LIFT SWEEP  fast={args.fast} slow={args.slow}  TP={tp*100:.1f}% SL={sl*100:.1f}% "
          f"(no trailing)  round-trip={rt:.0f}bps  max_holds={max_holds}  timeframes={timeframes}")
    print(f"warmup={args.warmup_frac:.0%} of bars untraded; OOS split={args.oos_split:.0%} test (judge on test half)")
    print("=" * 108)

    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        for tf in timeframes:
            bars = _fetch_bars_yf(sym, args.days, tf)
            if bars is None or len(bars) < args.slow + 10:
                print(f"\n{sym} [{tf}]: no usable data -- skipped")
                continue
            bars = bars.sort("t")
            closes = [float(c) for c in bars["close"].to_list()]
            n = len(closes)
            warmup = int(n * args.warmup_frac)
            split_i = warmup + int((n - warmup) * (1 - args.oos_split))
            ts = bars["t"].to_list()
            print(f"\n{sym} [{tf}]  ({n} bars, {ts[0]} .. {ts[-1]}, warmup={warmup}, "
                  f"test starts idx {split_i})")
            print(f"  BUY&HOLD eval-window: {buy_and_hold(closes, warmup, rt):+8.2f}%   "
                  f"OOS: {buy_and_hold(closes, split_i, rt):+8.2f}%")

            bh = _BAR_HOURS.get(tf, 1)
            for mh in max_holds:
                mh_bars = max(1, round(mh / bh))
                r = run_sma_backtest(sym, closes, None, args.fast, args.slow,
                                     tp, sl, rt, warmup, trail=None, max_hold_bars=mh_bars)
                tr = [t for t in r.trades if t.exit_i < split_i]
                te = [t for t in r.trades if t.exit_i >= split_i]
                print(f"  maxhold={mh:>4.0f}h ({mh_bars:>3} bar) SMA :{fmt(r, n)}   exits[{reasons(r.trades)}]")
                for lbl, ts in ((" IS ", tr), (" OOS", te)):
                    if ts:
                        rr = Result(symbol=sym, trades=ts)
                        rr.total_return_pct, rr.max_drawdown_pct = rr.curve_stats()
                        print(f"  {'':>20}     {lbl} :{fmt(rr, n)}   exits[{reasons(ts)}]")
                    else:
                        print(f"  {'':>20}     {lbl} : (no trades)")

    print("\n" + "=" * 108)
    print("Judge on the OOS rows. If lifting the 8h cap does not turn the test-half")
    print("expectancy positive, the cap is not what is causing the losses.")


async def run_long_daily(args) -> None:
    """Longest available DAILY history; SMA(fast/slow) crossover, TP/SL, no
    trailing; report full-period and per-calendar-year. Buy&hold per year too.
    Pre-registered pairs: 20/50 and 50/200."""
    tp, sl = 0.04, 0.04
    rt = round_trip_bps()
    pairs = []
    for p in args.pairs.split(","):
        f, s = p.split("/")
        pairs.append((int(f), int(s)))

    print("=" * 108)
    print(f"LONG DAILY SMA  pairs={pairs}  TP={tp*100:.1f}% SL={sl*100:.1f}% (no trailing)  "
          f"round-trip={rt:.0f}bps  period=max (longest available daily history)")
    print("=" * 108)

    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        bars = _fetch_bars_yf(sym, 0, "1d", period=args.period)
        if bars is None or len(bars) < 210:
            print(f"\n{sym}: only {0 if bars is None else len(bars)} daily bars -- skipped")
            continue
        bars = bars.sort("t")
        closes = [float(c) for c in bars["close"].to_list()]
        dates = [str(t)[:10] for t in bars["t"].to_list()]
        n = len(closes)
        print(f"\n{sym}  ({n} daily bars, {dates[0]} .. {dates[-1]})")
        print(f"  BUY&HOLD full: {buy_and_hold(closes, 0, rt):+9.2f}%")

        # Per-year buy & hold, so the strategy is read against its market.
        years = sorted({d[:4] for d in dates})
        for y in years:
            idx = [i for i, d in enumerate(dates) if d[:4] == y]
            if len(idx) < 2:
                continue
            bh_y = (closes[idx[-1]] / closes[idx[0]] - 1) * 100
            print(f"    {y} buy&hold: {bh_y:+8.2f}%")

        for fast, slow in pairs:
            if n <= slow + 5:
                print(f"  SMA {fast}/{slow}: not enough bars -- skipped")
                continue
            r = run_sma_backtest(sym, closes, None, fast, slow, tp, sl, rt, 0,
                                 trail=None, max_hold_bars=None)
            print(f"  SMA {fast}/{slow} FULL :{fmt(r, n)}   exits[{reasons(r.trades)}]")
            print(f"  {'':>9}   yr    trades  win%    exp%/trade   ret%     exits")
            for y in years:
                yt = [t for t in r.trades if dates[t.exit_i][:4] == y]
                if not yt:
                    continue
                rr = Result(symbol=sym, trades=yt)
                rr.total_return_pct, rr.max_drawdown_pct = rr.curve_stats()
                print(f"  {'':>9}   {y}  {len(yt):>6}  {rr.win_rate*100:5.1f}  "
                      f"{rr.expectancy_pct:+9.3f}   {rr.total_return_pct:+7.2f}   [{reasons(yt)}]")

    print("\n" + "=" * 108)
    print("Compare each year's strategy return against that year's buy&hold. A trend")
    print("strategy should at least protect capital in down years; losing MORE than")
    print("buy&hold in a down year means the entries are the problem, not the exits.")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default="BTC/USD,ETH/USD,SOL/USD")
    ap.add_argument("--days", type=int, default=700)
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=50)
    ap.add_argument("--warmup-frac", type=float, default=0.25,
                    help="Fraction of bars used as untraded warm-up before evaluation.")
    ap.add_argument("--oos-split", type=float, default=0.5,
                    help="Fraction of the evaluated window that is out-of-sample (train first).")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="Comma-separated subset of exit variants to run.")
    ap.add_argument("--maxhold-sweep", action="store_true",
                    help="Run the max-hold lift sweep instead of the exit-variant comparison.")
    ap.add_argument("--long-daily", action="store_true",
                    help="Run the longest-history daily SMA comparison with per-year breakdown.")
    ap.add_argument("--pairs", default="20/50,50/200",
                    help="Comma-separated fast/slow pairs for --long-daily.")
    ap.add_argument("--period", default="max",
                    help="yfinance period for --long-daily (e.g. 'max', '10y').")
    ap.add_argument("--max-holds", default="8,24,48,96",
                    help="Comma-separated MAX_HOLD_HOURS values for the sweep.")
    ap.add_argument("--timeframes", default="1h,1d",
                    help="Comma-separated timeframes for the sweep.")
    ap.add_argument("--no-committee", action="store_true",
                    help="Skip the (slow) committee engine; SMA baseline only.")
    args = ap.parse_args()

    if args.long_daily:
        await run_long_daily(args)
        return
    if args.maxhold_sweep:
        await run_maxhold_sweep(args)
        return

    from src.backtest import run_backtest

    rt = round_trip_bps()
    max_hold_bars = int(settings.MAX_HOLD_HOURS)  # 1h bars
    print("=" * 108)
    print(f"BASELINE COMPARISON  fast={args.fast} slow={args.slow}  "
          f"round-trip cost={rt:.0f}bps (fee={settings.TX_COST_FEE_BPS:.0f} "
          f"slip={settings.TX_COST_SLIPPAGE_BPS:.0f} spread={settings.TX_COST_SPREAD_BPS:.0f})  "
          f"max_hold={settings.MAX_HOLD_HOURS:.0f}h ({max_hold_bars} bars)")
    print("variants (fixed up front): " + "  ".join(
        f"{k}(tp={VARIANTS[k]['tp']} sl={VARIANTS[k]['sl']} trail={VARIANTS[k]['trail']})"
        for k in VARIANTS))
    print("=" * 108)

    chosen = [v.strip() for v in args.variants.split(",") if v.strip()]

    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        bars = _fetch_bars_yf(sym, args.days, args.timeframe)
        if bars is None or len(bars) < args.slow + 10:
            print(f"\n{sym}: no usable data (got {0 if bars is None else len(bars)} bars) -- skipped")
            continue

        bars = bars.sort("t")
        closes = [float(c) for c in bars["close"].to_list()]
        times = [str(t) for t in bars["t"].to_list()]
        n = len(closes)
        warmup = int(n * args.warmup_frac)
        span_days = (bars["t"].to_list()[-1], bars["t"].to_list()[0])
        print(f"\n{sym}  ({n} bars, {span_days[1]} .. {span_days[0]}, warmup={warmup})")

        for vname in chosen:
            v = VARIANTS[vname]
            base = run_sma_backtest(
                sym, closes, times, args.fast, args.slow,
                v["tp"], v["sl"], rt, warmup,
                trail=v["trail"], max_hold_bars=max_hold_bars,
            )
            print(f"  [{vname:9}] SMA  :{fmt(base, n)}   exits[{reasons(base.trades)}]")

            split_i = warmup + int((n - warmup) * (1 - args.oos_split))
            te = [t for t in base.trades if t.exit_i >= split_i]
            if te:
                rr = Result(symbol=sym, trades=te)
                rr.total_return_pct, rr.max_drawdown_pct = rr.curve_stats()
                print(f"  [{vname:9}]  OOS :{fmt(rr, n)}   exits[{reasons(te)}]")

            if args.no_committee:
                continue
            try:
                kw = {"TX_COST_USE_DYNAMIC": False}
                if v["tp"] is not None:
                    kw["PROFIT_TARGET_PCT"] = v["tp"]
                kw["STOP_LOSS_PCT"] = v["sl"]
                kw["TRAILING_STOP_ENABLED"] = v["trail"] is not None
                if v["trail"] is not None:
                    kw["TRAILING_ACTIVATION_PCT"] = v["trail"][0]
                    kw["TRAILING_DISTANCE_PCT"] = v["trail"][1]
                # C1 "trailing-only" = no fixed target: push TP out of reach.
                if v["tp"] is None:
                    kw["PROFIT_TARGET_PCT"] = 0.5  # config max; SL fires first
                with override_settings(**kw):
                    cres = await run_backtest(symbol=sym, bars=bars, use_committee=True)
                cr = Result(symbol=sym)
                for t in cres.trades:
                    cr.trades.append(Trade(0, 0, t.entry_price, t.exit_price,
                                           t.pnl_pct - rt / 100, t.reason))
                print(f"  [{vname:9}] COMM :  trades={cres.n_trades:>3}  win={cres.win_rate:5.1f}%  "
                      f"exp={cr.expectancy_pct:+6.3f}%/trade  ret={cres.total_return_pct:+7.2f}%  "
                      f"maxDD={cres.max_drawdown_pct:6.2f}%  fees=${cres.total_fees_paid:.2f}  "
                      f"exits[{reasons(cr.trades)}]")
            except Exception as e:
                print(f"  [{vname:9}] COMM : FAILED ({type(e).__name__}: {e})")

    print("\n" + "=" * 108)
    print("Read the trade COUNT first: a few dozen trades is a small sample and the")
    print("per-trade numbers can be luck. Compare expectancy (exp, %/trade), which is")
    print("size-independent, on the OOS window. If no exit variant beats the SMA")
    print("baseline over the longest window, the entry signal -- not the exit -- is the problem.")


if __name__ == "__main__":
    asyncio.run(main())
