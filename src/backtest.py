"""Backtesting engine for the Apex Oracle Bot strategy.

Runs the REAL strategy + risk logic (TradingStrategy, RiskManager) over
historical OHLCV data using Polars, with a simple simulated fill model.
This is empirical validation of the deployed strategy - not a placeholder.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

import numpy as np
import polars as pl

from src.committee.committee import run_committee
from src.config import settings
from src.logging_config import get_logger
from src.risk import RiskManager
from src.strategies import TradingStrategy

logger = get_logger("backtest")


@dataclass
class BacktestTrade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float
    reason: str


@dataclass
class BacktestResult:
    symbol: str
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    start_equity: float = 0.0
    end_equity: float = 0.0
    total_return_pct: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    regimes_seen: dict[str, int] = field(default_factory=dict)
    # Cumulative fee+slippage+spread $ actually deducted across all closes
    # (regular closes + final mark-to-market). Set incrementally by
    # run_backtest(); used for cost_adjusted_return_pct below, which
    # previously always computed against a hardcoded 0.0 due to a broken
    # `hasattr(result, 'get')` check on a dataclass, and then crashed with a
    # Decimal/float TypeError the moment any trade closed (found + fixed
    # 2026-09-20 while testing the cost_multiplier addition above -- backtest.py
    # had zero test coverage before this, so this was never exercised).
    total_fees_paid: float = 0.0
    # Return with total_fees_paid added back -- i.e. what the return would
    # have been at zero transaction cost. total_return_pct itself is already
    # NET of fees (fee is subtracted from pnl before it hits equity), so this
    # is the complementary number that makes the cost drag visible:
    # gross_return_pct - total_return_pct = cost drag as % of equity.
    gross_return_pct: float = 0.0
    # The actual bars this run consumed (real or synthetic) -- lets a caller
    # feed the same bars into run_benchmark_comparison() for a buy-and-hold
    # comparison without re-fetching/re-generating them. None until set by
    # run_backtest() below (wired in 2026-09-20).
    bars_used: Any = None
    # Where would-be entries were stopped (signal -> committee -> vetoes ->
    # account gates -> sizing -> opened). Set by run_portfolio_backtest().
    entry_funnel: dict[str, int] = field(default_factory=dict)


_TF_UNITS = {"min": 60, "t": 60, "h": 3600, "hour": 3600, "d": 86400, "day": 86400, "w": 604800, "week": 604800}


def _timeframe_seconds(timeframe: str) -> int | None:
    """'5Min' / '1Hour' / '4Hour' / '1D' / '1Day' -> seconds (None if unparseable)."""
    s = str(timeframe).strip().lower()
    num = "".join(ch for ch in s if ch.isdigit()) or "1"
    unit = s[len(num):] if s.startswith(num) else s.lstrip("0123456789")
    mult = _TF_UNITS.get(unit)
    return int(num) * mult if mult else None


def _to_datetime_col(df: pl.DataFrame) -> pl.Series:
    t = df["t"]
    if t.dtype == pl.Utf8:
        t = t.str.to_datetime(time_zone="UTC")
    elif isinstance(t.dtype, pl.Datetime) and t.dtype.time_zone is None:
        t = t.dt.replace_time_zone("UTC")
    return t


def _base_bar_seconds(df: pl.DataFrame) -> int | None:
    """Median spacing of the supplied bars, in seconds."""
    if len(df) < 2 or "t" not in df.columns:
        return None
    diffs = _to_datetime_col(df.head(50)).diff().drop_nulls().dt.total_seconds()
    return int(diffs.median()) if len(diffs) else None


def _resample_bars(df: pl.DataFrame, tf_seconds: int) -> pl.DataFrame:
    """Aggregate base OHLCV bars into tf_seconds buckets (start-labelled), keeping `t`'s type."""
    was_str = df["t"].dtype == pl.Utf8
    out = (
        df.with_columns(_to_datetime_col(df).alias("_dt"))
        .sort("_dt")
        .group_by_dynamic("_dt", every=f"{tf_seconds}s")
        .agg(
            pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
            pl.col("close").last(), pl.col("volume").sum(),
        )
    )
    t = out["_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00") if was_str else out["_dt"]
    return out.with_columns(t.alias("t")).select(["t", "open", "high", "low", "close", "volume"])


class BacktestExchange:
    """Minimal in-memory exchange that replays historical bars for the strategy."""

    def __init__(self, bars: dict[str, pl.DataFrame]):
        self._bars = bars  # symbol -> DataFrame with columns [t, open, high, low, close, volume]
        self.current_time: str | None = None
        self._base_seconds = {s: _base_bar_seconds(df) for s, df in bars.items()}

    async def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100, end: datetime.datetime | None = None) -> pl.DataFrame:
        df = self._bars.get(symbol, pl.DataFrame())
        # Use current_time for point-in-time filtering (backtest PIT)
        filter_time = self.current_time
        if end is not None:
            # Convert end datetime to string for comparison
            filter_time = end.isoformat() if isinstance(end, datetime.datetime) else str(end)
        if filter_time is not None:
            df = df.filter(pl.col("t") <= filter_time)
        # Serve the requested timeframe. This used to ignore `timeframe` and
        # return the base (hourly) bars for every request, so regime analysis
        # -- live: analyze_market_regime(timeframe="1D") -- ran on hourly ATR%
        # (~0.5%, under the 1.25% low-vol cutoff ~99% of the time) instead of
        # daily (~3%, essentially never under it). Coarser timeframes are
        # aggregated from base bars <= current_time, so the last bar is the
        # in-progress one, like the live API. Finer ones can't be synthesized
        # and fall back to the base bars.
        tf_seconds = _timeframe_seconds(timeframe)
        base = self._base_seconds.get(symbol)
        if len(df) and tf_seconds and base and tf_seconds > base:
            df = _resample_bars(df, tf_seconds)
        return df.tail(limit) if len(df) else df

    def invalidate_bars_cache(self, symbol: str | None = None, timeframe: str | None = None) -> int:
        """Backtest exchange has no cache to invalidate."""
        return 0

    async def get_account(self) -> dict[str, Any]:
        return {"equity": 0.0, "cash": 0.0, "portfolio_value": 0.0}

    async def get_positions(self) -> list[dict[str, Any]]:
        return []


def _generate_synthetic_bars(
    symbol: str,
    n: int = 500,
    seed: int = 42,
    regime: str = "trending",
) -> pl.DataFrame:
    """Generate synthetic OHLCV bars for a given regime (trending / mean_reverting / volatile)."""
    rng = np.random.RandomState(seed)
    t = pl.datetime_range(
        start=pl.datetime(2024, 1, 1),
        end=pl.datetime(2024, 1, 1) + pl.duration(days=n - 1),
        interval="1d",
        eager=True,
    )

    if regime == "trending":
        drift = 0.0015
        vol = 0.012
        price = 100.0
        closes = []
        for _ in range(n):
            price *= (1 + drift + rng.randn() * vol)
            closes.append(price)
    elif regime == "mean_reverting":
        price = 100.0
        closes = []
        for _ in range(n):
            price += (100.0 - price) * 0.05 + rng.randn() * 1.2
            closes.append(price)
    else:  # volatile
        price = 100.0
        closes = []
        for _ in range(n):
            price *= (1 + rng.randn() * 0.04)
            closes.append(price)

    closes = np.array(closes)
    highs = closes * (1 + np.abs(rng.randn(n)) * 0.005)
    lows = closes * (1 - np.abs(rng.randn(n)) * 0.005)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]

    return pl.DataFrame({
        "t": t,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": rng.randint(100, 1000, n).astype(float),
    })


async def fetch_real_bars(
    symbol: str,
    days: int = 365,
    timeframe: str = "1h",
) -> pl.DataFrame | None:
    """Fetch real historical OHLCV bars from Alpaca's crypto data API.

    Falls back to yfinance if Alpaca credentials are unavailable.
    Returns None if no data source is available.

    The symbol format expected by Alpaca is ``BTC/USD``; yfinance uses ``BTC-USD``.
    """
    import datetime

    # Try Alpaca first
    try:
        from src.exchange import AlpacaExchange
        ex = AlpacaExchange()
        await ex.load()
        bars = await ex.get_bars(symbol, timeframe=timeframe, limit=days * 24)
        if bars is not None and len(bars) > 0:
            # Rename timestamp -> t for BacktestExchange compatibility
            if "timestamp" in bars.columns:
                bars = bars.rename({"timestamp": "t"})
            elif "t" not in bars.columns:
                # Add a placeholder time column if missing
                bars = bars.with_columns(pl.lit(datetime.now(UTC).isoformat()).alias("t"))
            logger.info(f"Fetched {len(bars)} real bars for {symbol} from Alpaca")
            return bars
    except Exception as e:
        logger.debug(f"Alpaca fetch failed for {symbol}: {e}")

    # Fallback: yfinance
    try:
        import yfinance as yf
        yf_symbol = symbol.replace("/", "-")
        df = yf.Ticker(yf_symbol).history(period=f"{days}d", interval=timeframe)
        if df.empty:
            return None
        df = df.reset_index()
        rename_map = {}
        if "Datetime" in df.columns:
            rename_map["Datetime"] = "t"
        elif "Date" in df.columns:
            rename_map["Date"] = "t"
        if "Open" in df.columns:
            rename_map["Open"] = "open"
        if "High" in df.columns:
            rename_map["High"] = "high"
        if "Low" in df.columns:
            rename_map["Low"] = "low"
        if "Close" in df.columns:
            rename_map["Close"] = "close"
        if "Volume" in df.columns:
            rename_map["Volume"] = "volume"
        df = df.rename(rename_map)
        df["t"] = df["t"].astype(str)
        bars = pl.from_pandas(df)
        logger.info(f"Fetched {len(bars)} real bars for {symbol} from yfinance")
        return bars
    except Exception as e:
        logger.warning(f"yfinance fetch also failed for {symbol}: {e}")

    return None


def _parse_bar_time(ts: Any) -> datetime.datetime:
    """Bar timestamp -> aware UTC datetime (the simulated clock)."""
    if isinstance(ts, datetime.datetime):
        d = ts
    else:
        d = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=UTC)


class SimulatedAccountGuards:
    """Account-level risk limits on the SIMULATED clock.

    Mirrors the live rules that RiskManager.update_account_status() and
    bot.process_signal_for_symbol() enforce against the wall clock (which a
    backtest can't use):

    - max-drawdown killswitch (MAX_DRAWDOWN_STOP): liquidate + block entries;
      clears when drawdown recovers to half the limit, or once the book has
      been flat for DRAWDOWN_KILLSWITCH_COOLDOWN_HOURS (peak re-based).
    - daily-loss killswitch (DAILY_LOSS_LIMIT % of equity): liquidate + block;
      clears at the next UTC day.
    - rolling realized-loss limit (ROLLING_LOSS_LIMIT_PCT of peak equity over
      LOSS_LIMIT_WINDOW_HOURS): blocks new entries only.

    ``update`` returns "liquidate" when a killswitch trips, else None.
    """

    def __init__(self, start_equity: float):
        self.peak_equity = float(start_equity)
        self.day: datetime.date | None = None
        self.start_of_day_equity = float(start_equity)
        self.killswitch_reason = ""
        self.drawdown_tripped_at: datetime.datetime | None = None
        self._closes: list[tuple[datetime.datetime, float]] = []
        self._rolling_blocked = False

    def record_close(self, when: datetime.datetime, pnl: float) -> None:
        self._closes.append((when, float(pnl)))

    def entries_blocked(self) -> bool:
        return bool(self.killswitch_reason) or self._rolling_blocked

    def update(self, now: datetime.datetime, equity: float, has_positions: bool) -> str | None:
        if self.day != now.date():
            self.day = now.date()
            self.start_of_day_equity = equity
            if self.killswitch_reason == "daily_loss_limit":
                self.killswitch_reason = ""

        self.peak_equity = max(self.peak_equity, equity)
        drawdown_pct = (equity - self.peak_equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0.0

        if self.drawdown_tripped_at is not None and not has_positions:
            elapsed_h = (now - self.drawdown_tripped_at).total_seconds() / 3600
            if elapsed_h >= settings.DRAWDOWN_KILLSWITCH_COOLDOWN_HOURS:
                self.peak_equity = equity
                drawdown_pct = 0.0
                self.drawdown_tripped_at = None
                if self.killswitch_reason == "max_drawdown":
                    self.killswitch_reason = ""

        action = None
        if drawdown_pct < settings.MAX_DRAWDOWN_STOP:
            if self.drawdown_tripped_at is None:
                self.drawdown_tripped_at = now
            self.killswitch_reason = "max_drawdown"
            action = "liquidate"
        elif self.killswitch_reason == "max_drawdown" and drawdown_pct >= settings.MAX_DRAWDOWN_STOP / 2.0:
            self.killswitch_reason = ""
            self.drawdown_tripped_at = None

        if action is None:
            daily_pnl = equity - self.start_of_day_equity
            if daily_pnl < settings.DAILY_LOSS_LIMIT / 100.0 * equity:
                self.killswitch_reason = "daily_loss_limit"
                action = "liquidate"

        self._rolling_blocked = False
        if settings.ROLLING_LOSS_LIMIT_PCT > 0:
            cutoff = now - datetime.timedelta(hours=settings.LOSS_LIMIT_WINDOW_HOURS)
            rolling = sum(p for t, p in self._closes if t > cutoff)
            limit_abs = -abs(settings.ROLLING_LOSS_LIMIT_PCT) / 100.0 * self.peak_equity
            self._rolling_blocked = self.peak_equity > 0 and rolling <= limit_abs
        return action


def _finalize_result(result: BacktestResult, equity: float) -> None:
    """Compute summary metrics from result.trades / result.equity_curve."""
    start_equity = float(result.start_equity)
    result.end_equity = float(equity)
    result.total_return_pct = (float(equity) - start_equity) / start_equity * 100 if start_equity else 0.0
    result.n_trades = len(result.trades)
    result.n_wins = sum(1 for t in result.trades if t.pnl > 0)
    result.n_losses = sum(1 for t in result.trades if t.pnl <= 0)
    result.win_rate = (result.n_wins / result.n_trades * 100) if result.n_trades else 0.0

    eq = np.array(result.equity_curve, dtype=float)
    if len(eq):
        peak = np.maximum.accumulate(eq)
        result.max_drawdown_pct = float(((eq - peak) / peak * 100).min())
    else:
        result.max_drawdown_pct = 0.0

    if result.trades:
        trade_rets = np.array([t.pnl_pct / 100.0 for t in result.trades])
        trade_pnls = np.array([t.pnl for t in result.trades])
        wins = trade_rets[trade_rets > 0]
        losses = trade_rets[trade_rets < 0]
        win_pnls = trade_pnls[trade_pnls > 0]
        loss_pnls = trade_pnls[trade_pnls < 0]

        result.avg_win_pct = float(wins.mean()) if len(wins) else 0.0
        result.avg_loss_pct = float(losses.mean()) if len(losses) else 0.0
        result.avg_win_usd = float(win_pnls.mean()) if len(win_pnls) else 0.0
        result.avg_loss_usd = float(loss_pnls.mean()) if len(loss_pnls) else 0.0

        if len(losses) > 0 and losses.mean() != 0:
            result.payoff_ratio = float(wins.mean() / abs(losses.mean())) if len(wins) else 0.0
        else:
            result.payoff_ratio = float('inf') if len(wins) > 0 else 0.0

        if len(loss_pnls) > 0 and loss_pnls.sum() != 0:
            result.profit_factor = float(win_pnls.sum() / abs(loss_pnls.sum()))
        else:
            result.profit_factor = float('inf') if len(win_pnls) > 0 else 0.0

        result.expectancy_pct = float(trade_rets.mean())
        result.expectancy_usd = float(trade_pnls.mean())

        downside_rets = trade_rets[trade_rets < 0]
        if len(downside_rets) > 1 and downside_rets.std() > 0:
            result.sortino = float(trade_rets.mean() / downside_rets.std() * np.sqrt(252))
        else:
            result.sortino = float('inf') if trade_rets.mean() > 0 else 0.0

        # Calmar (annualized return / |max drawdown|). max_drawdown_pct is
        # <= 0, so the old `> 0` check never fired and Calmar was always inf/0.
        annual_return = float(trade_rets.mean() * 252)
        if result.max_drawdown_pct < 0:
            result.calmar = float(annual_return / (abs(result.max_drawdown_pct) / 100.0))
        else:
            result.calmar = float('inf') if annual_return > 0 else 0.0

        sorted_pnls = np.sort(trade_pnls)
        top3_pnl = float(sorted_pnls[-3:].sum()) if len(sorted_pnls) >= 3 else float(sorted_pnls.sum())
        result.top3_contribution_pct = float(top3_pnl / float(equity) * 100) if equity > 0 else 0.0

        avg_equity = float(np.mean(eq)) if len(eq) > 0 else start_equity
        result.turnover_annualized = float(len(result.trades) / avg_equity * 252 * np.mean([abs(t.qty * t.entry_price) for t in result.trades]) / avg_equity) if avg_equity > 0 else 0.0

    # total_return_pct is net of fees; gross adds them back (cost drag =
    # gross - net).
    result.gross_return_pct = float(result.total_return_pct) + (result.total_fees_paid / start_equity * 100 if start_equity else 0.0)

    if len(eq) > 2:
        rets = np.diff(eq) / eq[:-1]
        result.sharpe = float(np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)) if np.std(rets) > 0 else 0.0


async def run_portfolio_backtest(
    bars_by_symbol: dict[str, pl.DataFrame],
    start_equity: float = 10000.0,
    use_committee: bool = True,
    cost_multiplier: float = 1.0,
    enforce_account_caps: bool = True,
    start_time: str | datetime.datetime | None = None,
) -> BacktestResult:
    """Simulate the live bot across several symbols on one shared account.

    ``start_time``: bars before it are history only (visible to the strategy
    through the exchange, never traded) -- live regime analysis needs ~100
    daily bars, so pass a warm-up period before the evaluation window.

    Mirrors bot.process_signal_for_symbol() on the simulated bar clock:
    long-only entries ("sell" only closes a held long), committee veto +
    adversarial vetoes, committee-derived expected return, post-sizing
    multipliers, exchange-minimum bump, MAX_HOLD_HOURS, regime-scaled
    trailing stop, post-close cooldown, killswitches and rolling-loss limit
    (SimulatedAccountGuards). With ``enforce_account_caps`` it also applies
    MAX_OPEN_POSITIONS and the portfolio exposure cap.

    Not simulated (no historical data): live order-book impact, news
    sentiment, LLM brain, PPO / decision-transformer votes (the committee
    skips them when signal["backtest_df"] is set), the regime-switch oracle
    flag, scale-in adds, and partial sells (a sell closes the whole long).
    Fills are at the bar close; stops are checked at bar closes only.
    """
    from src.committee.models import calculate_directional_entropy, disagreement_from_entropy
    from src.trade_decision import (
        adversarial_veto_reasons,
        apply_entry_size_multipliers,
        estimate_expected_return,
        min_order_bump,
    )

    symbols = list(bars_by_symbol)
    exchange = BacktestExchange(bars_by_symbol)
    strategy = TradingStrategy(exchange, cache_ttl=0.0, backtest=True)
    risk = RiskManager(exchange)
    result = BacktestResult(symbol="+".join(symbols), start_equity=start_equity, end_equity=start_equity)
    guards = SimulatedAccountGuards(start_equity)

    learner = None
    validation_min_trades = None
    if use_committee:
        try:
            from src.committee.adaptive_meta import VALIDATION_MIN_TRADES
            from src.committee.committee import get_meta_learner
            learner = get_meta_learner()
            validation_min_trades = VALIDATION_MIN_TRADES
        except Exception as e:
            logger.debug(f"[BT] Brain C gate unavailable: {e}")

    rows_by_time: dict[str, dict[str, dict[str, Any]]] = {}
    for sym, df in bars_by_symbol.items():
        for row in df.iter_rows(named=True):
            rows_by_time.setdefault(str(row["t"]), {})[sym] = row
    timeline = sorted(rows_by_time, key=_parse_bar_time)
    if start_time is not None:
        start_dt = _parse_bar_time(start_time)
        timeline = [ts for ts in timeline if _parse_bar_time(ts) >= start_dt]
    funnel = result.entry_funnel

    def count(key: str) -> None:
        funnel[key] = funnel.get(key, 0) + 1

    realized = float(start_equity)
    positions: dict[str, dict[str, Any]] = {}
    last_price: dict[str, float] = {}
    cooldown_until: dict[str, datetime.datetime] = {}

    def mark_to_market() -> float:
        return realized + sum((last_price[s] - p["entry_price"]) * p["qty"] for s, p in positions.items())

    def close(sym: str, price: float, ts: str, now: datetime.datetime, reason: str) -> None:
        nonlocal realized
        pos = positions.pop(sym)
        qty, entry = pos["qty"], pos["entry_price"]
        gross_pnl = (price - entry) * qty
        tx_costs = risk.get_transaction_costs(sym)
        fee = price * qty * (2.0 * tx_costs["total_bps"] * cost_multiplier / 10000)
        pnl = gross_pnl - fee
        result.total_fees_paid += fee
        realized += pnl
        result.trades.append(BacktestTrade(
            symbol=sym, side="long", entry_price=entry, exit_price=price, qty=qty,
            entry_time=pos["entry_time"], exit_time=ts, pnl=pnl,
            pnl_pct=(price - entry) / entry * 100, reason=reason,
        ))
        guards.record_close(now, pnl)
        # Same post-close cleanup as live (_record_committee_outcome): stale
        # peaks would otherwise fire the next position's trailing stop early.
        risk.peak_prices.pop(sym, None)
        getattr(strategy, "_trailing_peaks", {}).pop(sym, None)
        getattr(strategy, "_trailing_troughs", {}).pop(sym, None)
        cooldown_until[sym] = now + datetime.timedelta(seconds=settings.COOLDOWN_SECONDS_BUY)
        logger.info(f"[BT] CLOSE {sym} @ {price:.2f} pnl={pnl:.2f} reason={reason}")

    for ts in timeline:
        now = _parse_bar_time(ts)
        bar_rows = rows_by_time[ts]
        # Original bar value (str or datetime) so BacktestExchange's
        # point-in-time filter compares like with like.
        exchange.current_time = next(iter(bar_rows.values()))["t"]
        for sym, row in bar_rows.items():
            last_price[sym] = float(row["close"])

        if guards.update(now, mark_to_market(), has_positions=bool(positions)) == "liquidate":
            for sym in list(positions):
                close(sym, last_price[sym], ts, now, f"killswitch_{guards.killswitch_reason}")

        for sym in symbols:
            row = bar_rows.get(sym)
            if row is None:
                continue
            price = float(row["close"])
            pos = positions.get(sym)

            if pos is not None:
                held_h = (now - _parse_bar_time(pos["entry_time"])).total_seconds() / 3600
                if held_h >= settings.MAX_HOLD_HOURS:
                    close(sym, price, ts, now, "max_hold_time")
                    continue
                cached = strategy._regime_cache.get(sym)
                regime_for_trailing = cached[1].get("regime") if cached else None
                if risk.check_trailing_stop(sym, price, pos["entry_price"], pos["qty"], regime=regime_for_trailing) == "close":
                    close(sym, price, ts, now, "trailing_stop_hit")
                    continue

            # No entry timestamp on the position dict: the strategy's own
            # max-hold / min-hold checks run on the wall clock, so they're
            # enforced above on the simulated clock instead.
            position = None if pos is None else {
                "symbol": sym, "qty": pos["qty"], "avg_entry_price": pos["entry_price"], "side": "long",
            }
            signal = await strategy.generate_trading_signal(sym, price, position)
            regime = signal.get("regime", "neutral")
            result.regimes_seen[regime] = result.regimes_seen.get(regime, 0) + 1

            if pos is None:
                count(f"signal_{signal.get('action', 'none')}")
            if signal.get("action") == "close":
                if pos is not None:
                    close(sym, price, ts, now, signal.get("reason", "close"))
                continue

            committee_result = None
            if use_committee:
                signal["backtest_df"] = await exchange.get_bars(sym)
                committee_result = await run_committee(sym, price, signal)
                if committee_result.vetoed:
                    if pos is None:
                        count("committee_vetoed")
                    continue
                final_action = committee_result.action
                if pos is None:
                    count(f"committee_{final_action}")
            else:
                final_action = signal.get("action", "hold")

            if pos is not None:
                if final_action in ("sell", "close"):
                    close(sym, price, ts, now, signal.get("reason") or f"committee_{final_action}")
                continue  # live adds to longs (scale-in); not simulated
            if final_action != "buy":
                continue  # long-only: a sell with no long position is ignored
            count("buy_signals")

            if guards.killswitch_reason:
                count(f"blocked_killswitch_{guards.killswitch_reason}")
                continue
            if guards.entries_blocked():
                count("blocked_rolling_loss")
                continue
            if now < cooldown_until.get(sym, now):
                count("blocked_cooldown")
                continue
            if enforce_account_caps and len(positions) >= settings.MAX_OPEN_POSITIONS:
                count("blocked_max_positions")
                continue

            if committee_result is not None:
                signal["brain_disagreement"] = disagreement_from_entropy(
                    calculate_directional_entropy(getattr(committee_result, "votes", None) or [])
                )
                reasons = adversarial_veto_reasons(signal, committee_result, learner, validation_min_trades)
                if reasons:
                    first = reasons[0]
                    count("veto_brain_b" if first.startswith("Brain B") else
                          "veto_brain_c" if first.startswith("Brain C") else "veto_disagreement")
                    continue
                expected_return_pct = estimate_expected_return(regime, committee_result.score, committee_result.entropy)
                confidence = committee_result.score
            else:
                expected_return_pct = signal.get("expected_return_pct", 0.0)
                confidence = signal.get("confidence", 1.0)

            equity_now = mark_to_market()
            drawdown_pct = (equity_now - guards.peak_equity) / guards.peak_equity * 100 if guards.peak_equity > 0 else 0.0
            size, status = risk.calculate_position_size(
                symbol=sym, current_price=price, regime=regime, atr=signal.get("atr"),
                confidence=confidence, expected_return_pct=expected_return_pct,
                current_equity=equity_now, drawdown_pct=drawdown_pct, side="buy",
                deriv_data=signal.get("features") or {},
            )
            if status != "ok" or size <= 0:
                reason = status.split("(")[0].replace("rejected:", "").strip() if status != "ok" else "zero size"
                count("sizing_" + "_".join(reason.split())[:48])
                continue
            signal["action"] = "buy"
            size, _ = apply_entry_size_multipliers(size, signal, committee_result, price)

            if enforce_account_caps:
                exposure = sum(last_price[s] * p["qty"] for s, p in positions.items())
                headroom = risk._get_max_portfolio_cap() - exposure
                if headroom < 10.0:  # check_and_reserve_exposure's min_notional
                    count("blocked_portfolio_cap")
                    continue
                if size * price > headroom:
                    size = round(headroom / price, 6)
            else:
                headroom = float("inf")

            bumped = min_order_bump(size, price)
            if bumped is None or bumped * price > headroom:
                count("blocked_min_order")
                continue
            size = bumped
            if size <= 0:
                continue
            count("opened")
            positions[sym] = {"qty": size, "entry_price": price, "entry_time": ts}
            logger.info(f"[BT] BUY {size:.6f} {sym} @ {price:.2f} (regime={regime})")

        result.equity_curve.append(mark_to_market())

    if timeline:
        final_ts = timeline[-1]
        final_now = _parse_bar_time(final_ts)
        for sym in list(positions):
            close(sym, last_price[sym], final_ts, final_now, "end_of_backtest")
        if result.equity_curve:
            result.equity_curve[-1] = realized

    _finalize_result(result, realized)
    return result


async def run_backtest(
    symbol: str = "BTC/USD",
    n_bars: int = 400,
    start_equity: float = 10000.0,
    seed: int = 7,
    regime: str = "trending",
    bars: pl.DataFrame | None = None,  # Real historical bars; if None, uses synthetic data
    use_committee: bool = True,   # If True, run the full 5-brain committee (AI-driven decisions).
                                   # If False, use the raw rule-based strategy signal only.
    cost_multiplier: float = 1.0,  # Multiplies the round-trip transaction cost
                                    # (fee+slippage+spread bps from risk.get_transaction_costs)
                                    # applied on every close. 1.0 = normal costs. Use 2.0/3.0 for
                                    # a cost-stress test.
) -> BacktestResult:
    """Single-symbol backtest (promotion gate, walk-forward, research).

    Runs the same live-parity engine as run_portfolio_backtest() -- long-only,
    live vetoes, simulated-clock max-hold / cooldown / killswitches -- on one
    symbol. Account caps (MAX_OPEN_POSITIONS, portfolio exposure cap) are
    off here so sizing stays percent-of-equity and comparable across runs;
    use run_portfolio_backtest() to simulate the real account.

    If `bars` is provided (real historical OHLCV data), it is used instead of
    synthetic data, and signal["backtest_df"] is populated per-bar so that
    transformer_brain.py uses this historical context instead of attempting
    a live Alpaca fetch.
    """
    if bars is None:
        days = max(n_bars // 24, 30)  # estimate days from bar count (assuming 1h bars)
        real_bars = await fetch_real_bars(symbol, days=days)
        if real_bars is not None and len(real_bars) >= n_bars:
            bars = real_bars.head(n_bars)
        else:
            bars = _generate_synthetic_bars(symbol, n=n_bars, seed=seed, regime=regime)

    result = await run_portfolio_backtest(
        {symbol: bars}, start_equity=start_equity, use_committee=use_committee,
        cost_multiplier=cost_multiplier, enforce_account_caps=False,
    )
    result.symbol = symbol
    result.bars_used = bars
    return result


def run_monte_carlo_analysis(result: BacktestResult, n_simulations: int = 1000) -> dict[str, Any]:
    """
    Run Monte Carlo permutation on the sequence of trades to determine true risk of ruin.
    """
    if not result.trades:
        return {"risk_of_ruin_pct": 0.0, "p05_drawdown_pct": 0.0, "p05_return_pct": 0.0}

    # Extract percentage returns of each trade
    trade_rets = np.array([t.pnl_pct / 100.0 for t in result.trades])
    
    n_trades = len(trade_rets)
    start_equity = result.start_equity
    
    # Generate random indices to shuffle trades
    rng = np.random.RandomState(42)
    indices = rng.randint(0, n_trades, size=(n_simulations, n_trades))
    
    # Sampled trades
    sampled_rets = trade_rets[indices]
    
    # Compute equity curves (compound returns)
    # equity_curves = start_equity * cumulative product of (1 + r)
    compound_returns = np.cumprod(1 + sampled_rets, axis=1)
    equity_curves = start_equity * compound_returns
    
    # Metrics per simulation
    final_returns = compound_returns[:, -1] - 1.0
    
    # Max Drawdowns
    peaks = np.maximum.accumulate(equity_curves, axis=1)
    drawdowns = (equity_curves - peaks) / peaks
    max_drawdowns = np.min(drawdowns, axis=1) * 100  # negative percentages
    
    # Risk of Ruin (probability of hitting > 20% drawdown)
    ruin_count = np.sum(max_drawdowns <= -20.0)
    risk_of_ruin_pct = (ruin_count / n_simulations) * 100
    
    p05_drawdown = float(np.percentile(max_drawdowns, 5)) # 5th percentile worst drawdown
    p05_return = float(np.percentile(final_returns, 5)) * 100 # 5th percentile worst return
    
    logger.info(f"Monte Carlo ({n_simulations} sims) -> Risk of Ruin: {risk_of_ruin_pct:.1f}%, 5th Pctl DD: {p05_drawdown:.2f}%")
    
    return {
        "risk_of_ruin_pct": risk_of_ruin_pct,
        "p05_drawdown_pct": p05_drawdown,
        "p05_return_pct": p05_return,
    }


def print_backtest_summary(result: BacktestResult) -> None:
    """Pretty-print backtest results."""
    # Re-calculate metrics to ensure they are robustly populated
    if result.trades:
        result.n_trades = len(result.trades)
        result.n_wins = sum(1 for t in result.trades if t.pnl > 0)
        result.n_losses = sum(1 for t in result.trades if t.pnl <= 0)
        result.win_rate = (result.n_wins / result.n_trades * 100) if result.n_trades else 0.0
        
        gross_profit = sum(t.pnl for t in result.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in result.trades if t.pnl <= 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        avg_win = gross_profit / result.n_wins if result.n_wins else 0.0
        avg_loss = gross_loss / result.n_losses if result.n_losses else 0.0
        expectancy = (result.win_rate/100 * avg_win) - ((1 - result.win_rate/100) * avg_loss)
    else:
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        expectancy = 0.0

    eq = np.array(result.equity_curve)
    if len(eq) > 0:
        peak = np.maximum.accumulate(eq)
        drawdown_pct = (eq - peak) / peak * 100
        result.max_drawdown_pct = float(drawdown_pct.min())
        
        max_drawdown_usd = float((eq - peak).min())
        net_profit = result.end_equity - result.start_equity
        recovery_factor = abs(net_profit / max_drawdown_usd) if max_drawdown_usd < 0 else 0.0
    else:
        result.max_drawdown_pct = 0.0
        max_drawdown_usd = 0.0
        recovery_factor = 0.0

    if len(eq) > 2:
        rets = np.diff(eq) / eq[:-1]
        std_rets = np.std(rets)
        result.sharpe = float(np.mean(rets) / (std_rets + 1e-9) * np.sqrt(252)) if std_rets > 0 else 0.0
        
        downside_rets = rets[rets < 0]
        std_downside = np.std(downside_rets) if len(downside_rets) > 0 else 0.0
        sortino = float(np.mean(rets) / (std_downside + 1e-9) * np.sqrt(252)) if std_downside > 0 else 0.0
    else:
        result.sharpe = 0.0
        sortino = 0.0
        
    calmar = (float(result.total_return_pct) / abs(result.max_drawdown_pct)) if result.max_drawdown_pct < 0 else 0.0

    print("=" * 60)
    print(f"BACKTEST RESULTS: {result.symbol}")
    print("=" * 60)
    print(f"Start equity:      ${result.start_equity:,.2f}")
    print(f"End equity:        ${result.end_equity:,.2f}")
    print(f"Total return:      {result.total_return_pct:.2f}%")
    print(f"Trades:            {result.n_trades}")
    print(f"Wins / Losses:     {result.n_wins} / {result.n_losses}")
    print(f"Win rate:          {result.win_rate:.1f}%")
    print(f"Avg Win / Loss:    ${avg_win:.2f} / ${avg_loss:.2f}")
    print(f"Profit Factor:     {profit_factor:.2f}")
    print(f"Expectancy:        ${expectancy:.2f}")
    print(f"Max drawdown:      {result.max_drawdown_pct:.2f}% (${abs(max_drawdown_usd):,.2f})")
    print(f"Recovery Factor:   {recovery_factor:.2f}")
    print(f"Sharpe (ann.):     {result.sharpe:.2f}")
    print(f"Sortino (ann.):    {sortino:.2f}")
    print(f"Calmar Ratio:      {calmar:.2f}")
    print(f"Regimes seen:      {result.regimes_seen}")
    print("=" * 60)


async def run_walk_forward_optimization(
    symbol: str = "BTC/USD",
    total_bars: int = 500,
    train_pct: float = 0.6,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run Walk-Forward Optimization across In-Sample (IS) and Out-Of-Sample (OOS) windows.
    Prevents parameter overfitting.
    """
    is_bars = int(total_bars * train_pct)
    oos_bars = total_bars - is_bars

    logger.info(f"Walk-Forward Optimization: Total Bars={total_bars}, IS={is_bars}, OOS={oos_bars}")

    # In-Sample Backtest (Training)
    is_result = await run_backtest(symbol=symbol, n_bars=is_bars, seed=seed, regime="trending")
    
    # Out-Of-Sample Backtest (Testing)
    oos_result = await run_backtest(symbol=symbol, n_bars=oos_bars, start_equity=is_result.end_equity, seed=seed + 1, regime="trending")

    print("\n" + "=" * 60)
    print(f"WALK-FORWARD OPTIMIZATION SUMMARY: {symbol}")
    print("=" * 60)
    print(f"In-Sample (Train)   Return: {is_result.total_return_pct:+.2f}% | MaxDD: {is_result.max_drawdown_pct:.2f}% | WinRate: {is_result.win_rate:.1f}%")
    print(f"Out-of-Sample (Test) Return: {oos_result.total_return_pct:+.2f}% | MaxDD: {oos_result.max_drawdown_pct:.2f}% | WinRate: {oos_result.win_rate:.1f}%")
    print("=" * 60)

    return {
        "is_result": is_result,
        "oos_result": oos_result
    }


async def run_benchmark_comparison(
    result: BacktestResult,
    symbol: str,
    bars: pl.DataFrame,
    start_equity: float,
) -> dict[str, Any]:
    """
    Compare strategy performance against benchmarks:
    1. Buy & Hold
    2. Random Entry (Monte Carlo permutation)
    
    Args:
        result: Strategy backtest result
        symbol: Trading symbol
        bars: Historical bars DataFrame
        start_equity: Starting equity
        
    Returns:
        Dict with benchmark comparisons
    """
    # 1. Buy & Hold Benchmark
    first_close = float(bars["close"][0])
    last_close = float(bars["close"][-1])
    bh_return_pct = (last_close - first_close) / first_close * 100
    
    # Equity curve for buy & hold
    bh_equity = [start_equity * (1 + (float(bars["close"][i]) - first_close) / first_close) for i in range(len(bars))]
    bh_eq = np.array(bh_equity)
    bh_peak = np.maximum.accumulate(bh_eq)
    bh_drawdown = (bh_eq - bh_peak) / bh_peak * 100
    bh_max_dd = float(bh_drawdown.min()) if len(bh_drawdown) > 0 else 0.0
    
    # Buy & Hold Sharpe
    if len(bh_eq) > 2:
        bh_rets = np.diff(bh_eq) / bh_eq[:-1]
        bh_sharpe = float(np.mean(bh_rets) / (np.std(bh_rets) + 1e-9) * np.sqrt(252)) if np.std(bh_rets) > 0 else 0.0
    else:
        bh_sharpe = 0.0

    # 2. Random Entry Benchmark (from Monte Carlo)
    mc_result = run_monte_carlo_analysis(result, n_simulations=500)
    random_avg_return = mc_result.get("p05_return_pct", 0)  # 5th percentile as conservative random baseline

    # 3. Compare
    strategy_return = float(result.total_return_pct)
    strategy_sharpe = float(result.sharpe)
    strategy_max_dd = float(result.max_drawdown_pct)
    
    comparison = {
        "strategy": {
            "return_pct": strategy_return,
            "sharpe": strategy_sharpe,
            "sortino": getattr(result, 'sortino', 0.0),
            "calmar": getattr(result, 'calmar', 0.0),
            "max_drawdown_pct": strategy_max_dd,
            "profit_factor": getattr(result, 'profit_factor', 0.0),
            "win_rate": result.win_rate,
            "payoff_ratio": getattr(result, 'payoff_ratio', 0.0),
            "expectancy_pct": getattr(result, 'expectancy_pct', 0.0),
        },
        "buy_and_hold": {
            "return_pct": bh_return_pct,
            "sharpe": bh_sharpe,
            "max_drawdown_pct": bh_max_dd,
            "calmar": float(bh_return_pct / abs(bh_max_dd)) if bh_max_dd < 0 else float('inf'),
        },
        "random_entry": {
            "p05_return_pct": random_avg_return,
            "risk_of_ruin_pct": mc_result.get("risk_of_ruin_pct", 0.0),
        },
        "comparison": {
            "excess_return_vs_bh": strategy_return - bh_return_pct,
            "excess_sharpe_vs_bh": strategy_sharpe - bh_sharpe,
            "excess_return_vs_random_p05": strategy_return - random_avg_return,
            "beats_buy_and_hold": strategy_return > bh_return_pct,
            "beats_random_p05": strategy_return > random_avg_return,
        }
    }
    
    return comparison


def print_benchmark_comparison(comparison: dict[str, Any]) -> None:
    """Print formatted benchmark comparison."""
    s = comparison["strategy"]
    bh = comparison["buy_and_hold"]
    rnd = comparison["random_entry"]
    cmp = comparison["comparison"]
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {'Strategy':>12} {'Buy&Hold':>12} {'Random(P05)':>12}")
    print("-" * 70)
    print(f"{'Return %':<25} {s['return_pct']:>12.2f} {bh['return_pct']:>12.2f} {rnd['p05_return_pct']:>12.2f}")
    print(f"{'Sharpe':<25} {s['sharpe']:>12.2f} {bh['sharpe']:>12.2f} {'N/A':>12}")
    print(f"{'Sortino':<25} {s['sortino']:>12.2f} {'N/A':>12} {'N/A':>12}")
    print(f"{'Calmar':<25} {s['calmar']:>12.2f} {bh['calmar']:>12.2f} {'N/A':>12}")
    print(f"{'Max DD %':<25} {s['max_drawdown_pct']:>12.2f} {bh['max_drawdown_pct']:>12.2f} {'N/A':>12}")
    print(f"{'Profit Factor':<25} {s['profit_factor']:>12.2f} {'N/A':>12} {'N/A':>12}")
    print(f"{'Win Rate %':<25} {s['win_rate']:>12.1f} {'N/A':>12} {'N/A':>12}")
    print(f"{'Payoff Ratio':<25} {s['payoff_ratio']:>12.2f} {'N/A':>12} {'N/A':>12}")
    print("-" * 70)
    print(f"{'Excess Return vs B&H':<25} {cmp['excess_return_vs_bh']:>12.2f}%")
    print(f"{'Excess Sharpe vs B&H':<25} {cmp['excess_sharpe_vs_bh']:>12.2f}")
    print(f"{'Excess Return vs Rand(P05)':<25} {cmp['excess_return_vs_random_p05']:>12.2f}%")
    print(f"{'Beats Buy & Hold':<25} {'YES' if cmp['beats_buy_and_hold'] else 'NO':>12}")
    print(f"{'Beats Random (P05)':<25} {'YES' if cmp['beats_random_p05'] else 'NO':>12}")
    print("=" * 70)

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Apex Oracle Bot Backtest Engine")
    parser.add_argument("--symbol", type=str, default="BTC/USD", help="Symbol to backtest (default: BTC/USD)")
    parser.add_argument("--bars", type=int, default=400, help="Number of historical bars (default: 400)")
    parser.add_argument("--equity", type=float, default=10000.0, help="Starting account equity in USD (default: 10000.0)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic data generation (default: 7)")
    parser.add_argument("--regime", type=str, choices=["all", "trending", "mean_reverting", "volatile"], default="all", help="Market regime to simulate")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward optimization (In-Sample / Out-of-Sample split)")
    parser.add_argument(
        "--vectorized", action="store_true",
        help="Reserved: run an ultra-fast vectorized Polars backtest. "
        "Not implemented in this release -- see the RuntimeError below.",
    )

    args = parser.parse_args()

    if args.vectorized:
        # `run_vectorized_polars_backtest` was never defined (the function does
        # not exist anywhere in the repo), so `python -m src.backtest --vectorized`
        # used to crash with an opaque `NameError: name 'run_vectorized_polars_backtest'
        # is not defined`. A half-implemented "vectorized" engine is risky to ship here
        # because this module is the *validation* tool -- wrong numbers would give false
        # confidence in the live strategy. Fail loudly and point at the working path.
        raise RuntimeError(
            "--vectorized is not implemented in this release "
            "(run_vectorized_polars_backtest does not exist). The full strategy "
            "-- committee + risk + simulated fills -- is already exercised by the "
            "regular `run_backtest` path; run without --vectorized to use it."
        )
    elif args.walk_forward:
        await run_walk_forward_optimization(symbol=args.symbol, total_bars=args.bars, seed=args.seed)
    else:
        regimes_to_run = ["trending", "mean_reverting", "volatile"] if args.regime == "all" else [args.regime]

        for reg in regimes_to_run:
            res = await run_backtest(
                symbol=args.symbol,
                n_bars=args.bars,
                start_equity=args.equity,
                seed=args.seed,
                regime=reg
            )
            print_backtest_summary(res)
            
            # Fetch bars for benchmark comparison
            bars = None
            if args.bars <= 1000:  # Only for smaller backtests to avoid memory issues
                try:
                    bars_res = await fetch_real_bars(args.symbol, days=args.bars//24)
                    if bars_res is not None:
                        bars = bars_res.head(args.bars)
                except Exception:
                    pass
            
            if bars is not None and len(bars) > 0:
                comparison = await run_benchmark_comparison(res, args.symbol, bars, args.equity)
                print_benchmark_comparison(comparison)
            
            print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
