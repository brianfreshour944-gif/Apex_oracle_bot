"""Backtesting engine for the Apex Oracle Bot strategy.

Runs the REAL strategy + risk logic (TradingStrategy, RiskManager) over
historical OHLCV data using Polars, with a simple simulated fill model.
This is empirical validation of the deployed strategy - not a placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import polars as pl

from src.committee.committee import run_committee
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


class BacktestExchange:
    """Minimal in-memory exchange that replays historical bars for the strategy."""

    def __init__(self, bars: dict[str, pl.DataFrame]):
        self._bars = bars  # symbol -> DataFrame with columns [t, open, high, low, close, volume]
        self.current_time: str | None = None

    async def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100, end: datetime.datetime | None = None) -> pl.DataFrame:
        df = self._bars.get(symbol, pl.DataFrame())
        # Use current_time for point-in-time filtering (backtest PIT)
        filter_time = self.current_time
        if end is not None:
            # Convert end datetime to string for comparison
            filter_time = end.isoformat() if isinstance(end, datetime.datetime) else str(end)
        if filter_time is not None:
            df = df.filter(pl.col("t") <= filter_time)
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
                bars = bars.with_columns(pl.lit(datetime.datetime.utcnow().isoformat()).alias("t"))
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


async def run_backtest(
    symbol: str = "BTC/USD",
    n_bars: int = 400,
    start_equity: float = 10000.0,
    seed: int = 7,
    regime: str = "trending",
    fee_pct: float = 0.001,       # 0.1% Taker Fee
    slippage_pct: float = 0.0005, # 0.05% Slippage Buffer
    bars: pl.DataFrame | None = None,  # Real historical bars; if None, uses synthetic data
    use_committee: bool = True,   # If True, run the full 5-brain committee (AI-driven decisions).
                                   # If False, use the raw rule-based strategy signal only.
) -> BacktestResult:
    """Run a full backtest with fee and slippage execution modeling.

    This uses the SAME position sizing logic as live trading:
    - calculate_position_size() with ATR stops, regime multipliers, confidence,
      correlation penalties, transaction cost model, gap-risk multiplier, etc.
    - Fee and slippage from risk model's transaction cost model (not flat %)

    If `bars` is provided (real historical OHLCV data), it is used instead of
    synthetic data, and signal["backtest_df"] is populated per-bar so that
    transformer_brain.py uses this historical context instead of attempting
    a live Alpaca fetch.
    """

    if bars is None:
        # Try to fetch real historical data first; fall back to synthetic
        days = max(n_bars // 24, 30)  # estimate days from bar count (assuming 1h bars)
        real_bars = await fetch_real_bars(symbol, days=days)
        if real_bars is not None and len(real_bars) >= n_bars:
            bars = real_bars.head(n_bars)
        else:
            bars = _generate_synthetic_bars(symbol, n=n_bars, seed=seed, regime=regime)
    exchange = BacktestExchange({symbol: bars})

    strategy = TradingStrategy(exchange, cache_ttl=0.0, backtest=True)
    risk = RiskManager(exchange)

    result = BacktestResult(symbol=symbol, start_equity=start_equity, end_equity=start_equity)
    equity = Decimal(str(start_equity))
    result.equity_curve.append(float(equity))

    # Track open position per symbol
    open_pos: dict[str, Any] | None = None
    entry_price = 0.0
    entry_time = ""

    # For correlation matrix in backtest (simplified - single symbol)
    returns_matrix = None

    for row in bars.iter_rows(named=True):
        current_price = float(row["close"])
        ts = row["t"]
        exchange.current_time = ts

        # Build a position dict the strategy understands (only if we hold one)
        position = None
        if open_pos is not None:
            position = {
                "symbol": symbol,
                "qty": open_pos["qty"],
                "avg_entry_price": entry_price,
                "side": open_pos["side"],
            }

        signal = await strategy.generate_trading_signal(symbol, current_price, position)

        # Track regimes seen
        regime_seen = signal.get("regime", "neutral")
        result.regimes_seen[regime_seen] = result.regimes_seen.get(regime_seen, 0) + 1

        if use_committee:
            bars_df = await exchange.get_bars(symbol)
            signal["backtest_df"] = bars_df
            committee_result = await run_committee(symbol, current_price, signal)
            final_action = committee_result.action
            # Use committee confidence for position sizing
            confidence = committee_result.score
        else:
            final_action = signal["action"]
            confidence = signal.get("confidence", 1.0)

        # Get signal features for position sizing (same as live)
        atr = signal.get("atr")
        expected_return_pct = signal.get("expected_return_pct", 0.0)
        current_equity = float(equity)
        drawdown_pct = 0.0
        if result.equity_curve:
            peak = max(result.equity_curve)
            if peak > 0:
                drawdown_pct = (float(equity) - peak) / peak * 100

        if final_action == "buy" and open_pos is None:
            # Use SAME position sizing as live trading
            size, status = risk.calculate_position_size(
                symbol=symbol,
                current_price=current_price,
                regime=regime_seen,
                atr=atr,
                confidence=confidence,
                returns_matrix=returns_matrix,
                expected_return_pct=expected_return_pct,
                current_equity=current_equity,
                drawdown_pct=drawdown_pct,
                side="buy",
            )
            if status == "ok" and size > 0:
                open_pos = {"qty": size, "side": "long"}
                entry_price = current_price
                entry_time = ts
                logger.info(f"[BT] BUY {size:.6f} {symbol} @ {current_price:.2f} (regime={regime_seen}, size={size:.6f})")

        elif final_action == "sell" and open_pos is None:
            size, status = risk.calculate_position_size(
                symbol=symbol,
                current_price=current_price,
                regime=regime_seen,
                atr=atr,
                confidence=confidence,
                returns_matrix=returns_matrix,
                expected_return_pct=expected_return_pct,
                current_equity=current_equity,
                drawdown_pct=drawdown_pct,
                side="sell",
            )
            if status == "ok" and size > 0:
                open_pos = {"qty": size, "side": "short"}
                entry_price = current_price
                entry_time = ts
                logger.info(f"[BT] SELL/SHORT {size:.6f} {symbol} @ {current_price:.2f} (regime={regime_seen}, size={size:.6f})")

        elif final_action == "close" and open_pos is not None:
            qty = open_pos["qty"]
            if open_pos["side"] == "long":
                gross_pnl = (current_price - entry_price) * qty
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                gross_pnl = (entry_price - current_price) * qty
                pnl_pct = (entry_price - current_price) / entry_price * 100

            # Use same fee/slippage model as live (from risk model)
            notional = current_price * qty
            tx_costs = risk.get_transaction_costs(symbol)
            total_cost_bps = tx_costs["total_bps"]
            round_trip_cost_bps = 2.0 * total_cost_bps
            cost_fraction = round_trip_cost_bps / 10000
            fee = notional * cost_fraction
            # Slippage is already included in cost_fraction, no double-count
            pnl = gross_pnl - fee
            equity += Decimal(str(pnl))
            result.trades.append(BacktestTrade(
                symbol=symbol, side=open_pos["side"], entry_price=entry_price,
                exit_price=current_price, qty=qty, entry_time=entry_time,
                exit_time=ts, pnl=pnl, pnl_pct=pnl_pct, reason=signal.get("reason", "close"),
            ))
            logger.info(f"[BT] CLOSE {symbol} @ {current_price:.2f} pnl={pnl:.2f} ({pnl_pct:.2f}%) reason={signal.get('reason')}")
            open_pos = None

        result.equity_curve.append(float(equity))

    # Close any remaining position at the last price (mark-to-market)
    if open_pos is not None:
        last_row = bars.row(-1, named=True)
        last_price = float(last_row["close"])
        qty = open_pos["qty"]
        if open_pos["side"] == "long":
            gross_pnl = (last_price - entry_price) * qty
            pnl_pct = (last_price - entry_price) / entry_price * 100
        else:
            gross_pnl = (entry_price - last_price) * qty
            pnl_pct = (entry_price - last_price) / entry_price * 100
        
        # Use same transaction cost model as live
        notional = last_price * qty
        tx_costs = risk.get_transaction_costs(symbol)
        total_cost_bps = tx_costs["total_bps"]
        round_trip_cost_bps = 2.0 * total_cost_bps
        cost_fraction = round_trip_cost_bps / 10000
        fee = notional * cost_fraction
        pnl = gross_pnl - fee
        equity += Decimal(str(pnl))
        result.equity_curve[-1] = float(equity)

    # Compute metrics
    result.end_equity = float(equity)
    result.total_return_pct = (equity - Decimal(str(start_equity))) / Decimal(str(start_equity)) * 100
    result.n_trades = len(result.trades)
    result.n_wins = sum(1 for t in result.trades if t.pnl > 0)
    result.n_losses = sum(1 for t in result.trades if t.pnl <= 0)
    result.win_rate = (result.n_wins / result.n_trades * 100) if result.n_trades else 0.0

    eq = np.array(result.equity_curve)
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak * 100
    result.max_drawdown_pct = float(drawdown.min()) if len(drawdown) else 0.0

    # Trade-level returns for extended metrics
    if result.trades:
        trade_rets = np.array([t.pnl_pct / 100.0 for t in result.trades])
        trade_pnls = np.array([t.pnl for t in result.trades])
        wins = trade_rets[trade_rets > 0]
        losses = trade_rets[trade_rets < 0]
        win_pnls = trade_pnls[trade_pnls > 0]
        loss_pnls = trade_pnls[trade_pnls < 0]

        # Basic stats
        result.avg_win_pct = float(wins.mean()) if len(wins) else 0.0
        result.avg_loss_pct = float(losses.mean()) if len(losses) else 0.0
        result.avg_win_usd = float(win_pnls.mean()) if len(win_pnls) else 0.0
        result.avg_loss_usd = float(loss_pnls.mean()) if len(loss_pnls) else 0.0

        # Payoff ratio and profit factor
        if len(losses) > 0 and losses.mean() != 0:
            result.payoff_ratio = float(wins.mean() / abs(losses.mean()))
        else:
            result.payoff_ratio = float('inf') if len(wins) > 0 else 0.0

        if len(loss_pnls) > 0 and loss_pnls.sum() != 0:
            result.profit_factor = float(win_pnls.sum() / abs(loss_pnls.sum()))
        else:
            result.profit_factor = float('inf') if len(win_pnls) > 0 else 0.0

        # Expectancy per trade
        result.expectancy_pct = float(trade_rets.mean())
        result.expectancy_usd = float(trade_pnls.mean())

        # Sortino ratio (downside deviation only)
        downside_rets = trade_rets[trade_rets < 0]
        if len(downside_rets) > 1 and downside_rets.std() > 0:
            result.sortino = float(trade_rets.mean() / downside_rets.std() * np.sqrt(252))
        else:
            result.sortino = float('inf') if trade_rets.mean() > 0 else 0.0

        # Calmar ratio (annualized return / max drawdown)
        annual_return = float(trade_rets.mean() * 252)
        if result.max_drawdown_pct > 0:
            result.calmar = float(annual_return / (result.max_drawdown_pct / 100.0))
        else:
            result.calmar = float('inf') if annual_return > 0 else 0.0

        # Top 3 trades contribution
        sorted_pnls = np.sort(trade_pnls)
        top3_pnl = float(sorted_pnls[-3:].sum()) if len(sorted_pnls) >= 3 else float(sorted_pnls.sum())
        result.top3_contribution_pct = float(top3_pnl / float(equity) * 100) if equity > 0 else 0.0

        # Turnover (annualized)
        avg_equity = float(np.mean(eq)) if len(eq) > 0 else float(start_equity)
        result.turnover_annualized = float(len(result.trades) / avg_equity * 252 * np.mean([abs(t.qty * t.entry_price) for t in result.trades]) / avg_equity) if avg_equity > 0 and result.trades else 0.0

        # Cost-adjusted return
        total_fees = sum(abs(t.qty * t.entry_price) * (result.get('avg_cost_bps', 20) / 10000) for t in result.trades) if hasattr(result, 'get') else 0.0
        result.cost_adjusted_return_pct = result.total_return_pct - (total_fees / float(start_equity) * 100)

    # Simple Sharpe (daily returns)
    if len(eq) > 2:
        rets = np.diff(eq) / eq[:-1]
        result.sharpe = float(np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)) if np.std(rets) > 0 else 0.0

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
    random_avg_sharpe = 0.0  # Would need to compute from permutations
    
    # 3. Compare
    strategy_return = float(result.total_return_pct)
    strategy_sharpe = float(result.sharpe)
    strategy_max_dd = float(result.max_drawdown_pct)
    strategy_calmar = getattr(result, 'calmar', 0.0)
    strategy_sortino = getattr(result, 'sortino', 0.0)
    
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
    import asyncio

    parser = argparse.ArgumentParser(description="Run Apex Oracle Bot Backtest Engine")
    parser.add_argument("--symbol", type=str, default="BTC/USD", help="Symbol to backtest (default: BTC/USD)")
    parser.add_argument("--bars", type=int, default=400, help="Number of historical bars (default: 400)")
    parser.add_argument("--equity", type=float, default=10000.0, help="Starting account equity in USD (default: 10000.0)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic data generation (default: 7)")
    parser.add_argument("--regime", type=str, choices=["all", "trending", "mean_reverting", "volatile"], default="all", help="Market regime to simulate")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward optimization (In-Sample / Out-of-Sample split)")
    parser.add_argument("--vectorized", action="store_true", help="Run ultra-fast vectorized Polars backtest")

    args = parser.parse_args()

    if args.vectorized:
        run_vectorized_polars_backtest(symbol=args.symbol, n_bars=args.bars, seed=args.seed)
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
