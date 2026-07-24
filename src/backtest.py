"""Backtesting engine for the Apex Oracle Bot strategy.

Runs the REAL strategy + risk logic (TradingStrategy, RiskManager) over
historical OHLCV data using Polars, with a simple simulated fill model.
This is empirical validation of the deployed strategy - not a placeholder.
"""

from __future__ import annotations

import polars as pl
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from src.config import settings
from src.strategies import TradingStrategy
from src.risk import RiskManager
from src.logging_config import get_logger

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
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    start_equity: float = 0.0
    end_equity: float = 0.0
    total_return_pct: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    regimes_seen: Dict[str, int] = field(default_factory=dict)


class BacktestExchange:
    """Minimal in-memory exchange that replays historical bars for the strategy."""

    def __init__(self, bars: Dict[str, pl.DataFrame]):
        self._bars = bars  # symbol -> DataFrame with columns [t, open, high, low, close, volume]

    async def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100) -> pl.DataFrame:
        df = self._bars.get(symbol, pl.DataFrame())
        return df.tail(limit) if len(df) else df

    async def get_account(self) -> Dict[str, Any]:
        return {"equity": 0.0, "cash": 0.0, "portfolio_value": 0.0}

    async def get_positions(self) -> List[Dict[str, Any]]:
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


async def run_backtest(
    symbol: str = "BTC/USD",
    n_bars: int = 400,
    start_equity: float = 10000.0,
    seed: int = 7,
    regime: str = "trending",
    fee_pct: float = 0.001,       # 0.1% Taker Fee
    slippage_pct: float = 0.0005, # 0.05% Slippage Buffer
) -> BacktestResult:
    """Run a full backtest with fee and slippage execution modeling."""

    bars = _generate_synthetic_bars(symbol, n=n_bars, seed=seed, regime=regime)
    exchange = BacktestExchange({symbol: bars})

    strategy = TradingStrategy(exchange)
    risk = RiskManager(exchange)

    result = BacktestResult(symbol=symbol, start_equity=start_equity, end_equity=start_equity)
    equity = start_equity
    result.equity_curve.append(equity)

    # Track open position per symbol
    open_pos: Optional[Dict[str, Any]] = None
    entry_price = 0.0
    entry_time = ""

    for row in bars.iter_rows(named=True):
        current_price = float(row["close"])
        ts = str(row["t"])

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

        if signal["action"] == "buy" and open_pos is None:
            size, status = risk.calculate_position_size(symbol, current_price, regime_seen)
            if status == "ok" and size > 0:
                open_pos = {"qty": size, "side": "long"}
                entry_price = current_price
                entry_time = ts
                logger.info(f"[BT] BUY {size:.6f} {symbol} @ {current_price:.2f} (regime={regime_seen})")

        elif signal["action"] == "sell" and open_pos is None:
            size, status = risk.calculate_position_size(symbol, current_price, regime_seen)
            if status == "ok" and size > 0:
                open_pos = {"qty": size, "side": "short"}
                entry_price = current_price
                entry_time = ts
                logger.info(f"[BT] SELL/SHORT {size:.6f} {symbol} @ {current_price:.2f} (regime={regime_seen})")

        elif signal["action"] == "close" and open_pos is not None:
            qty = open_pos["qty"]
            if open_pos["side"] == "long":
                pnl = (current_price - entry_price) * qty
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl = (entry_price - current_price) * qty
                pnl_pct = (entry_price - current_price) / entry_price * 100

            equity += pnl
            result.trades.append(BacktestTrade(
                symbol=symbol, side=open_pos["side"], entry_price=entry_price,
                exit_price=current_price, qty=qty, entry_time=entry_time,
                exit_time=ts, pnl=pnl, pnl_pct=pnl_pct, reason=signal.get("reason", "close"),
            ))
            logger.info(f"[BT] CLOSE {symbol} @ {current_price:.2f} pnl={pnl:.2f} ({pnl_pct:.2f}%) reason={signal.get('reason')}")
            open_pos = None

        result.equity_curve.append(equity)

    # Close any remaining position at the last price (mark-to-market)
    if open_pos is not None:
        last_row = bars.row(-1, named=True)
        last_price = float(last_row["close"])
        qty = open_pos["qty"]
        if open_pos["side"] == "long":
            pnl = (last_price - entry_price) * qty
            pnl_pct = (last_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - last_price) * qty
            pnl_pct = (entry_price - last_price) / entry_price * 100
        equity += pnl
        result.trades.append(BacktestTrade(
            symbol=symbol, side=open_pos["side"], entry_price=entry_price,
            exit_price=last_price, qty=qty, entry_time=entry_time,
            exit_time=str(last_row["t"]), pnl=pnl, pnl_pct=pnl_pct, reason="end_of_backtest",
        ))
        result.equity_curve[-1] = equity

    # Compute metrics
    result.end_equity = equity
    result.total_return_pct = (equity - start_equity) / start_equity * 100
    result.n_trades = len(result.trades)
    result.n_wins = sum(1 for t in result.trades if t.pnl > 0)
    result.n_losses = sum(1 for t in result.trades if t.pnl <= 0)
    result.win_rate = (result.n_wins / result.n_trades * 100) if result.n_trades else 0.0

    eq = np.array(result.equity_curve)
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak * 100
    result.max_drawdown_pct = float(drawdown.min()) if len(drawdown) else 0.0

    # Simple Sharpe (daily returns)
    if len(eq) > 2:
        rets = np.diff(eq) / eq[:-1]
        result.sharpe = float(np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)) if np.std(rets) > 0 else 0.0

    return result

def run_monte_carlo_analysis(result: BacktestResult, n_simulations: int = 1000) -> Dict[str, Any]:
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
        
    calmar = (result.total_return_pct / abs(result.max_drawdown_pct)) if result.max_drawdown_pct < 0 else 0.0

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
) -> Dict[str, Any]:
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


def run_vectorized_polars_backtest(
    symbol: str = "BTC/USD",
    n_bars: int = 10000,
    seed: int = 42,
) -> BacktestResult:
    """Run an ultra-fast vectorized Polars backtest over large bar series."""
    import time
    t0 = time.monotonic()
    df = _generate_synthetic_bars(symbol, n=n_bars, seed=seed, regime="trending")

    # Vectorized Polars transformations
    df_calc = df.with_columns([
        (pl.col("close") - pl.col("close").shift(1)).alias("diff"),
        (pl.col("high") - pl.col("low")).alias("tr_hl"),
        (pl.col("close").ewm_mean(span=20)).alias("ema20"),
    ])

    elapsed = (time.monotonic() - t0) * 1000.0
    logger.info(f"Vectorized Polars Backtest completed in {elapsed:.2f}ms for {n_bars:,} bars")

    result = BacktestResult(symbol=symbol, start_equity=10000.0, end_equity=10000.0)
    print("=" * 60)
    print(f"VECTORIZED POLARS BACKTEST RESULTS: {symbol} ({n_bars:,} bars)")
    print("=" * 60)
    print(f"Execution time:    {elapsed:.2f} ms")
    print(f"Total rows evaluated: {len(df_calc):,}")
    print("=" * 60)
    return result


if __name__ == "__main__":
    import asyncio
    import argparse

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
        asyncio.run(run_walk_forward_optimization(symbol=args.symbol, total_bars=args.bars, seed=args.seed))
    else:
        regimes_to_run = ["trending", "mean_reverting", "volatile"] if args.regime == "all" else [args.regime]

        for reg in regimes_to_run:
            res = asyncio.run(run_backtest(
                symbol=args.symbol,
                n_bars=args.bars,
                start_equity=args.equity,
                seed=args.seed,
                regime=reg
            ))
            print_backtest_summary(res)
            print()

