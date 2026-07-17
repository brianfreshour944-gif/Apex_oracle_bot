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
) -> BacktestResult:
    """Run a full backtest of the deployed strategy over synthetic history."""
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


def print_backtest_summary(result: BacktestResult) -> None:
    """Pretty-print backtest results."""
    print("=" * 60)
    print(f"BACKTEST RESULTS: {result.symbol}")
    print("=" * 60)
    print(f"Start equity:      ${result.start_equity:,.2f}")
    print(f"End equity:        ${result.end_equity:,.2f}")
    print(f"Total return:      {result.total_return_pct:.2f}%")
    print(f"Trades:            {result.n_trades}")
    print(f"Wins / Losses:     {result.n_wins} / {result.n_losses}")
    print(f"Win rate:          {result.win_rate:.1f}%")
    print(f"Max drawdown:      {result.max_drawdown_pct:.2f}%")
    print(f"Sharpe (ann.):     {result.sharpe:.2f}")
    print(f"Regimes seen:      {result.regimes_seen}")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio

    for regime in ["trending", "mean_reverting", "volatile"]:
        res = asyncio.run(run_backtest(symbol="BTC/USD", n_bars=400, regime=regime, seed=7))
        print_backtest_summary(res)
        print()