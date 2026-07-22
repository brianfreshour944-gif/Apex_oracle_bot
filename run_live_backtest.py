import asyncio
import polars as pl
import yfinance as yf
from src.backtest import BacktestExchange, print_backtest_summary, BacktestResult, BacktestTrade
from src.strategies import TradingStrategy
from src.risk import RiskManager

class LiveBacktestExchange(BacktestExchange):
    def __init__(self, bars):
        super().__init__(bars)
        self.current_time = None

    async def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100) -> pl.DataFrame:
        df = self._bars.get(symbol, pl.DataFrame())
        if self.current_time is not None:
            df = df.filter(pl.col("t") <= self.current_time)
        return df.tail(limit) if len(df) else df


async def run_live_data_backtest(symbol: str = "BTC-USD", n_bars: int = 400):
    # Fetch live/historical data using yfinance
    print(f"Fetching historical 1-hour data for {symbol}...")
    try:
        # yfinance 1h data max is 730d, we'll fetch 365d
        ticker = yf.Ticker(symbol)
        df_pandas = ticker.history(period="365d", interval="1h")
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")
        return
    
    if df_pandas.empty:
        print(f"Failed to fetch data for {symbol}.")
        return

    # Reset index to get the Date column
    df_pandas = df_pandas.reset_index()
    
    # Rename columns to match expected schema: "t", "open", "high", "low", "close", "volume"
    df_pandas = df_pandas.rename(columns={
        "Datetime": "t",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    
    # Convert to Polars DataFrame
    bars = pl.from_pandas(df_pandas[["t", "open", "high", "low", "close", "volume"]])
    
    # Print a small preview of the data
    print("Data Preview: (disabled due to windows console unicode issue)")
    
    # Run the backtest logic using the fetched bars
    # Since we can't easily modify the `run_backtest` function to accept pre-fetched bars without modifying backtest.py,
    # we'll reimplement the loop here. It's essentially the exact same logic.
    
    exchange = LiveBacktestExchange({symbol: bars})
    strategy = TradingStrategy(exchange)
    risk = RiskManager(exchange)
    
    start_equity = 10000.0
    result = BacktestResult(symbol=symbol, start_equity=start_equity, end_equity=start_equity)
    equity = start_equity
    result.equity_curve.append(equity)
    
    open_pos = None
    entry_price = 0.0
    entry_time = ""
    
    print("\nStarting Backtest Simulation...")
    for row in bars.iter_rows(named=True):
        current_price = float(row["close"])
        ts = str(row["t"])
        exchange.current_time = row["t"]
        
        position = None
        if open_pos is not None:
            position = {
                "symbol": symbol,
                "qty": open_pos["qty"],
                "avg_entry_price": entry_price,
                "side": open_pos["side"],
            }
            
        signal = await strategy.generate_trading_signal(symbol, current_price, position)
        if signal["action"] not in ("hold", "stand_aside"):
            print(f"[{ts}] Signal: {signal['action']} (regime={signal.get('regime')}, rsi={signal.get('rsi'):.2f}, reason={signal.get('reason')})")
        
        regime_seen = signal.get("regime", "neutral")
        result.regimes_seen[regime_seen] = result.regimes_seen.get(regime_seen, 0) + 1
        
        if signal["action"] == "buy" and open_pos is None:
            size, status = risk.calculate_position_size(symbol, current_price, regime_seen)
            if status == "ok" and size > 0:
                open_pos = {"qty": size, "side": "long"}
                entry_price = current_price
                entry_time = ts
                
        elif signal["action"] == "sell" and open_pos is None:
            size, status = risk.calculate_position_size(symbol, current_price, regime_seen)
            if status == "ok" and size > 0:
                open_pos = {"qty": size, "side": "short"}
                entry_price = current_price
                entry_time = ts
                
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
            open_pos = None
            
        result.equity_curve.append(equity)
        
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
        
    import numpy as np
    
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
    
    print_backtest_summary(result)


if __name__ == "__main__":
    asyncio.run(run_live_data_backtest(symbol="BTC-USD", n_bars=365))
