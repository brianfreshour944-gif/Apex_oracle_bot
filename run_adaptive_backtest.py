import asyncio
import os
import json
import numpy as np
import polars as pl
import yfinance as yf

# Override settings BEFORE importing components to ensure sandbox isolation
from src.config import settings
settings.ADAPTIVE_ML_ENABLED = True
settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE = 0
settings.ADAPTIVE_STATE_PATH = "data/sandbox_meta_state.json"

from src.backtest import BacktestResult, BacktestTrade, print_backtest_summary
from src.strategies import TradingStrategy
from src.risk import RiskManager
from src.committee.committee import run_committee, get_meta_learner
from src.committee.outcome_tracker import from_decision_snapshot

# Clean slate sandbox for the meta-learner
if os.path.exists(settings.ADAPTIVE_STATE_PATH):
    os.remove(settings.ADAPTIVE_STATE_PATH)

class LiveBacktestExchange:
    def __init__(self, bars):
        self._bars = bars
        self.current_time = None

    async def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 100) -> pl.DataFrame:
        df = self._bars.get(symbol, pl.DataFrame())
        if self.current_time is not None:
            df = df.filter(pl.col("t") <= self.current_time)
        return df.tail(limit) if len(df) else df

async def run_adaptive_simulation(symbol: str = "BTC-USD", n_days: int = 180):
    print(f"===========================================================")
    print(f"INITIALIZING ADAPTIVE META-LEARNER SIMULATION")
    print(f"Symbol: {symbol} | Period: {n_days} days | Sandbox: ON")
    print(f"===========================================================\n")

# ... (I'll just replace the specific lines instead of the whole block, let me use the tool correctly)


    # Fetch live/historical data using yfinance
    print(f"Fetching historical 1-hour data for {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        df_pandas = ticker.history(period=f"{n_days}d", interval="1h")
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")
        return
    
    if df_pandas.empty:
        print(f"Failed to fetch data for {symbol}.")
        return

    df_pandas = df_pandas.reset_index()
    df_pandas = df_pandas.rename(columns={
        "Datetime": "t", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    })
    
    bars = pl.from_pandas(df_pandas[["t", "open", "high", "low", "close", "volume"]])
    
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
    entry_snapshot = None  # To store the decision context for learning
    
    # Initialize the learner globally
    learner = get_meta_learner()
    
    print("\n[SIMULATION STARTED]")
    for row in bars.iter_rows(named=True):
        current_price = float(row["close"])
        ts = str(row["t"])
        exchange.current_time = row["t"]
        
        # Clear caches because real-world time isn't advancing
        strategy._regime_cache.clear()

        
        position = None
        if open_pos is not None:
            position = {
                "symbol": symbol,
                "qty": open_pos["qty"],
                "avg_entry_price": entry_price,
                "side": open_pos["side"],
            }
            
        signal = await strategy.generate_trading_signal(symbol, current_price, position)
        
        # Pass through the Adaptive Committee
        committee_result = await run_committee(symbol, current_price, signal)
        final_action = committee_result.action
        regime_seen = signal.get("regime", "neutral")
        
        result.regimes_seen[regime_seen] = result.regimes_seen.get(regime_seen, 0) + 1
        
        if final_action == "buy" and open_pos is None:
            # Use the multiplier derived from the committee's confidence
            size, status = risk.calculate_position_size(symbol, current_price, regime_seen)
            size = size * committee_result.size_multiplier

            if status == "ok" and size > 0:
                open_pos = {"qty": size, "side": "long"}
                entry_price = current_price
                entry_time = ts
                
                t_votes = [v for v in committee_result.votes if v.name == "transformer"]
                tensor_state = t_votes[0].tensor_state if t_votes else None
                t_prob = t_votes[0].confidence if t_votes else 0.5
                
                # Snapshot the committee's state for training
                entry_snapshot = {
                    "symbol": symbol,
                    "regime": regime_seen,
                    "final_action": "buy",
                    "confidence": committee_result.score,
                    "brain_votes": {v.name: v.action for v in committee_result.votes},
                    "weights": committee_result.active_weights,
                    "entry_time": entry_time,
                    "tensor_state": tensor_state,
                    "t_prob": t_prob,
                    "atr": signal.get("atr", 0.0),
                    "volatility": signal.get("volatility", 0.0)
                }
                
        elif final_action == "sell" and open_pos is None:
            size, status = risk.calculate_position_size(symbol, current_price, regime_seen)
            size = size * committee_result.size_multiplier

            if status == "ok" and size > 0:
                open_pos = {"qty": size, "side": "short"}
                entry_price = current_price
                entry_time = ts
                
                t_votes = [v for v in committee_result.votes if v.name == "transformer"]
                tensor_state = t_votes[0].tensor_state if t_votes else None
                t_prob = t_votes[0].confidence if t_votes else 0.5
                
                entry_snapshot = {
                    "symbol": symbol,
                    "regime": regime_seen,
                    "final_action": "sell",
                    "confidence": committee_result.score,
                    "brain_votes": {v.name: v.action for v in committee_result.votes},
                    "weights": committee_result.active_weights,
                    "entry_time": entry_time,
                    "tensor_state": tensor_state,
                    "t_prob": t_prob,
                    "atr": signal.get("atr", 0.0),
                    "volatility": signal.get("volatility", 0.0)
                }
                
        elif final_action == "close" and open_pos is not None:
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
            
            # --- META-LEARNER TRAINING STEP ---
            if entry_snapshot:
                entry_snapshot["exit_time"] = ts
                example = from_decision_snapshot(entry_snapshot, realized_pnl=pnl, return_pct=pnl_pct)
                learner.update(example.to_decision_snapshot(), example.to_realized_outcome())
                
                # Log to Transformer Replay Buffer
                if entry_snapshot.get("tensor_state") is not None:
                    t_label = 1.0 if pnl > 0 else 0.0
                    buffer_path = "data/live_experiences.jsonl"
                    os.makedirs("data", exist_ok=True)
                    
                    record = {
                        "tensor": entry_snapshot["tensor_state"],
                        "prediction": "buy" if entry_snapshot["t_prob"] > 0.5 else "sell",
                        "actual_outcome": t_label,
                        "confidence": entry_snapshot["t_prob"],
                        "committee_weights": entry_snapshot["weights"],
                        "market_regime": entry_snapshot["regime"],
                        "symbol": entry_snapshot["symbol"],
                        "atr": entry_snapshot["atr"],
                        "volatility": entry_snapshot["volatility"],
                        "entry_time": entry_snapshot["entry_time"],
                        "exit_time": ts,
                        "reward": pnl_pct,
                        "label": t_label
                    }
                    
                    with open(buffer_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                
                label = "WIN" if pnl > 0 else "LOSS"
                print(f"[{ts}] Trade Closed: {label} ({pnl_pct:+.2f}%). Sent to Meta-Learner & Replay Buffer.")
            
            open_pos = None
            entry_snapshot = None
            
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
        
        if entry_snapshot:
            entry_snapshot["exit_time"] = str(last_row["t"])
            example = from_decision_snapshot(entry_snapshot, realized_pnl=pnl, return_pct=pnl_pct)
            learner.update(example.to_decision_snapshot(), example.to_realized_outcome())
            
        result.equity_curve[-1] = equity
        
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
    
    print("\n===========================================================")
    print("FINAL ADAPTIVE META-LEARNER WEIGHTS AFTER SIMULATION")
    print("===========================================================")
    weights = learner.weights
    for regime, b_weights in weights.items():
        print(f"Regime [{regime.upper()}]:")
        for brain, w in sorted(b_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {brain.ljust(12)} : {w:.3f}")
    print("===========================================================\n")

if __name__ == "__main__":
    try:
        asyncio.run(run_adaptive_simulation(symbol="BTC-USD", n_days=365))
    except Exception as e:
        import traceback
        traceback.print_exc()
