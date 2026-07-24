#!/usr/bin/env python3
"""
Historical Replay Augmentation Script

Bootstraps the Transformer replay buffer by simulating the current champion
strategy across deep historical data for multiple crypto assets. This prevents
the cold-start data starvation problem.
"""

import os
import sys
import json
import asyncio
import polars as pl
import yfinance as yf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.strategies import TradingStrategy
from src.risk import RiskManager
from src.committee.committee import run_committee
from src.logging_config import get_logger

logger = get_logger("replay_generator")

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD", "ADA-USD", "LINK-USD", "LTC-USD", "AVAX-USD", "BCH-USD"]
BUFFER_PATH = "data/transformer_replay_buffer.jsonl"
HISTORY_DAYS = 365 * 3  # 3 years

class FastExchange:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.current_time = None
        
    async def get_bars(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pl.DataFrame:
        if self.current_time is not None:
            filtered = self.df.filter(pl.col("t") <= self.current_time)
            return filtered.tail(limit)
        return self.df.tail(limit)
        
    async def get_account(self) -> dict:
        return {"equity": 10000.0, "cash": 10000.0, "portfolio_value": 10000.0}
        
    async def get_positions(self) -> list:
        return []

async def generate_history_for_symbol(symbol: str, bars: pl.DataFrame):
    logger.info(f"Generating trades for {symbol}...")
    exchange = FastExchange(bars)
    strategy = TradingStrategy(exchange)
    risk = RiskManager()
    
    open_pos = None
    entry_price = 0.0
    entry_time = None
    entry_snapshot = None
    
    trades_logged = 0
    
    for i in range(100, len(bars)):
        row = bars.row(i, named=True)
        ts = str(row["t"])
        current_price = float(row["close"])
        exchange.current_time = ts
        
        position = None
        if open_pos is not None:
            position = {
                "symbol": symbol,
                "qty": open_pos["qty"],
                "side": open_pos["side"],
                "entry_price": entry_price,
                "current_price": current_price
            }
            
        signal = await strategy.generate_trading_signal(symbol, current_price, position)
        
        # Here we ACTUALLY call the committee so the Transformer runs
        committee_result = await run_committee(symbol, current_price, signal)
        final_action = committee_result.action
        regime_seen = signal.get("regime", "neutral")
        
        if final_action == "buy" and open_pos is None:
            size, status = risk.calculate_position_size(symbol, current_price, regime_seen)
            if status == "ok" and size > 0:
                open_pos = {"qty": size, "side": "long"}
                entry_price = current_price
                entry_time = ts
                
                t_votes = [v for v in committee_result.votes if v.name == "transformer"]
                tensor_state = t_votes[0].tensor_state if t_votes else None
                t_prob = t_votes[0].confidence if t_votes else 0.5
                
                entry_snapshot = {
                    "symbol": symbol,
                    "regime": regime_seen,
                    "final_action": "buy",
                    "confidence": committee_result.score,
                    "weights": committee_result.active_weights,
                    "entry_time": entry_time,
                    "tensor_state": tensor_state,
                    "t_prob": t_prob,
                    "atr": signal.get("atr", 0.0),
                    "volatility": signal.get("volatility", 0.0)
                }
                
        elif final_action == "sell" and open_pos is None:
            size, status = risk.calculate_position_size(symbol, current_price, regime_seen)
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
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100
                
            if entry_snapshot and entry_snapshot.get("tensor_state") is not None:
                t_label = 1.0 if pnl_pct > 0 else 0.0
                
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
                
                os.makedirs("data", exist_ok=True)
                with open(BUFFER_PATH, "a") as f:
                    f.write(json.dumps(record) + "\n")
                
                trades_logged += 1
            
            open_pos = None
            entry_snapshot = None
            
    return trades_logged

async def main():
    logger.info("Initializing Bulk Replay Dataset Generator...")
    total_generated = 0
    
    for sym in SYMBOLS:
        try:
            logger.info(f"Fetching {HISTORY_DAYS} days of data for {sym}...")
            ticker = yf.Ticker(sym)
            df = ticker.history(period=f"{HISTORY_DAYS}d", interval="1h")
            if df.empty:
                logger.warning(f"No data for {sym}")
                continue
                
            df = df.reset_index()
            df = df.rename(columns={"Datetime": "t", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            df["t"] = df["t"].astype(str)
            bars = pl.from_pandas(df)
            
            logged = await generate_history_for_symbol(sym, bars)
            logger.info(f"✅ Generated {logged} historical replay records for {sym}")
            total_generated += logged
            
        except Exception as e:
            logger.error(f"Failed processing {sym}: {e}")
            
    logger.info(f"🎉 Bulk generation complete. Added {total_generated} records to {BUFFER_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
