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
import numpy as np
import hashlib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.strategies import TradingStrategy
from src.risk import RiskManager
from src.committee.committee import run_committee
from src.committee.models import BrainVote
from src.logging_config import get_logger

# When set, transformer_brain uses a fast synthetic model instead of the
# real PyTorch inference so the replay dataset can be bootstrapped in
# minutes rather than hours.  Set REPLAY_FAST_MODE=0 to use the real model.
FAST_MODE = os.environ.get("REPLAY_FAST_MODE", "1") == "1"

if FAST_MODE:
    # Disable all heavy ML during fast replay generation
    os.environ["ADAPTIVE_ML_ENABLED"] = "0"

if FAST_MODE:
    import src.committee.transformer_brain as _tb
    import src.committee.committee as _cm
    import src.committee.rl_meta as _rlm

    async def _fast_transformer_brain(symbol: str, price: float, signal: dict) -> BrainVote:
        """Synthetic transformer brain that generates deterministic tensor_state
        without loading the PyTorch model."""
        raw_action = signal.get("action", "hold")
        regime = signal.get("regime", "unknown")
        features = signal.get("features", {})
        
        seed_input = f"{symbol}:{price}:{regime}:{raw_action}"
        for k in sorted(features.keys()):
            seed_input += f":{k}={features[k]}"
        seed = int(hashlib.sha256(seed_input.encode()).hexdigest()[:8], 16) % 10000 / 10000.0
        
        if raw_action == "buy":
            prob = 0.65 + seed * 0.2
        elif raw_action == "sell":
            prob = 0.35 - seed * 0.2
        else:
            prob = 0.50 + (seed - 0.5) * 0.1
        
        prob = max(0.01, min(0.99, prob))
        rng = np.random.RandomState(seed=int(seed * 100000))
        tensor_state = [float(x) for x in rng.randn(128).astype(np.float32)]
        
        if prob > 0.58:
            action = "buy"
        elif prob < 0.42:
            action = "sell"
        else:
            action = "hold"
        
        return BrainVote(
            name="transformer",
            action=action,
            confidence=prob,
            weight=0.35,
            regime=regime,
            reason=f"Fast synthetic inference prob={prob:.3f}",
            causal_reasoning=None,
            tensor_state=tensor_state
        )

    # Patch in both places: the module where it's defined AND the module
    # where it's imported/used by run_committee().
    _tb.transformer_brain = _fast_transformer_brain
    _cm.transformer_brain = _fast_transformer_brain

    # Also prevent the RL meta-learner from loading
    _rlm.RLMetaLearner = type('RLMetaLearner', (), {
        '__init__': lambda self, *a, **kw: setattr(self, 'model', None),
        'combine': lambda *a, **kw: None,
    })

logger = get_logger("replay_generator")

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]
BUFFER_PATH = "data/historical_experiences.jsonl"
HISTORY_DAYS = 180  # ~6 months of data for replay bootstrap

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
    strategy = TradingStrategy(exchange, cache_ttl=300.0, backtest=True)
    risk = RiskManager(exchange)
    
    open_pos = None
    entry_price = 0.0
    entry_time = None
    entry_snapshot = None
    
    trades_logged = 0
    batch_records = []
    BATCH_SIZE = 500
    STEP_SIZE = 1
    
    for i in range(100, len(bars), STEP_SIZE):
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
                "avg_entry_price": entry_price,
                "current_price": current_price
            }
            
        signal = await strategy.generate_trading_signal(symbol, current_price, position)
        regime_seen = signal.get("regime", "neutral")
        
        # Only invoke the full committee (with expensive transformer inference)
        # when we have an active position to manage OR the strategy signals a
        # directional trade. This cuts transformer calls by >10x since the
        # strategy emits "hold" on the vast majority of bars.
        needs_committee = open_pos is not None or signal.get("action") in ("buy", "sell")
        
        if needs_committee:
            bars_df = await exchange.get_bars(symbol)
            signal["backtest_df"] = bars_df
            committee_result = await run_committee(symbol, current_price, signal)
            final_action = committee_result.action
        else:
            final_action = signal.get("action", "hold")
            committee_result = None
        
        if final_action == "buy" and open_pos is None:
            if committee_result is not None:
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
            if committee_result is not None:
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
                
                batch_records.append(record)
                trades_logged += 1
                
                if len(batch_records) >= BATCH_SIZE:
                    os.makedirs("data", exist_ok=True)
                    with open(BUFFER_PATH, "a") as f:
                        for rec in batch_records:
                            f.write(json.dumps(rec) + "\n")
                    batch_records.clear()
            
            open_pos = None
            entry_snapshot = None
            
    if batch_records:
        os.makedirs("data", exist_ok=True)
        with open(BUFFER_PATH, "a") as f:
            for rec in batch_records:
                f.write(json.dumps(rec) + "\n")
    
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
            logger.info(f"Generated {logged} historical replay records for {sym}")
            total_generated += logged
            
        except Exception as e:
            logger.error(f"Failed processing {sym}: {e}")
            
    logger.info(f"Bulk generation complete. Added {total_generated} records to {BUFFER_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
