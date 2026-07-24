#!/usr/bin/env python3
"""
Evolutionary PPO Trainer Pipeline.

Mutates strategy parameters through multiple generations. Only strategies that
pass rigid statistical hurdles (Sharpe, Max DD, Win Rate, Monte Carlo) are 
permitted to contribute their historical trades to the PPO Meta-Learner.
"""

import sys
import os
import copy
import random
import asyncio
import numpy as np
import polars as pl
from datetime import datetime
from typing import Dict, Any, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.backtest import BacktestResult, BacktestTrade, run_monte_carlo_analysis
from src.strategies import TradingStrategy
from src.risk import RiskManager
from src.committee.committee import run_committee
from src.committee.outcome_tracker import from_decision_snapshot
from src.logging_config import get_logger

logger = get_logger("evolutionary_ppo")

# GA Parameters
GENERATIONS = 500
MUTATION_RATE = 0.3
PARAM_BOUNDS = {
    "STOP_LOSS_PCT": (0.01, 0.10),
    "PROFIT_TARGET_PCT": (0.02, 0.15),
    "RSI_OVERSOLD": (20.0, 40.0),
    "RSI_OVERBOUGHT": (60.0, 80.0),
}

# Criteria
MIN_SHARPE = 1.2
MAX_DD_PCT = 15.0
MIN_WIN_RATE = 52.0
MAX_RISK_OF_RUIN = 5.0

class FastExchange:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.current_time = None
        
    async def get_bars(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pl.DataFrame:
        if self.current_time is not None:
            filtered = self.df.filter(pl.col("t") <= self.current_time)
            return filtered.tail(limit)
        return self.df.tail(limit)
        
    async def get_account(self) -> Dict[str, Any]:
        return {"equity": 0.0, "cash": 0.0, "portfolio_value": 0.0}
        
    async def get_positions(self) -> List[Dict[str, Any]]:
        return []

def mutate(genome: Dict[str, Any]) -> Dict[str, Any]:
    child = copy.deepcopy(genome)
    for key, bounds in PARAM_BOUNDS.items():
        if random.random() < MUTATION_RATE:
            # Nudge by +/- 20%
            nudge = child[key] * random.uniform(-0.2, 0.2)
            child[key] = round(max(bounds[0], min(bounds[1], child[key] + nudge)), 3)
    return child

def build_backtest_stats(result: BacktestResult):
    """Computes advanced stats for an in-memory result object."""
    if not result.trades:
        return
        
    result.n_trades = len(result.trades)
    result.n_wins = sum(1 for t in result.trades if t.pnl > 0)
    result.n_losses = sum(1 for t in result.trades if t.pnl <= 0)
    result.win_rate = (result.n_wins / result.n_trades * 100) if result.n_trades else 0.0
    
    eq = np.array(result.equity_curve)
    if len(eq) > 0:
        peak = np.maximum.accumulate(eq)
        drawdown_pct = (eq - peak) / peak * 100
        result.max_drawdown_pct = float(drawdown_pct.min())
    else:
        result.max_drawdown_pct = 0.0
        
    if len(eq) > 2:
        rets = np.diff(eq) / eq[:-1]
        std_rets = np.std(rets)
        result.sharpe = float(np.mean(rets) / (std_rets + 1e-9) * np.sqrt(252)) if std_rets > 0 else 0.0

async def simulate_candidate(bars: pl.DataFrame, symbol: str, config: Dict[str, Any]) -> tuple[BacktestResult, List[Dict]]:
    # Apply config
    originals = {k: getattr(settings, k) for k in config.keys()}
    for k, v in config.items():
        setattr(settings, k, v)
        
    exchange = FastExchange(bars)
    strategy = TradingStrategy(exchange)
    risk = RiskManager()
    
    start_equity = 10000.0
    equity = start_equity
    open_pos = None
    entry_price = 0.0
    entry_time = None
    entry_snapshot = None
    
    result = BacktestResult(symbol=symbol, start_equity=start_equity)
    snapshots = []
    
    try:
        # Step through data
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
            
            # NOTE: We skip Transformer here to keep GA loop fast, using only fast quant/momentum brains
            # For a true committee, we would use run_committee() but that calls PyTorch.
            # We mock the committee decision using the primary strategy signal to gather baseline trades.
            raw_action = signal.get("action", "hold")
            regime_seen = signal.get("regime", "neutral")
            
            if raw_action == "buy" and open_pos is None:
                size, status = risk.calculate_position_size(symbol, current_price, regime_seen)
                if status == "ok" and size > 0:
                    open_pos = {"qty": size, "side": "long"}
                    entry_price = current_price
                    entry_time = ts
                    entry_snapshot = {
                        "symbol": symbol,
                        "regime": regime_seen,
                        "final_action": "buy",
                        "confidence": 0.8,
                        "brain_votes": {"quant": "buy", "momentum": "buy"},
                        "features": {"rsi": signal.get("rsi", 50.0), "atr": signal.get("atr", 0.0)},
                        "entry_time": entry_time
                    }
                    
            elif raw_action == "sell" and open_pos is None:
                size, status = risk.calculate_position_size(symbol, current_price, regime_seen)
                if status == "ok" and size > 0:
                    open_pos = {"qty": size, "side": "short"}
                    entry_price = current_price
                    entry_time = ts
                    entry_snapshot = {
                        "symbol": symbol,
                        "regime": regime_seen,
                        "final_action": "sell",
                        "confidence": 0.8,
                        "brain_votes": {"quant": "sell", "momentum": "sell"},
                        "features": {"rsi": signal.get("rsi", 50.0), "atr": signal.get("atr", 0.0)},
                        "entry_time": entry_time
                    }
                    
            elif raw_action == "close" and open_pos is not None:
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
                    exit_time=ts, pnl=pnl, pnl_pct=pnl_pct, reason="signal"
                ))
                
                if entry_snapshot:
                    entry_snapshot["exit_time"] = ts
                    entry_snapshot["realized_pnl"] = pnl
                    snapshots.append(entry_snapshot)
                
                open_pos = None
                entry_snapshot = None
                
            result.equity_curve.append(equity)
            
        result.end_equity = equity
        result.total_return_pct = (equity - start_equity) / start_equity * 100
        build_backtest_stats(result)
        
        return result, snapshots
    finally:
        for k, v in originals.items():
            setattr(settings, k, v)


async def main():
    logger.info("Initializing Multi-Asset Evolutionary PPO Pipeline...")
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD", "ADA-USD", "LINK-USD", "LTC-USD", "AVAX-USD", "BCH-USD"]
    bars_dict = {}
    
    import yfinance as yf
    for sym in symbols:
        try:
            logger.info(f"Fetching {sym} data...")
            ticker = yf.Ticker(sym)
            df = ticker.history(period="180d", interval="1h")
            if not df.empty:
                df = df.reset_index()
                df = df.rename(columns={"Datetime": "t", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                df["t"] = df["t"].astype(str)
                bars_dict[sym] = pl.from_pandas(df)
        except Exception as e:
            logger.error(f"Data fetch failed for {sym}: {e}")
            
    if not bars_dict:
        logger.error("No data fetched for any symbols.")
        return
        
    current_model = {k: getattr(settings, k) for k in PARAM_BOUNDS.keys()}
    champion_score = -float('inf')
    
    surviving_snapshots = []
    
    for generation in range(GENERATIONS):
        logger.info(f"--- Generation {generation+1}/{GENERATIONS} ---")
        candidate = mutate(current_model)
        
        # 1. Backtest across all symbols
        total_trades = 0
        total_wins = 0
        all_snapshots = []
        composite_returns = []
        worst_dd = 0.0
        worst_res = None
        
        for sym, bars in bars_dict.items():
            res, snapshots = await simulate_candidate(bars, sym, candidate)
            total_trades += res.n_trades
            total_wins += res.n_wins
            all_snapshots.extend(snapshots)
            composite_returns.append(res.total_return_pct)
            
            if res.max_drawdown_pct < worst_dd:
                worst_dd = res.max_drawdown_pct
                worst_res = res
                
        if total_trades < 50:
            logger.info(f"Gen {generation+1}: Rejected (Too few aggregate trades: {total_trades})")
            continue
            
        composite_wr = (total_wins / total_trades) * 100
        avg_ret = np.mean(composite_returns)
        composite_sharpe = avg_ret / (np.std(composite_returns) + 1e-9) if len(composite_returns) > 1 else 0.0
        
        logger.info(f"Gen {generation+1} Candidate: Composite Sharpe ~{composite_sharpe:.2f} | Worst DD {worst_dd:.2f}% | WR {composite_wr:.1f}%")
        
        # 2. Reject bad models
        if composite_sharpe < MIN_SHARPE:
            logger.info("Rejected (Low Composite Sharpe)")
            continue
        if abs(worst_dd) > MAX_DD_PCT:
            logger.info("Rejected (High Worst-Case Drawdown)")
            continue
        if composite_wr < MIN_WIN_RATE:
            logger.info("Rejected (Low Composite Win Rate)")
            continue
            
        # 3. Monte Carlo validation (on the worst performing symbol)
        logger.info(f"Passed filters. Running Monte Carlo on worst performer ({worst_res.symbol})...")
        mc = run_monte_carlo_analysis(worst_res, n_simulations=200)
        
        if mc["risk_of_ruin_pct"] > MAX_RISK_OF_RUIN:
            logger.info(f"Rejected (Risk of Ruin too high: {mc['risk_of_ruin_pct']:.1f}%)")
            continue
            
        logger.info(f"✅ Candidate SURVIVED! Added {len(all_snapshots)} trades to PPO training buffer.")
        surviving_snapshots.extend(all_snapshots)
        
        score = composite_sharpe * 100 + avg_ret
        if score > champion_score:
            champion_score = score
            current_model = candidate
            logger.info(f"🏆 NEW CHAMPION: {current_model}")
            
    if surviving_snapshots:
        logger.info(f"Training PPO Meta-Learner on {len(surviving_snapshots)} surviving trades...")
        from src.committee.rl_env import MetaDecisionEnv
        from stable_baselines3 import PPO
        import os
        
        env = MetaDecisionEnv(surviving_snapshots)
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        save_path = os.path.join(models_dir, 'ppo_meta_weights.zip')
        
        def evaluate_ppo(test_model, test_env):
            obs, _ = test_env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action, _ = test_model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = test_env.step(action)
                total_reward += reward
                if truncated: done = True
            return total_reward

        # 1. Evaluate PPO Champion
        champion_reward = float('-inf')
        if os.path.exists(save_path):
            try:
                champion_model = PPO.load(save_path, env=env)
                champion_reward = evaluate_ppo(champion_model, env)
                logger.info(f"🏆 PPO Champion Baseline Reward: {champion_reward:.2f}")
            except Exception as e:
                logger.warning(f"Could not load existing PPO Champion: {e}")
                
        # 2. Train PPO Challenger
        logger.info("Training PPO Challenger...")
        challenger = PPO("MlpPolicy", env, verbose=0, learning_rate=0.001)
        challenger.learn(total_timesteps=len(surviving_snapshots) * 50)
        
        # 3. Evaluate PPO Challenger
        challenger_reward = evaluate_ppo(challenger, env)
        logger.info(f"⚔️ PPO Challenger Reward: {challenger_reward:.2f}")
        
        # 4. Champion vs Challenger
        if challenger_reward > champion_reward:
            challenger.save(save_path)
            logger.info(f"✅ PPO Challenger WINS! Saved updated meta-learner to {save_path}")
        else:
            logger.info("❌ PPO Challenger FAILED to beat Champion. Discarding new weights.")
    else:
        logger.warning("No candidates survived the GA filter. PPO was not trained.")

if __name__ == "__main__":
    asyncio.run(main())
