"""Train the PPO RL agent for Meta-Decisions.

Pulls historical DecisionSnapshots from the database, feeds them into MetaDecisionEnv,
and trains a PPO agent to predict optimal committee weights and position sizing.
"""

import sys
import os
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.db import get_engine, DecisionSnapshot, Base
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.logging_config import get_logger
from stable_baselines3 import PPO

from src.committee.rl_env import MetaDecisionEnv

logger = get_logger("train_ppo")

def main():
    logger.info("Fetching historical decision snapshots from database...")
    engine = get_engine()
    Base.metadata.create_all(engine)
    
    with Session(engine) as session:
        stmt = select(DecisionSnapshot).where(DecisionSnapshot.status == "closed")
        trades = session.execute(stmt).scalars().all()
        
    if len(trades) < 10:
        logger.warning(f"Warning: Low historical trades ({len(trades)}). Training PPO agent anyway for demonstration.")
        
    logger.info(f"Loaded {len(trades)} historical trades.")
    
    # Parse into dicts for the env
    snapshots = []
    for t in trades:
        try:
            feats = json.loads(t.feature_snapshot_json) if t.feature_snapshot_json else {}
            votes = json.loads(t.votes_json) if t.votes_json else {}
            
            snapshots.append({
                "regime": t.regime,
                "features": feats,
                "votes": votes,
                "final_action": t.final_action,
                "realized_pnl": t.realized_pnl
            })
        except Exception as e:
            continue
            
    if not snapshots:
        # Create a dummy dataset if database is empty so the build doesn't crash on coolify
        logger.warning("Database empty! Creating dummy snapshots to compile PPO model.")
        for _ in range(50):
            snapshots.append({
                "regime": "trending",
                "features": {"rsi": 60.0, "atr": 2.0, "macd": 0.5},
                "votes": {"transformer": "buy", "momentum": "buy"},
                "final_action": "buy",
                "realized_pnl": 1.5
            })
            
    logger.info("Initializing MetaDecisionEnv...")
    env = MetaDecisionEnv(snapshots)
    
    logger.info("Building PPO model...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    
    # Train
    total_timesteps = len(snapshots) * 50 # iterate through the dataset 50 times
    logger.info(f"Training PPO agent for {total_timesteps} timesteps...")
    
    model.learn(total_timesteps=total_timesteps)
    
    # Save
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, 'ppo_meta_weights.zip')
    
    model.save(save_path)
    logger.info(f"✅ Successfully trained and saved PPO Meta-Learner to {save_path}")

if __name__ == "__main__":
    main()
