"""Reinforcement Learning Environment for Meta-Decisions.

A standard Gymnasium environment where the agent learns to optimally
weight the committee brains and size positions based on historical market snapshots.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

BRAINS = ["transformer", "quant", "momentum", "sentinel", "llm"]
REGIMES = ["trending", "mean_reverting", "volatile", "choppy", "breakout", "default"]

class MetaDecisionEnv(gym.Env):
    """
    State:
      - Regime (One-hot encoded, 6 dims)
      - Technical features (RSI, ATR, MACD) (3 dims)
      - Brain votes (Buy=1, Sell=-1, Hold=0) (5 dims)
      Total Observation Space = 14 dimensions
      
    Action:
      - 5 Brain Weights (softmaxed internally to sum to 1)
      - Position Size Multiplier (scaled to 0.5 - 1.5)
      - Confidence Threshold (scaled to 0.5 - 0.8)
      Total Action Space = 7 continuous dimensions
      
    Reward:
      - Realized PnL of the resulting theoretical trade
    """
    
    def __init__(self, historical_snapshots: list[dict[str, Any]]):
        super().__init__()
        
        self.snapshots = historical_snapshots
        self.current_step = 0
        
        # Observation Space: 1(RSI) + 1(ATR) + 1(Transformer Conf) + 1(Sentiment) + 6(Regimes) + 1(ADX) + 1(Volatility) + 5(Votes) = 17
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(17,), dtype=np.float32)
        
        # Action Space: 7 continuous variables between -1 and 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
    def _get_obs(self):
        if self.current_step >= len(self.snapshots):
            return np.zeros(17, dtype=np.float32)
            
        snap = self.snapshots[self.current_step]
        
        # 1. Features
        feats = snap.get("features", {})
        rsi = (feats.get("rsi", 50.0) - 50) / 50.0  # scale -1 to 1
        atr = feats.get("atr", 0.0) / 100.0
        transformer_conf = snap.get("confidence", 0.5)
        sentiment = np.clip(feats.get("sentiment_score", 0.0), -1.0, 1.0)
        adx = feats.get("adx", 20.0) / 100.0
        volatility = feats.get("volatility", 0.0)
        
        # 2. Regime One-Hot
        regime = snap.get("regime", "default")
        regime_vec = np.zeros(len(REGIMES), dtype=np.float32)
        if regime in REGIMES:
            regime_vec[REGIMES.index(regime)] = 1.0
            
        feature_vec = np.array([rsi, atr, transformer_conf, sentiment, adx, volatility], dtype=np.float32)
        
        # 3. Brain Votes
        votes = snap.get("votes", snap.get("brain_votes", {}))
        vote_vec = np.zeros(len(BRAINS), dtype=np.float32)
        for i, b in enumerate(BRAINS):
            v = votes.get(b, "hold")
            if v == "buy":
                vote_vec[i] = 1.0
            elif v == "sell":
                vote_vec[i] = -1.0
                
        obs = np.concatenate([feature_vec, regime_vec, vote_vec])
        return np.nan_to_num(obs, 0.0).astype(np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}
        
    def step(self, action):
        if self.current_step >= len(self.snapshots):
            return np.zeros(14, dtype=np.float32), 0.0, True, False, {}
            
        snap = self.snapshots[self.current_step]
        
        # Extract actions
        raw_weights = action[0:5]
        # softmax the weights
        exp_w = np.exp(raw_weights - np.max(raw_weights))
        weights = exp_w / exp_w.sum()
        
        # scaled to 0.5 - 1.5
        pos_size_mult = ((action[5] + 1.0) / 2.0) + 0.5 
        
        # scaled to 0.5 - 0.8
        conf_thresh = ((action[6] + 1.0) / 2.0) * 0.3 + 0.5
        
        # Calculate resulting action
        votes = snap.get("votes", {})
        buy_score = 0.0
        sell_score = 0.0
        
        for i, b in enumerate(BRAINS):
            v = votes.get(b, "hold")
            if v == "buy":
                buy_score += weights[i]
            elif v == "sell":
                sell_score += weights[i]
                
        final_action = "hold"
        _confidence = 0.0
        if buy_score > sell_score and buy_score > conf_thresh:
            final_action = "buy"
            _confidence = buy_score
        elif sell_score > buy_score and sell_score > conf_thresh:
            final_action = "sell"
            _confidence = sell_score
            
        # Compare to reality to get reward
        realized_pnl = snap.get("realized_pnl", 0.0)
        profitable_dir = "hold"
        if realized_pnl > 0:
            profitable_dir = snap.get("final_action", "hold")
        elif realized_pnl < 0:
            profitable_dir = "sell" if snap.get("final_action") == "buy" else "buy"
            
        reward = 0.0
        
        if final_action == "hold" or final_action == "stand_aside":
            # If we held, and the trade was a loser, we saved money!
            if realized_pnl < 0:
                reward = abs(realized_pnl) * 0.5 # small reward for dodging a bullet
        else:
            if final_action == profitable_dir:
                # We picked the winning direction. Reward is proportional to size multiplier
                reward = abs(realized_pnl) * pos_size_mult
            else:
                # We picked the losing direction. Penalty proportional to size multiplier
                reward = -abs(realized_pnl) * pos_size_mult
                
        self.current_step += 1
        done = self.current_step >= len(self.snapshots)
        
        return self._get_obs(), float(reward), done, False, {"action_taken": final_action}
