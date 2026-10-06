"""Reinforcement Learning Environment for Meta-Decisions.

A standard Gymnasium environment where the agent learns to optimally
weight the committee brains and size positions based on historical market snapshots.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .regime_utils import (
    RL_OBS_DIM,
    build_rl_observation,
    confidence_threshold,
    position_size_multiplier,
)
from .regime_utils import normalize_regime as normalize_regime

BRAINS = ["transformer", "quant", "momentum", "sentinel", "llm"]
REGIMES = ["trending", "mean_reverting", "volatile", "choppy", "breakout", "default"]

class MetaDecisionEnv(gym.Env):
    """
    State (26 dims, built by regime_utils.build_rl_observation so it is
    byte-for-byte identical to the live learner RLMetaLearner._build_obs):
      - Regime (One-hot encoded, 6 dims)
      - Numeric features (RSI, ATR, MACD, on-chain, sentiment) (9 dims)
      - Event type (one-hot, 6 dims)
      - Brain votes (Buy=1, Sell=-1, Hold=0) (5 dims)
      Total Observation Space = 26 dimensions

    Action:
      - 5 Brain Weights (softmaxed internally to sum to 1)
      - Position Size Multiplier (scaled to 0.5 - 1.5)
      - Confidence Threshold (scaled to 0.35 - 0.55, matching rl_meta.combine)
      Total Action Space = 7 continuous dimensions

    Reward:
      - Realized PnL of the resulting theoretical trade
    """

    def __init__(self, historical_snapshots: list[dict[str, Any]]):
        super().__init__()

        self.snapshots = historical_snapshots
        self.current_step = 0

        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(RL_OBS_DIM,), dtype=np.float32)

        # Action Space: 7 continuous variables between -1 and 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    def _get_obs(self):
        if self.current_step >= len(self.snapshots):
            return np.zeros(RL_OBS_DIM, dtype=np.float32)

        snap = self.snapshots[self.current_step]
        votes = snap.get("votes", snap.get("brain_votes", {}))
        features = dict(snap.get("features", {}) or {})
        # Snapshot-level confidence/sentiment feed the numeric feature slots.
        features.setdefault("sentiment_score", snap.get("sentiment_score", 0.0))
        return build_rl_observation(snap.get("regime", "default"), features, votes)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        if self.current_step >= len(self.snapshots):
            return np.zeros(RL_OBS_DIM, dtype=np.float32), 0.0, True, False, {}

        snap = self.snapshots[self.current_step]

        # Extract actions
        raw_weights = action[0:5]
        # softmax the weights
        exp_w = np.exp(raw_weights - np.max(raw_weights))
        weights = exp_w / exp_w.sum()

        # Shared mappings with rl_meta.combine so a policy learned here means
        # the same thing live.
        pos_size_mult = position_size_multiplier(action[5])
        conf_thresh = confidence_threshold(action[6])

        # Calculate resulting action
        votes = snap.get("votes", snap.get("brain_votes", {}))
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
