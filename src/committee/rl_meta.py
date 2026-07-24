"""Reinforcement Learning Meta-Learner (PPO).

Loads the trained stable-baselines3 PPO agent to dynamically determine
committee weights, position sizing, and confidence thresholds.
"""

import json
import asyncio
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from src.logging_config import get_logger
from .models import BrainVote
from .adaptive_meta import AdaptiveDecision

logger = get_logger("rl_meta")

BRAINS = ["transformer", "quant", "momentum", "sentinel", "llm"]
REGIMES = ["trending", "mean_reverting", "volatile", "choppy", "breakout", "default"]

_model = None

def get_ppo_model():
    global _model
    if _model is not None:
        return _model
        
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    save_path = os.path.join(models_dir, 'ppo_meta_weights.zip')
    
    if os.path.exists(save_path):
        try:
            from stable_baselines3 import PPO
            _model = PPO.load(save_path)
            logger.info("Loaded PPO Meta-Learner weights successfully.")
        except Exception as e:
            logger.error(f"Failed to load PPO model: {e}")
            _model = False
    else:
        _model = False
        
    return _model


class RLMetaLearner:
    """Predicts optimal committee weights and position size using a trained PPO agent."""
    
    def __init__(self):
        self.model = get_ppo_model()
        
    def _build_obs(self, brain_outputs: List[BrainVote], regime: str, features: Dict[str, Any]) -> np.ndarray:
        # 1. Regime One-Hot
        regime_vec = np.zeros(len(REGIMES), dtype=np.float32)
        if regime in REGIMES:
            regime_vec[REGIMES.index(regime)] = 1.0
            
        # 2. Features
        rsi = (features.get("rsi", 50.0) - 50) / 50.0  
        atr = features.get("atr", 0.0) / 100.0 
        macd = np.clip(features.get("macd", 0.0), -1.0, 1.0)
        
        # On-Chain Features
        fr = np.clip(features.get("funding_rate", 0.0) * 1000, -1.0, 1.0) # Scale funding rate
        oi = np.clip(features.get("open_interest", 0.0) / 1e9, 0.0, 10.0) # Scale OI
        lsr = np.clip((features.get("long_short_ratio", 1.0) - 1.0), -1.0, 1.0) # Center LSR at 0
        imb = np.clip(features.get("bid_ask_imbalance", 0.0), -1.0, 1.0) # L2 Imbalance
        
        # Sentiment Features
        sent_score = np.clip(features.get("sentiment_score", 0.0), -1.0, 1.0)
        sent_conf = np.clip(features.get("sentiment_conf", 0.0), 0.0, 1.0)
        
        event_types = ["earnings", "regulation", "macro", "security", "adoption", "none"]
        event = features.get("event_type", "none")
        event_vec = np.zeros(len(event_types), dtype=np.float32)
        if event in event_types:
            event_vec[event_types.index(event)] = 1.0
            
        feature_vec = np.array([rsi, atr, macd, fr, oi, lsr, imb, sent_score, sent_conf], dtype=np.float32)
        
        # 3. Brain Votes
        votes = {v.name: v.action for v in brain_outputs}
        vote_vec = np.zeros(len(BRAINS), dtype=np.float32)
        for i, b in enumerate(BRAINS):
            v = votes.get(b, "hold")
            if v == "buy":
                vote_vec[i] = 1.0
            elif v == "sell":
                vote_vec[i] = -1.0
                
        obs = np.concatenate([regime_vec, feature_vec, event_vec, vote_vec])
        return np.nan_to_num(obs, 0.0).astype(np.float32)

    def combine(self, brain_outputs: List[BrainVote], regime: str, features: Dict[str, Any]) -> AdaptiveDecision:
        if not self.model:
            # Fallback to simple equal weights if PPO isn't trained yet
            from .adaptive_meta import BrainScore
            
            action_scores = {}
            for v in brain_outputs:
                if v.action in ["buy", "sell"]:
                    action_scores[v.action] = action_scores.get(v.action, 0.0) + (v.confidence * 0.2)
                    
            action = max(action_scores, key=action_scores.get) if action_scores else "stand_aside"
            confidence = action_scores.get(action, 0.0)
            
            return AdaptiveDecision(
                action=action,
                confidence=confidence,
                regime=regime,
                weights={b: 0.2 for b in BRAINS},
                explanation="PPO model not loaded. Fallback equal weights."
            )
            
        # Inference
        obs = self._build_obs(brain_outputs, regime, features)
        action, _states = self.model.predict(obs, deterministic=True)
        
        # Extract actions
        raw_weights = action[0:5]
        exp_w = np.exp(raw_weights - np.max(raw_weights))
        weights_arr = exp_w / exp_w.sum()
        
        pos_size_mult = ((action[5] + 1.0) / 2.0) + 0.5 
        conf_thresh = ((action[6] + 1.0) / 2.0) * 0.3 + 0.5
        
        weights = {BRAINS[i]: float(weights_arr[i]) for i in range(len(BRAINS))}
        
        # Calculate resulting action
        action_scores = {}
        from .adaptive_meta import BrainScore
        scores = []
        
        for v in brain_outputs:
            w = weights.get(v.name, 0.2)
            scores.append(BrainScore(name=v.name, action=v.action, confidence=float(v.confidence), weight=w))
            if v.action in ["buy", "sell"]:
                action_scores[v.action] = action_scores.get(v.action, 0.0) + (v.confidence * w)
                
        final_action = "stand_aside"
        confidence = 0.0
        
        if action_scores:
            best_action = max(action_scores, key=action_scores.get)
            best_conf = action_scores[best_action]
            
            if best_conf > conf_thresh:
                final_action = best_action
                confidence = best_conf
                
        explanation = f"RL_PPO[{regime}] {final_action}={confidence:.3f} | sz={pos_size_mult:.2f}x | thresh={conf_thresh:.2f}"
        
        # We inject the size multiplier into the AdaptiveDecision class by modifying it dynamically
        decision = AdaptiveDecision(
            action=final_action,
            confidence=confidence,
            regime=regime,
            weights=weights,
            scores=scores,
            explanation=explanation
        )
        # Monkey patch the pos_size_mult so bot.py can read it
        decision.pos_size_mult = float(pos_size_mult)
        
        return decision
