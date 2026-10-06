"""Reinforcement Learning Meta-Learner (PPO).

Loads the trained stable-baselines3 PPO agent to dynamically determine
committee weights, position sizing, and confidence thresholds.
"""

import os
from typing import Any

import numpy as np

from src import model_store
from src.logging_config import get_logger

from .adaptive_meta import AdaptiveDecision
from .models import BrainVote
from .regime_utils import build_rl_observation, confidence_threshold, position_size_multiplier
from .regime_utils import normalize_regime as normalize_regime

logger = get_logger("rl_meta")

BRAINS = ["transformer", "quant", "momentum", "sentinel", "llm"]
REGIMES = ["trending", "mean_reverting", "volatile", "choppy", "breakout", "default"]

_model = None
# Separate from `_model` itself: `_model` being None means either "never
# tried" or "tried and failed", and callers (decision_gate.py's
# `getattr(rl_learner, "model", None) is not None` readiness check) need
# None to mean "not loaded" unambiguously. Previously this function cached
# a failed load as `_model = False` -- `False is not None` is True, so the
# committee gate reported "PPO model loaded" for a model that was never
# actually loaded. stable_baselines3 isn't in requirements.txt/pyproject.toml
# at all, so a load failure via missing import is the default production
# state, not an edge case. Found via an external correctness audit,
# reproduced directly (get_ppo_model() -> False, gate -> "PPO model
# loaded"), 2026-09-22.
_model_load_attempted = False

def get_ppo_model():
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True

    save_path = model_store.bundle_path("ppo", "ppo_meta_weights.zip")

    if os.path.exists(save_path):
        try:
            from stable_baselines3 import PPO
            _model = PPO.load(save_path)
            logger.info("Loaded PPO Meta-Learner weights successfully.")
        except Exception as e:
            logger.error(f"Failed to load PPO model: {e}")
            _model = None
    else:
        _model = None

    return _model


def reset_ppo_model():
    """Clear the cached PPO model so the next call reloads the active weights."""
    global _model, _model_load_attempted
    _model = None
    _model_load_attempted = False


class RLMetaLearner:
    """Predicts optimal committee weights and position size using a trained PPO agent."""
    
    def __init__(self):
        self.model = get_ppo_model()
        
    def _build_obs(self, brain_outputs: list[BrainVote], regime: str, features: dict[str, Any]) -> np.ndarray:
        # Delegate to the shared 26-dim builder so this stays byte-for-byte
        # aligned with rl_env's training observations. They used to be built
        # independently and drifted (26 live vs 17 training), which silently
        # blocked every weekly PPO promotion -- see regime_utils docstring.
        votes = {v.name: v.action for v in brain_outputs}
        return build_rl_observation(regime, features, votes)

    def combine(self, brain_outputs: list[BrainVote], regime: str, features: dict[str, Any]) -> AdaptiveDecision:
        if not self.model:
            # Fallback to simple equal weights if PPO isn't trained yet
            from .adaptive_meta import BrainScore
            
            action_scores = {}
            active_weight = 0.0
            for v in brain_outputs:
                if v.action in ["buy", "sell", "hold"]:
                    action_scores[v.action] = action_scores.get(v.action, 0.0) + (v.confidence * 0.2)
                    active_weight += 0.2
                    
            if active_weight > 0:
                for action in action_scores:
                    action_scores[action] /= active_weight
                    
            action = max(action_scores, key=action_scores.get) if action_scores else "stand_aside"
            confidence = action_scores.get(action, 0.0)
            
            return AdaptiveDecision(
                action=action,
                confidence=confidence,
                regime=regime,
                weights=dict.fromkeys(BRAINS, 0.2),
                explanation="PPO model not loaded. Fallback equal weights."
            )
            
        # Inference
        obs = self._build_obs(brain_outputs, regime, features)
        action, _states = self.model.predict(obs, deterministic=True)
        
        # Extract actions
        raw_weights = action[0:5]
        exp_w = np.exp(raw_weights - np.max(raw_weights))
        weights_arr = exp_w / exp_w.sum()
        
        pos_size_mult = position_size_multiplier(action[5])
        conf_thresh = confidence_threshold(action[6])
        
        weights = {BRAINS[i]: float(weights_arr[i]) for i in range(len(BRAINS))}
        
        # Calculate resulting action
        action_scores = {}
        from .adaptive_meta import BrainScore
        scores = []
        
        active_weight = 0.0
        for v in brain_outputs:
            w = weights.get(v.name, 0.2)
            scores.append(BrainScore(name=v.name, action=v.action, confidence=float(v.confidence), weight=w))
            if v.action in ["buy", "sell", "hold"]:
                action_scores[v.action] = action_scores.get(v.action, 0.0) + (v.confidence * w)
                active_weight += w
                
        if active_weight > 0:
            for action in action_scores:
                action_scores[action] /= active_weight
                
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
