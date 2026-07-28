"""AI Strategy Selector that uses Adaptive Meta-Learning to pick the best strategy."""

import os
from typing import Dict, Any, List
from src.logging_config import get_logger
from src.committee.adaptive_meta import AdaptiveMetaLearner
from src.execution_strategies import STRATEGIES

logger = get_logger("strategy_selector")

# Singleton instance
_STRATEGY_LEARNER = None

def get_strategy_learner() -> AdaptiveMetaLearner:
    """Returns the singleton adaptive learner for strategy selection."""
    global _STRATEGY_LEARNER
    if _STRATEGY_LEARNER is None:
        state_path = os.path.join("data", "strategy_meta_state.json")
        try:
            _STRATEGY_LEARNER = AdaptiveMetaLearner(
                state_path=state_path,
                learning_rate=0.10,
                min_weight=0.01,
                max_weight=0.80, # Allow strong dominance
            )
        except Exception as e:
            logger.error(f"Failed to load strategy learner: {e}")
            raise
    return _STRATEGY_LEARNER

def select_best_strategy(regime: str) -> str:
    """Select the best strategy for the current regime."""
    learner = get_strategy_learner()
    weights = learner._clamp_normalize(learner._regime_weights(regime))
    
    # Default logical priors based on market regime
    # If the meta-learner is uninitialized, we shouldn't just guess randomly.
    # We should seed it with domain knowledge.
    if not weights:
        strategies = list(STRATEGIES.keys())
        for s in strategies:
            weights[s] = learner.min_weight
            
        if regime in ["trending", "bull", "bear"]:
            weights["trend_following"] = 0.5
            weights["momentum"] = 0.3
        elif regime == "sideways":
            weights["mean_reversion"] = 0.5
            weights["grid"] = 0.3
        elif regime == "high_volatility":
            weights["breakout"] = 0.5
            weights["scalping"] = 0.3
        elif regime == "low_volatility":
            weights["grid"] = 0.5
            weights["mean_reversion"] = 0.3
        else: # neutral
            weights["trend_following"] = 0.2
            weights["mean_reversion"] = 0.2
            weights["momentum"] = 0.2
            
    # Always include all available strategies in the pool
    for s in STRATEGIES.keys():
        if s not in weights:
            weights[s] = learner.min_weight
            
    # Remove any stale strategies from old state files
    stale_keys = [k for k in weights.keys() if k not in STRATEGIES]
    for k in stale_keys:
        del weights[k]
            
    # Re-normalize to ensure they sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
        
    best_strategy = max(weights, key=weights.get)
    return best_strategy

def record_strategy_outcome(regime: str, strategy_name: str, action: str, pnl: float, return_pct: float) -> None:
    """Record the outcome of a trade to train the strategy selector.

    ``action`` must be the actual trade direction (``"buy"`` or ``"sell"``).
    Passing the wrong direction inverts the reward signal for the adaptive
    learner and biases it away from whichever side happened to work.
    """
    learner = get_strategy_learner()
    
    # We construct a mock decision snapshot format expected by AdaptiveMetaLearner.
    # AdaptiveMetaLearner expects brain_votes map. Here, the "brains" are our strategies.
    # If the strategy made money, we simulate that it voted correctly.
    # If it lost money, we simulate that it voted incorrectly.
    
    # Normalise action: only "buy" and "sell" are meaningful directions.
    # Fall back to "buy" for unexpected values so downstream code is safe.
    effective_action = action if action in ("buy", "sell") else "buy"

    # The learner expects { "brain_votes": {strategy: action, ...}, "final_action": action }
    # To reward our strategy, we make it agree with the final action.
    mock_votes = {}
    for strat in STRATEGIES.keys():
        if strat == strategy_name:
            mock_votes[strat] = effective_action  # agrees with final action
        else:
            mock_votes[strat] = "stand_aside"  # neutral
            
    decision_snapshot = {
        "regime": regime,
        "final_action": effective_action,  # use the actual trade direction
        "brain_votes": mock_votes
    }
    
    outcome = {
        "net_pnl": pnl,
        "return_pct": return_pct
    }
    
    try:
        report = learner.update(decision_snapshot, outcome)
        if report.material_change:
            logger.info(f"Strategy weights updated for regime {regime}")
    except Exception as e:
        logger.error(f"Failed to update strategy learner: {e}")
