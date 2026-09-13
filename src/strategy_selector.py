"""AI Strategy Selector that uses Adaptive Meta-Learning to pick the best strategy."""

import os
from datetime import datetime
from typing import Any

from src.committee.adaptive_meta import AdaptiveMetaLearner
from src.config import settings
from src.execution_strategies import STRATEGIES
from src.logging_config import get_logger
from src.performance_tracker import record_trade_outcome

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


def _estimate_strategy_costs(strategy_name: str, regime: str, atr_pct: float) -> float:
    """
    Estimate relative transaction cost burden for a strategy in a given regime.
    Returns a multiplier (1.0 = baseline, >1.0 = higher cost burden).
    """
    # Base cost per round trip in bps
    base_cost_bps = settings.TX_COST_FEE_BPS + settings.TX_COST_SLIPPAGE_BPS + settings.TX_COST_SPREAD_BPS
    
    # Strategy-specific trade frequency multipliers (relative to trend_following)
    freq_multipliers = {
        "trend_following": 1.0,      # ~1-2 trades/day max
        "mean_reversion": 1.5,       # ~2-3 trades/day in chop
        "momentum": 1.3,             # ~1-2 trades/day
        "breakout": 1.2,             # ~1 trade/day
        "grid": 3.0,                 # Many small trades in range
        "scalping": 5.0,             # Very high frequency
    }
    
    freq_mult = freq_multipliers.get(strategy_name, 1.0)
    
    # Regime adjustments: in low vol, spread costs dominate; in high vol, slippage dominates
    if regime == "low_volatility":
        # Spread is relatively larger vs moves
        regime_mult = 1.5
    elif regime == "high_volatility":
        # Slippage is larger
        regime_mult = 1.3
    else:
        regime_mult = 1.0
    
    # ATR adjustment: higher ATR means larger moves, costs are smaller relative to move
    atr_adj = max(0.5, min(2.0, 2.0 / max(atr_pct, 0.5)))
    
    return freq_mult * regime_mult * atr_adj


def select_best_strategy(regime: str, features: dict[str, Any] | None = None) -> str:
    """Select the best strategy for the current regime with cost-awareness."""
    learner = get_strategy_learner()
    weights = learner._clamp_normalize(learner._regime_weights(regime))
    
    # Extract regime features for cost-aware selection
    atr_pct = features.get("atr", 0.0) / features.get("close", 1.0) * 100 if features else 1.0
    in_transition = features.get("in_transition", False) if features else False
    hurst_velocity = features.get("hurst_velocity", 0.0) if features else 0.0
    
    # Default logical priors based on market regime
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
    
    # Apply cost-aware penalties
    for strat_name in weights:
        cost_mult = _estimate_strategy_costs(strat_name, regime, atr_pct)
        # Penalty increases with cost multiplier; cap at 50% reduction
        penalty = min(0.5, (cost_mult - 1.0) * 0.2)
        weights[strat_name] *= (1.0 - penalty)
    
    # During regime transitions, penalize ALL strategies to reduce conviction
    # and favor the previous strategy (handled by _active_strategy in TradingStrategy)
    if in_transition:
        for strat_name in weights:
            weights[strat_name] *= 0.7  # Reduce all weights during transition
    
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
    
    # Normalise action: only "buy" and "sell" are meaningful directions.
    # Fall back to "buy" for unexpected values so downstream code is safe.
    effective_action = action if action in ("buy", "sell") else "buy"

    # Reward signal: strategy made money on this trade? (not "did it agree with final action")
    # This avoids circular logic where the strategy's mock vote is forced to match the action.
    profitable = pnl > 0
    
    # Construct votes: each strategy gets a directional vote based on its actual signal
    # at entry time. Since we only know which strategy was SELECTED, we simulate:
    # - The selected strategy voted in the trade direction (it caused the trade)
    # - Other strategies vote "stand_aside" (we don't know their hypothetical signals)
    # Reward is based on PnL, not vote agreement.
    mock_votes = {}
    for strat in STRATEGIES.keys():
        if strat == strategy_name:
            mock_votes[strat] = effective_action  # this strategy drove the trade
        else:
            mock_votes[strat] = "stand_aside"  # unknown, neutral
            
    decision_snapshot = {
        "regime": regime,
        "final_action": effective_action,
        "brain_votes": mock_votes
    }
    
    outcome = {
        "net_pnl": pnl,
        "return_pct": return_pct
    }
    
    try:
        report = learner.update(decision_snapshot, outcome)
        if report.material_change:
            logger.info(f"Strategy weights updated for regime {regime}: {strategy_name} {'profitable' if profitable else 'loss'}")
    except Exception as e:
        logger.error(f"Failed to update strategy learner: {e}")

    # Also record in performance tracker for decay monitoring
    try:
        record_trade_outcome(strategy_name, regime, pnl, return_pct, datetime.utcnow())
    except Exception as e:
        logger.error(f"Failed to record trade outcome in performance tracker: {e}")
