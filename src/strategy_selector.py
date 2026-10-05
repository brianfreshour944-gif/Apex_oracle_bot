"""AI Strategy Selector that uses Adaptive Meta-Learning to pick the best strategy."""

import os
from datetime import UTC, datetime
from typing import Any

from src.committee.adaptive_meta import AdaptiveMetaLearner
from src.execution_strategies import STRATEGIES
from src.logging_config import get_logger
from src.performance_tracker import record_trade_outcome

logger = get_logger("strategy_selector")

# Singleton instance
_STRATEGY_LEARNER = None

# Default logical priors by market regime (all other strategies: min_weight).
_REGIME_PRIORS: dict[str, dict[str, float]] = {
    "trending": {"trend_following": 0.5, "momentum": 0.3},
    "bull": {"trend_following": 0.5, "momentum": 0.3},
    "bear": {"trend_following": 0.5, "momentum": 0.3},
    "sideways": {"mean_reversion": 0.5, "grid": 0.3},
    "high_volatility": {"breakout": 0.5, "scalping": 0.3},
    "low_volatility": {"grid": 0.5, "mean_reversion": 0.3},
}
_NEUTRAL_PRIOR = {"trend_following": 0.2, "mean_reversion": 0.2, "momentum": 0.2}

# Committee brain names that only ever got into strategy state via the base
# class's brain-name seeding ("momentum" is both, so it isn't listed).
_BRAIN_ONLY_NAMES = {"transformer", "quant", "sentinel", "llm"}


def _prior_weights(regime: str, min_weight: float) -> dict[str, float]:
    weights = dict.fromkeys(STRATEGIES, min_weight)
    for name, w in _REGIME_PRIORS.get(regime, _NEUTRAL_PRIOR).items():
        if name in weights:
            weights[name] = w
    return weights


class StrategyMetaLearner(AdaptiveMetaLearner):
    """AdaptiveMetaLearner keyed by execution STRATEGIES, not committee brains.

    The base class seeds a new regime with equal weights for the committee
    brains (transformer/quant/momentum/sentinel/llm). Used as the strategy
    learner that meant the weights were never empty, so the regime priors in
    select_best_strategy never ran, and "momentum" -- the one name that is
    both a brain and a strategy -- kept the brain's weight while every real
    strategy sat at min_weight: momentum was selected in every regime
    (backtest probe 2026-10-04: 4,320/4,320 hours). Seed from the regime
    priors instead, and reset any regime whose saved weights were
    brain-seeded (its "momentum" value can't be trusted).
    """

    def _regime_weights(self, regime: str) -> dict[str, float]:
        weights = self.weights.get(regime)
        if not weights or _BRAIN_ONLY_NAMES & set(weights):
            if weights:
                logger.warning(f"Strategy learner: regime '{regime}' had brain-seeded weights; reset to strategy priors")
            weights = _prior_weights(regime, self.min_weight)
        else:
            weights = {k: v for k, v in weights.items() if k in STRATEGIES}
            for s in STRATEGIES:
                weights.setdefault(s, self.min_weight)
        self.weights[regime] = weights
        return weights

    def _clamp_normalize(self, weights: dict[str, float]) -> dict[str, float]:
        if sum(weights.values()) <= 0:
            return dict.fromkeys(STRATEGIES, 1.0 / len(STRATEGIES))
        return super()._clamp_normalize(weights)


def get_strategy_learner() -> AdaptiveMetaLearner:
    """Returns the singleton adaptive learner for strategy selection."""
    global _STRATEGY_LEARNER
    if _STRATEGY_LEARNER is None:
        state_path = os.path.join("data", "strategy_meta_state.json")
        try:
            _STRATEGY_LEARNER = StrategyMetaLearner(
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
    
    # Extract regime features for cost-aware selection.
    # `features.get("close", 1.0)` looks safe but isn't: the default only
    # applies when the KEY is missing, not when it's present with value 0.0 --
    # exactly what analyze_market_regime's insufficient-data fallback returns
    # (`"close": 0.0`). That produced a real 0.0/0.0 ZeroDivisionError in
    # production (caught by generate_trading_signal's broad try/except, so it
    # degraded safely to stand_aside, but the bot silently stopped evaluating
    # cost-aware strategy selection every time this fired). `or 1.0` handles
    # both "key missing" and "key present but falsy/zero". Fixed 2026-09-21.
    close_price = (features.get("close") or 1.0) if features else 1.0
    atr_pct = features.get("atr", 0.0) / close_price * 100 if features else 1.0
    in_transition = features.get("in_transition", False) if features else False

    # Regime priors are the learner's starting weights (StrategyMetaLearner).
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

    # Continuous reward instead of binary (prevents overfitting to small wins)
    # Scale by return_pct for normalized reward [-1, 1] range
    reward_score = max(-1.0, min(1.0, return_pct / 10.0))
    profitable = reward_score > 0  # Keep binary for log message only
    
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
        record_trade_outcome(strategy_name, regime, pnl, return_pct, datetime.now(UTC))
    except Exception as e:
        logger.error(f"Failed to record trade outcome in performance tracker: {e}")
