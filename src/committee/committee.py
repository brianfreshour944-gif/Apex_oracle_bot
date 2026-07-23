"""Advanced Committee Aggregator Engine.

Features:
1. Dynamic Regime-Adaptive Weight Allocation Matrix.
2. Vote Disagreement Entropy Calculation & Confidence Penalty.
3. Dynamic Position Sizing Multiplier (Kelly-inspired Confidence Sizing: 0.5x to 1.75x).
4. Hard Veto Sentinel Filter.
"""

import math
from collections import defaultdict
from typing import Dict, Any

from .models import CommitteeResult, BrainVote
from .transformer_brain import transformer_brain
from .quant_brain import quant_brain
from .momentum_brain import momentum_brain
from .sentinel_brain import sentinel_brain
from .llm_brain import llm_brain

WINNING_SCORE_THRESHOLD = 0.60

# Dynamic Weight Matrix based on Market Regime
REGIME_WEIGHT_MATRIX = {
    "uptrend": {
        "transformer": 0.40,
        "quant": 0.20,
        "momentum": 0.30,
        "sentinel": 0.05,
        "llm": 0.05
    },
    "downtrend": {
        "transformer": 0.40,
        "quant": 0.20,
        "momentum": 0.30,
        "sentinel": 0.05,
        "llm": 0.05
    },
    "dump_to_accumulation": {
        "transformer": 0.30,
        "quant": 0.30,
        "momentum": 0.30,
        "sentinel": 0.05,
        "llm": 0.05
    },
    "uptrend_to_distribution": {
        "transformer": 0.30,
        "quant": 0.30,
        "momentum": 0.30,
        "sentinel": 0.05,
        "llm": 0.05
    },
    "quiet": {
        "transformer": 0.20,
        "quant": 0.45,
        "momentum": 0.10,
        "sentinel": 0.10,
        "llm": 0.15
    },
    "ranging": {
        "transformer": 0.20,
        "quant": 0.45,
        "momentum": 0.10,
        "sentinel": 0.10,
        "llm": 0.15
    },
    "dump": {
        "transformer": 0.15,
        "quant": 0.15,
        "momentum": 0.20,
        "sentinel": 0.40,
        "llm": 0.10
    },
    "crash": {
        "transformer": 0.10,
        "quant": 0.10,
        "momentum": 0.20,
        "sentinel": 0.50,
        "llm": 0.10
    },
    "default": {
        "transformer": 0.35,
        "quant": 0.25,
        "momentum": 0.20,
        "sentinel": 0.10,
        "llm": 0.10
    }
}

def calculate_vote_entropy(votes: list[BrainVote]) -> float:
    """Calculates Shannon Entropy across brain actions to measure consensus conflict."""
    actions = [v.action for v in votes if v.action not in ["stand_aside", "skip"]]
    if not actions:
        return 0.0
    
    counts = defaultdict(int)
    for a in actions:
        counts[a] += 1
        
    entropy = 0.0
    total = len(actions)
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
        
    return entropy

def calculate_confidence_size_multiplier(score: float, entropy: float) -> float:
    """Calculates dynamic position sizing multiplier based on committee conviction score.
    
    Formula:
    - Base score threshold: 0.60
    - Multiplier ranges from 0.50x (marginal confidence) to 1.75x (unanimous conviction).
    - High vote entropy (>0.8) penalizes multiplier by up to 25%.
    """
    if score < WINNING_SCORE_THRESHOLD:
        return 0.0
    
    # Linear scale from 0.60 score -> 0.50x to 1.00 score -> 1.75x
    normalized_score = (score - WINNING_SCORE_THRESHOLD) / (1.0 - WINNING_SCORE_THRESHOLD)
    base_mult = 0.50 + normalized_score * 1.25  # 0.50 to 1.75
    
    # Entropy penalty for vote conflict
    if entropy > 0.5:
        penalty = min((entropy - 0.5) * 0.30, 0.25)
        base_mult *= (1.0 - penalty)
        
    return max(0.40, min(1.75, round(base_mult, 2)))

async def run_committee(symbol: str, price: float, signal: Dict[str, Any]) -> CommitteeResult:
    """Runs the 5-brain ensemble committee with dynamic weighting and confidence position sizing.
    
    Returns:
        CommitteeResult containing final action, weighted score, size_multiplier, and vote audit logs.
    """
    regime = signal.get("regime", "default")
    weights = REGIME_WEIGHT_MATRIX.get(regime, REGIME_WEIGHT_MATRIX["default"])

    # Run all 5 brains asynchronously
    raw_votes = [
        await transformer_brain(symbol, price, signal),
        await quant_brain(symbol, price, signal),
        await momentum_brain(symbol, price, signal),
        await sentinel_brain(symbol, price, signal),
        await llm_brain(symbol, price, signal),
    ]

    # Re-weight votes dynamically based on active regime matrix
    votes = []
    for v in raw_votes:
        w = weights.get(v.name, v.weight)
        votes.append(BrainVote(
            name=v.name,
            action=v.action,
            confidence=v.confidence,
            weight=w,
            regime=v.regime,
            reason=v.reason,
            is_veto=v.is_veto
        ))

    # Check for hard vetoes (Sentinel / Crash)
    for v in votes:
        if v.is_veto or (v.name == "sentinel" and v.action == "stand_aside" and v.confidence >= 0.85):
            return CommitteeResult(
                action="stand_aside",
                score=0.0,
                size_multiplier=0.0,
                entropy=0.0,
                votes=votes,
                active_weights=weights,
                vetoed=True,
                veto_reason=v.reason
            )

    # Compute weighted score per action
    scores = defaultdict(float)
    for v in votes:
        if v.action not in ["stand_aside", "skip"]:
            scores[v.action] += v.confidence * v.weight

    if not scores:
        return CommitteeResult(
            action="stand_aside",
            score=0.0,
            size_multiplier=0.0,
            entropy=0.0,
            votes=votes,
            active_weights=weights
        )

    winner = max(scores, key=scores.get)
    score = scores[winner]
    entropy = calculate_vote_entropy(votes)

    if score < WINNING_SCORE_THRESHOLD:
        final_action = "stand_aside"
        size_mult = 0.0
    else:
        final_action = winner
        size_mult = calculate_confidence_size_multiplier(score, entropy)

    return CommitteeResult(
        action=final_action,
        score=score,
        size_multiplier=size_mult,
        entropy=entropy,
        votes=votes,
        active_weights=weights
    )
