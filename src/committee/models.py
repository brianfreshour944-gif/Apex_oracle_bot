"""Data models for Committee voting system with confidence sizing and dynamic weighting."""

import math
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class BrainVote:
    name: str
    action: str        # "buy" | "sell" | "hold" | "stand_aside" | "skip"
    confidence: float  # 0.0 to 1.0
    weight: float      # Vote weight (e.g. 0.30 = 30%)
    regime: str
    reason: str
    is_veto: bool = False
    causal_reasoning: dict[str, float] | None = None
    tensor_state: list[list[float]] | None = None

@dataclass
class CommitteeResult:
    action: str                        # Final winning action ("buy", "sell", "hold", "stand_aside")
    score: float                       # Aggregated score of winning action (0.0 to 1.0)
    size_multiplier: float = 1.0       # Dynamic position sizing scale factor (e.g. 0.5x to 1.75x)
    entropy: float = 0.0               # Vote disagreement entropy
    votes: list[BrainVote] = field(default_factory=list)
    active_weights: dict[str, float] = field(default_factory=dict)
    vetoed: bool = False
    veto_reason: str | None = None
    # --- Adaptive meta-learner audit fields (populated when the learner runs) ---
    decision_id: str | None = None          # Correlates entry snapshot -> exit outcome
    adaptive_used: bool = False                # True if learned weights drove this decision
    adaptive_weights: dict[str, float] = field(default_factory=dict)  # per-brain weights used
    explanation: str | None = None          # Human-readable weighting rationale


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


def disagreement_from_entropy(entropy: float) -> str:
    """Map committee vote entropy to a discrete brain-disagreement level.

    Consumed by the adversarial veto in bot.py (HIGH + low score -> veto).
    Vote entropy excludes stand_aside/skip votes; with buy/sell/hold the max
    is log2(3)≈1.585. Thresholds: >0.8 HIGH (real conflict, e.g. near-even
    split), >0.3 MEDIUM (some dissent), else LOW (consensus).
    """
    if not (entropy == entropy):  # NaN guard
        entropy = 0.0
    if entropy > 0.8:
        return "HIGH"
    if entropy > 0.3:
        return "MEDIUM"
    return "LOW"
