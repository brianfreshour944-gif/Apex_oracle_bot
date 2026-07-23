"""Data models for Committee voting system with confidence sizing and dynamic weighting."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class BrainVote:
    name: str
    action: str        # "buy" | "sell" | "hold" | "stand_aside" | "skip"
    confidence: float  # 0.0 to 1.0
    weight: float      # Vote weight (e.g. 0.30 = 30%)
    regime: str
    reason: str
    is_veto: bool = False
    causal_reasoning: Optional[Dict[str, float]] = None

@dataclass
class CommitteeResult:
    action: str                        # Final winning action ("buy", "sell", "hold", "stand_aside")
    score: float                       # Aggregated score of winning action (0.0 to 1.0)
    size_multiplier: float = 1.0       # Dynamic position sizing scale factor (e.g. 0.5x to 1.75x)
    entropy: float = 0.0               # Vote disagreement entropy
    votes: List[BrainVote] = field(default_factory=list)
    active_weights: Dict[str, float] = field(default_factory=dict)
    vetoed: bool = False
    veto_reason: Optional[str] = None
    # --- Adaptive meta-learner audit fields (populated when the learner runs) ---
    decision_id: Optional[str] = None          # Correlates entry snapshot -> exit outcome
    adaptive_used: bool = False                # True if learned weights drove this decision
    adaptive_weights: Dict[str, float] = field(default_factory=dict)  # per-brain weights used
    explanation: Optional[str] = None          # Human-readable weighting rationale
