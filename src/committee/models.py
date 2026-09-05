"""Data models for Committee voting system with confidence sizing and dynamic weighting."""

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
