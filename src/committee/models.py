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


# Actions that express a directional market view. "hold" / "stand_aside" /
# "skip" are abstentions -- a brain taking no position is not disagreeing with
# anyone about direction.
DIRECTIONAL_ACTIONS = ("buy", "sell")


def calculate_directional_entropy(votes: list[BrainVote]) -> float:
    """Shannon entropy over DIRECTIONAL votes only (buy/sell).

    This is the metric the adversarial veto in bot.py must use, because that
    veto compares *disagreement* against the committee's *score*, and the two
    are only commensurate when they cover the same votes:

      - run_committee's score puts buy/sell confidence in the numerator but
        every non-abstention weight in the denominator, so an abstention
        (HOLD) dilutes the score.
      - calculate_vote_entropy counts HOLD as an opinion, so the same
        abstention simultaneously *inflates* the disagreement level.

    The result was a self-locking gate: the ordinary "one brain directional,
    three brains HOLD" pattern scored entropy 0.811 ("HIGH") while its score
    was diluted to ~0.23, so the veto rejected every symbol on every cycle.
    Reproduced against the live log: BTC/USD 0.811/0.230, SOL/USD 0.971/
    0.391, ETH/USD 1.371/0.195 -- all labeled VETO_ADVERSARIAL.

    Excluding abstentions makes HIGH mean "brains actively disagree about
    direction" -- 2 buy vs 2 sell -> 1.0 -> HIGH, a 4v1 minority dissent ->
    0.722 -> MEDIUM, and a lone directional vote among abstentions -> 0.0 ->
    LOW. The max is log2(2) = 1.0 rather than log2(3) = 1.585, so every
    threshold in disagreement_from_entropy() stays meaningful.

    calculate_vote_entropy() is deliberately left unchanged: it still feeds
    the sizing multiplier, Prometheus metrics and the OOD/decision-transformer
    state vectors, which expect the full 0..log2(3) range.
    """
    return calculate_vote_entropy(
        [v for v in votes if v.action in DIRECTIONAL_ACTIONS]
    )


def disagreement_from_entropy(entropy: float) -> str:
    """Map committee vote entropy to a discrete brain-disagreement level.

    Consumed by the adversarial veto in bot.py (HIGH + low score -> veto),
    which feeds it calculate_directional_entropy() -- see that function for
    why abstentions must not count as disagreement.

    Thresholds: >0.8 HIGH (real conflict, e.g. near-even split), >0.3 MEDIUM
    (some dissent), else LOW (consensus). calculate_vote_entropy() (the
    all-opinions variant, max log2(3)≈1.585) can also be mapped with these
    thresholds; the directional variant's own max is log2(2)=1.0.
    """
    if not (entropy == entropy):  # NaN guard
        entropy = 0.0
    if entropy > 0.8:
        return "HIGH"
    if entropy > 0.3:
        return "MEDIUM"
    return "LOW"
