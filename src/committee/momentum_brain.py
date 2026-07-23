"""Brain 3: Momentum / Regime Brain.

Detects structural market regime transitions and trend states.
"""

from .models import BrainVote

async def momentum_brain(symbol: str, price: float, signal: dict) -> BrainVote:
    """Evaluates market regime transitions and trend momentum."""
    regime = signal.get("regime", "neutral")

    if regime == "dump_to_accumulation":
        return BrainVote("momentum", "buy", 0.90, 0.20, regime, "Regime DUMP→ACCUM bottom reversal")
    elif regime == "uptrend_to_distribution":
        return BrainVote("momentum", "sell", 0.92, 0.20, regime, "Regime UPTREND→DIST top exhaustion")
    elif regime in ["dump", "crash"]:
        return BrainVote("momentum", "stand_aside", 0.85, 0.20, regime, "Avoid falling knife / market crash")
    elif regime == "uptrend":
        return BrainVote("momentum", "buy", 0.75, 0.20, regime, "Uptrend continuation")
    elif regime == "downtrend":
        return BrainVote("momentum", "sell", 0.75, 0.20, regime, "Downtrend continuation")
    elif regime == "quiet":
        return BrainVote("momentum", "stand_aside", 0.60, 0.20, regime, "Low liquidity / quiet market")

    return BrainVote("momentum", "stand_aside", 0.50, 0.20, regime, f"Neutral regime ({regime})")
