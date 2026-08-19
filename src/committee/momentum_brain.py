"""Brain 3: Momentum / Regime Brain.

Detects structural market regime transitions and trend states.
"""

from .models import BrainVote

async def momentum_brain(symbol: str, price: float, signal: dict) -> BrainVote:
    """Evaluates market regime transitions and trend momentum."""
    regime = signal.get("regime", "neutral")

    if regime == "bull":
        return BrainVote("momentum", "buy", 0.85, 0.20, regime, "Bull market continuation")
    elif regime == "bear":
        return BrainVote("momentum", "sell", 0.85, 0.20, regime, "Bear market continuation")
    elif regime == "trending":
        return BrainVote("momentum", "buy", 0.75, 0.20, regime, "Directional trend continuation")
    elif regime == "high_volatility":
        return BrainVote("momentum", "stand_aside", 0.85, 0.20, regime, "Avoid high volatility chop")
    elif regime == "low_volatility":
        return BrainVote("momentum", "stand_aside", 0.60, 0.20, regime, "Low liquidity / quiet market")
    elif regime == "sideways":
        # In sideways markets, momentum shouldn't just stand aside --
        # it should vote based on the underlying trend direction.
        # Standing aside in sideways markets means the committee
        # loses a brain that could provide directional context.
        return BrainVote("momentum", "hold", 0.45, 0.20, regime, "Sideways mean-reverting market")

    return BrainVote("momentum", "stand_aside", 0.50, 0.20, regime, f"Neutral regime ({regime})")
