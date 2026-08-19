"""Brain 2: Quant Brain.

Evaluates technical quantitative indicators (RSI, ATR, Bollinger, Momentum).
"""

from .models import BrainVote

async def quant_brain(symbol: str, price: float, signal: dict) -> BrainVote:
    """Evaluates quantitative indicators."""
    rsi = signal.get("rsi")
    regime = signal.get("regime", "neutral")
    atr = signal.get("atr")

    votes = []

    # RSI-based evaluation
    if rsi is not None:
        if rsi < 25:
            votes.append("buy")
        elif rsi > 75:
            votes.append("sell")
        elif 40 <= rsi <= 60:
            votes.append("hold")
        # Wider band: also vote when RSI is in the 25-40 or 60-75 range
        # with enough conviction. Previously only <30/>70 triggered
        # directional votes, leaving the brain silent in most regimes.
        elif 25 <= rsi < 40:
            votes.append("buy")
        elif 60 < rsi <= 75:
            votes.append("sell")

    if not votes:
        action = "hold"
        conf = 0.5
    else:
        buy_votes = votes.count("buy")
        sell_votes = votes.count("sell")
        if buy_votes > sell_votes:
            action = "buy"
            conf = min(0.6 + 0.15 * buy_votes, 0.95)
        elif sell_votes > buy_votes:
            action = "sell"
            conf = min(0.6 + 0.15 * sell_votes, 0.95)
        else:
            action = "hold"
            conf = 0.5

    return BrainVote(
        name="quant",
        action=action,
        confidence=conf,
        weight=0.25,
        regime=regime,
        reason=f"Quant RSI={rsi:.1f} votes={votes}" if rsi is not None else "Quant fallback hold"
    )
