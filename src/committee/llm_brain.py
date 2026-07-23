"""Brain 5: LLM Brain.

Secondary intelligence review layer (Gemini/Groq/Claude).
Acts as a confidence modifier or soft veto based on qualitative context.
"""

from .models import BrainVote

async def llm_brain(symbol: str, price: float, signal: dict) -> BrainVote:
    """Evaluates qualitative risk review and confidence adjustment.
    
    Can be connected to live LLM API (Gemini/Groq) for full reasoning.
    Currently operates as a rule-assisted qualitative brain stub.
    """
    raw_action = signal.get("action", "hold")
    regime = signal.get("regime", "neutral")
    rsi = signal.get("rsi")

    # Basic qualitative evaluation logic
    confidence = 0.60
    reason = "LLM review: Market structure matches standard strategy"

    if rsi is not None and (rsi < 20 or rsi > 80):
        # Extreme RSI territory
        confidence = 0.75
        reason = f"LLM review: High conviction signal at extreme RSI ({rsi:.1f})"

    return BrainVote(
        name="llm",
        action=raw_action if raw_action in ["buy", "sell", "hold"] else "hold",
        confidence=confidence,
        weight=0.10,
        regime=regime,
        reason=reason
    )
