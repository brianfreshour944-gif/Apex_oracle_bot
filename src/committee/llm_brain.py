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
    features = signal.get("features", {})
    
    # Read the pre-fetched structured sentiment from features
    sentiment_score = features.get("sentiment_score", 0.0)
    event_type = features.get("event_type", "none")
    sentiment_conf = features.get("sentiment_conf", 0.0)

    # Base evaluation logic based on technicals + LLM sentiment
    confidence = 0.50
    reason = "LLM review: Neutral qualitative outlook."
    action = "hold"
    is_veto = False

    # Security events (Hacks, Stolen funds, etc) are a HARD VETO
    if event_type == "security":
        action = "stand_aside"
        confidence = 0.95
        is_veto = True
        reason = "LLM VETO: Major security breach or hack detected in recent news."
    elif event_type == "regulation" and sentiment_score < -0.5:
        # Severe regulatory crackdown
        action = "stand_aside"
        confidence = 0.85
        is_veto = True
        reason = "LLM VETO: Severe negative regulatory event detected."
    else:
        # Normal trading conditions, let's use the sentiment to vote
        if sentiment_score > 0.4 and sentiment_conf > 0.5:
            action = "buy"
            confidence = 0.50 + (sentiment_score * 0.4) # scale up to 0.90
            reason = f"LLM review: Positive news sentiment ({event_type}, score: {sentiment_score:.2f})"
        elif sentiment_score < -0.4 and sentiment_conf > 0.5:
            action = "sell"
            confidence = 0.50 + (abs(sentiment_score) * 0.4)
            reason = f"LLM review: Negative news sentiment ({event_type}, score: {sentiment_score:.2f})"
        else:
            # If no strong news, defer to technical strategy action but low confidence
            action = raw_action if raw_action in ["buy", "sell", "hold"] else "hold"
            confidence = 0.40
            reason = f"LLM review: Deferring to technicals (Weak/No news, score: {sentiment_score:.2f})"

    return BrainVote(
        name="llm",
        action=action,
        confidence=confidence,
        weight=0.10,
        regime=regime,
        reason=reason,
        is_veto=is_veto
    )
