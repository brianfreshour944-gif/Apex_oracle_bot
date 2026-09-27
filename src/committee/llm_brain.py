"""Brain 5: LLM Brain.

Secondary intelligence review layer (Gemini/Groq/Claude).
Acts as a confidence modifier or soft veto based on qualitative context.
"""

import json
from typing import Any

import httpx

from src.config import settings
from src.logging_config import get_logger

from .models import BrainVote

logger = get_logger("llm_brain")


async def _call_groq_llm(prompt: str, api_key: str, model: str = "llama-3.1-70b-versatile") -> dict[str, Any] | None:
    """Call Groq LLM API."""
    if not api_key:
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a crypto trading risk analyst. Output only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"}
                },
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"Groq LLM call failed: {e}")
    return None


async def _call_gemini_llm(prompt: str, api_key: str, model: str = "gemini-1.5-flash") -> dict[str, Any] | None:
    """Call Gemini LLM API."""
    if not api_key:
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500, "responseMimeType": "application/json"}
                },
            )
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logger.warning(f"Gemini LLM call failed: {e}")
    return None


async def _call_llm(prompt: str) -> dict[str, Any] | None:
    """Try LLM providers in order of preference."""
    # Try Groq first (fast, free tier)
    if settings.GROQ_API_KEY:
        result = await _call_groq_llm(prompt, settings.GROQ_API_KEY)
        if result:
            return result
    # Fallback to Gemini
    if settings.GEMINI_API_KEY:
        result = await _call_gemini_llm(prompt, settings.GEMINI_API_KEY)
        if result:
            return result
    return None


async def llm_brain(symbol: str, price: float, signal: dict) -> BrainVote:
    """Evaluates qualitative risk review and confidence adjustment.
    
    Can be connected to live LLM API (Gemini/Groq) for full reasoning.
    Falls back to rule-assisted qualitative brain stub.
    """
    raw_action = signal.get("action", "hold")
    regime = signal.get("regime", "neutral")
    features = signal.get("features", {})
    
    # Read the pre-fetched structured sentiment from features
    sentiment_score = features.get("sentiment_score", 0.0)
    event_type = features.get("event_type", "none")
    sentiment_conf = features.get("sentiment_conf", 0.0)
    
    # Try to get LLM analysis
    llm_result = None
    if settings.GROQ_API_KEY or settings.GEMINI_API_KEY:
        prompt = f"""Analyze this crypto trading signal for {symbol}:
Current price: ${price:,.2f}
Technical action: {raw_action}
Market regime: {regime}
News sentiment: {sentiment_score:.2f} (confidence: {sentiment_conf:.2f})
Event type: {event_type}

Output JSON with: {{"action": "buy|sell|hold|stand_aside", "confidence": 0.0-1.0, "reason": "explanation", "is_veto": true/false}}"""
        
        llm_result = await _call_llm(prompt)
        if llm_result:
            try:
                llm_result = json.loads(llm_result)
            except json.JSONDecodeError:
                llm_result = None
    
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
    elif llm_result and llm_result.get("action"):
        # Use LLM result if available and valid
        action = llm_result.get("action", raw_action)
        confidence = float(llm_result.get("confidence", 0.5))
        reason = llm_result.get("reason", "LLM review")
        is_veto = bool(llm_result.get("is_veto", False))
    else:
        # Normal trading conditions, use sentiment to vote
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
