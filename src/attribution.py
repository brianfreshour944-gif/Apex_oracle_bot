"""Causal Attribution Engine.

Analyzes closed trades and uses an LLM to explain why the trade succeeded or failed
based on the committee's votes and the PyTorch Transformer's causal reasoning gradients.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional
from sqlalchemy import select

from src.db import get_db_session, DecisionSnapshot
from src.config import settings
from src.logging_config import get_logger

logger = get_logger("attribution")

async def analyze_closed_trade(decision_id: str) -> Optional[Dict[str, Any]]:
    """Analyzes a single closed trade to provide causal attribution."""

    def _load_snapshot():
        with get_db_session() as session:
            row = session.get(DecisionSnapshot, decision_id)
            if row is None or row.status != "closed":
                return None
            return {
                "symbol": row.symbol,
                "pnl": row.realized_pnl,
                "ret_pct": row.return_pct,
                "action": row.final_action,
                "votes_json": row.votes_json,
                "causal_reasoning_json": row.causal_reasoning_json,
            }

    # get_db_session()/session.get() is a synchronous SQLAlchemy call; run it
    # off the event loop so this fire-and-forget attribution task can't stall
    # other symbols' concurrent processing while it waits on the DB.
    loaded = await asyncio.to_thread(_load_snapshot)
    if loaded is None:
        logger.warning(f"Attribution failed: Trade {decision_id} not found or not closed.")
        return None

    symbol = loaded["symbol"]
    pnl = loaded["pnl"]
    ret_pct = loaded["ret_pct"]
    action = loaded["action"]

    try:
        votes = json.loads(loaded["votes_json"])
    except Exception:
        votes = {}

    try:
        causal = json.loads(loaded["causal_reasoning_json"])
    except Exception:
        causal = {}

    # Build prompt for LLM
    prompt = f"""
    You are the Chief Quantitative Researcher at an algorithmic trading firm.
    Analyze the following completed algorithmic trade and explain its performance.
    
    Trade Outcome:
    - Symbol: {symbol}
    - Action Taken: {action}
    - Realized PnL: ${pnl:.2f}
    - Return %: {ret_pct:.2f}%
    
    Committee Votes at Entry:
    {json.dumps(votes, indent=2)}
    
    Feature Gradients (Causal Reasoning from Transformer Neural Network):
    (Positive values pushed the model toward BUY, negative toward SELL. Magnitude indicates importance.)
    {json.dumps(causal, indent=2)}
    
    Task: Extract a structured JSON response explaining the trade outcome. 
    Do not include markdown formatting or backticks, just the JSON.
    Format:
    {{
        "success_factors": "A 1-2 sentence explanation of why the trade made or lost money based on the data.",
        "key_signals": ["List of top 2-3 most important features based on the gradients"],
        "mvp_member": "Which committee member (brain) had the most accurate vote?",
        "robustness_score": float between 0.0 and 1.0 (1.0 = highly robust consensus, 0.0 = fragile/single-signal),
        "lessons_learned": "A 1-sentence takeaway to improve future models."
    }}
    """
    
    groq_key = settings.GROQ_API_KEY
    gemini_key = settings.GEMINI_API_KEY
    
    res = None
    if groq_key and settings.GROQ_ATTRIBUTION_MODEL:
        res = await _call_groq_attribution(prompt, groq_key)
    elif gemini_key:
        res = await _call_gemini_attribution(prompt, gemini_key)
    else:
        logger.warning("No LLM keys found for attribution engine.")
        return None
        
    return res

async def _call_groq_attribution(prompt: str, api_key: str) -> Optional[Dict[str, Any]]:
    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=api_key)
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a quantitative finance attribution system. Output only raw JSON."},
                {"role": "user", "content": prompt}
            ],
            model=settings.GROQ_ATTRIBUTION_MODEL,  # llama-3.1-8b-instant was deprecated/shut down 08/16/26
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        content = chat_completion.choices[0].message.content
        result = json.loads(content)
        return result
    except Exception as e:
        logger.error(f"Groq LLM attribution failed: {e}")
        return None

async def _call_gemini_attribution(prompt: str, api_key: str) -> Optional[Dict[str, Any]]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
        # generate_content() is synchronous (blocking HTTP call). Run it in a thread
        # so we don't freeze the entire asyncio event loop during LLM inference.
        response = await asyncio.to_thread(model.generate_content, prompt)
        
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Gemini LLM attribution failed: {e}")
        return None
