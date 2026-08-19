"""Alternative Data & Sentiment Analyzer.

Fetches recent news from Alpaca News API and uses an LLM (Groq/Gemini/OpenAI) 
to extract a structured sentiment signal. Fallbacks to a heuristic stub if no keys are found.
"""

import os
import json
import httpx
import asyncio
from typing import Dict, Any

from src.config import settings
from src.logging_config import get_logger

logger = get_logger("sentiment")

EVENT_TYPES = ["earnings", "regulation", "macro", "security", "adoption", "none"]

async def fetch_news_headlines(symbol: str, limit: int = 5) -> str:
    """Fetch recent news headlines from Alpaca News API."""
    try:
        url = "https://data.alpaca.markets/v1beta1/news"
        params = {
            "symbols": symbol,
            "limit": limit
        }
        headers = {
            "APCA-API-KEY-ID": settings.ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY
        }
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                news_items = data.get("news", [])
                
                headlines = []
                for item in news_items:
                    headlines.append(f"- {item.get('headline', '')}: {item.get('summary', '')}")
                    
                if not headlines:
                    return "No recent news."
                return "\n".join(headlines)
            else:
                logger.warning(f"Alpaca news returned {response.status_code}")
                return "Failed to fetch news."
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return "Failed to fetch news."

def _heuristic_fallback(headlines: str) -> Dict[str, Any]:
    """Basic keyword matching fallback if no LLM API key is present."""
    headlines_lower = headlines.lower()
    
    # Event Types
    event_type = "none"
    if "hack" in headlines_lower or "exploit" in headlines_lower or "stolen" in headlines_lower:
        event_type = "security"
    elif "sec" in headlines_lower or "ban" in headlines_lower or "lawsuit" in headlines_lower or "regulation" in headlines_lower:
        event_type = "regulation"
    elif "fed" in headlines_lower or "inflation" in headlines_lower or "rate" in headlines_lower:
        event_type = "macro"
    elif "adopt" in headlines_lower or "launch" in headlines_lower or "integrate" in headlines_lower:
        event_type = "adoption"
        
    # Sentiment
    positive_words = ["surge", "jump", "integrate", "adopt", "launch", "partnership", "growth", "bull"]
    negative_words = ["plunge", "crash", "hack", "stolen", "lawsuit", "ban", "sec", "bear"]
    
    pos_count = sum(1 for w in positive_words if w in headlines_lower)
    neg_count = sum(1 for w in negative_words if w in headlines_lower)
    
    if event_type == "security":
        score = -0.9
        conf = 0.9
    elif pos_count > neg_count:
        score = min(1.0, (pos_count - neg_count) * 0.25)
        conf = min(0.8, pos_count * 0.2)
    elif neg_count > pos_count:
        score = max(-1.0, (pos_count - neg_count) * 0.25)
        conf = min(0.8, neg_count * 0.2)
    else:
        score = 0.0
        conf = 0.5
        
    return {
        "sentiment_score": score,
        "event_type": event_type,
        "confidence": conf,
        "duration_hrs": 24.0
    }

async def call_groq_llm(headlines: str, api_key: str) -> Dict[str, Any]:
    """Call Groq API (Llama 3 8B) for fast structured output."""
    prompt = f"""
    Analyze the following recent crypto news headlines:
    {headlines}
    
    Extract a structured JSON response evaluating the overall sentiment. Do not include markdown formatting or backticks, just the JSON.
    Format:
    {{
        "sentiment_score": float between -1.0 (very negative) and 1.0 (very positive),
        "event_type": "string matching exactly one of: earnings, regulation, macro, security, adoption, none",
        "confidence": float between 0.0 and 1.0 representing your confidence in this assessment,
        "duration_hrs": float representing estimated hours this news will impact price
    }}
    """
    
    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=api_key)
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a quantitative finance sentiment extraction system. Output only raw JSON."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        content = chat_completion.choices[0].message.content
        result = json.loads(content)
        return result
    except Exception as e:
        logger.error(f"Groq LLM extraction failed: {e}")
        return _heuristic_fallback(headlines)
        
async def call_gemini_llm(headlines: str, api_key: str) -> Dict[str, Any]:
    """Call Gemini API for structured output."""
    prompt = f"""
    Analyze the following recent crypto news headlines:
    {headlines}
    
    Extract a structured JSON response evaluating the overall sentiment. Do not include markdown formatting or backticks, just the JSON.
    Format:
    {{
        "sentiment_score": float between -1.0 (very negative) and 1.0 (very positive),
        "event_type": "string matching exactly one of: earnings, regulation, macro, security, adoption, none",
        "confidence": float between 0.0 and 1.0 representing your confidence in this assessment,
        "duration_hrs": float representing estimated hours this news will impact price
    }}
    """
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
        response = model.generate_content(prompt)
        
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Gemini LLM extraction failed: {e}")
        return _heuristic_fallback(headlines)

async def extract_sentiment(symbol: str) -> Dict[str, Any]:
    """Main entrypoint: Fetch news, parse sentiment using best available LLM."""
    headlines = await fetch_news_headlines(symbol)
    
    if headlines in ["No recent news.", "Failed to fetch news."]:
        return {
            "sentiment_score": 0.0,
            "event_type": "none",
            "confidence": 0.0,
            "duration_hrs": 0.0
        }
        
    groq_key = settings.GROQ_API_KEY
    gemini_key = settings.GEMINI_API_KEY
    
    if groq_key:
        logger.info(f"Using Groq for Sentiment Analysis on {symbol}")
        res = await call_groq_llm(headlines, groq_key)
    elif gemini_key:
        logger.info(f"Using Gemini for Sentiment Analysis on {symbol}")
        res = await call_gemini_llm(headlines, gemini_key)
    else:
        logger.info(f"No LLM keys found. Using heuristic fallback for Sentiment Analysis on {symbol}")
        res = _heuristic_fallback(headlines)
        
    # Validation
    try:
        score = float(res.get("sentiment_score", 0.0))
        event = str(res.get("event_type", "none")).lower()
        conf = float(res.get("confidence", 0.5))
        dur = float(res.get("duration_hrs", 24.0))
        
        if event not in EVENT_TYPES:
            event = "none"
            
        return {
            "sentiment_score": max(-1.0, min(1.0, score)),
            "event_type": event,
            "confidence": max(0.0, min(1.0, conf)),
            "duration_hrs": dur
        }
    except Exception as e:
        logger.warning(f"Error validating sentiment output: {e}")
        return {
            "sentiment_score": 0.0,
            "event_type": "none",
            "confidence": 0.0,
            "duration_hrs": 24.0
        }
