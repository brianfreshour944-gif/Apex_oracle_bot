import json
import aiohttp

class AIClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def _call_llm(self, prompt: str) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            ) as resp:
                data = await resp.json()
                text = data["choices"][0]["message"]["content"]

        try:
            return json.loads(text)
        except Exception:
            return {"error": "parse_error", "raw": text}

    async def evaluate_trade(self, context: dict) -> dict:
        prompt = f"""
        You are an expert quantitative trading assistant for a live crypto bot.

        Evaluate this context and respond ONLY with a JSON object.

        Context:
        {json.dumps(context, indent=2)}

        Requirements:
        - "action": "BUY", "SELL", or "HOLD".
        - "confidence": 0–1.
        - "reason": short explanation.
        - "position_size_factor": 0–1.
        - "risk_score": 0–1.
        - "regime_override": null or a short string.

        Output MUST be valid JSON only.
        """

        result = await self._call_llm(prompt)

        if "error" in result:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "AI parse error",
                "position_size_factor": 0.0,
                "risk_score": 1.0,
                "regime_override": None,
            }

        return {
            "action": result.get("action", "HOLD"),
            "confidence": float(result.get("confidence", 0.0)),
            "reason": result.get("reason", "No reason"),
            "position_size_factor": float(result.get("position_size_factor", 0.0)),
            "risk_score": float(result.get("risk_score", 1.0)),
            "regime_override": result.get("regime_override"),
        }
