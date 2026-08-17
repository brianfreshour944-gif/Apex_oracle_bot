import asyncio
import os
os.environ.setdefault("ALPACA_API_KEY", "test")
os.environ.setdefault("ALPACA_SECRET_KEY", "test")

from unittest.mock import patch, AsyncMock
from src.committee.models import BrainVote
from src.committee import committee

async def main():
    committee.settings.ADAPTIVE_ML_ENABLED = False

    votes = {
        "transformer": BrainVote("transformer", "sell", 0.40, 0.0, "sideways", "x"),
        "quant":       BrainVote("quant", "hold", 0.5, 0.0, "sideways", "x"),
        "momentum":    BrainVote("momentum", "stand_aside", 0.50, 0.0, "sideways", "x"),
        "sentinel":    BrainVote("sentinel", "hold", 0.5, 0.0, "sideways", "x"),
        "llm":         BrainVote("llm", "hold", 0.40, 0.0, "sideways", "x"),
    }

    with patch("src.committee.committee.transformer_brain", new=AsyncMock(return_value=votes["transformer"])), \
         patch("src.committee.committee.quant_brain", new=AsyncMock(return_value=votes["quant"])), \
         patch("src.committee.committee.momentum_brain", new=AsyncMock(return_value=votes["momentum"])), \
         patch("src.committee.committee.sentinel_brain", new=AsyncMock(return_value=votes["sentinel"])), \
         patch("src.committee.committee.llm_brain", new=AsyncMock(return_value=votes["llm"])):

        signal = {"regime": "sideways", "rsi": 50.0, "atr": 0.0, "features": {}}
        result = await committee.run_committee("ETH/USD", 100.0, signal)
        print(f"ETH/USD: action={result.action.upper()}  score={result.score:.3f}")
        expected_ok = result.action == "stand_aside" and result.score < 0.20
        print("PASS - hold no longer wins" if expected_ok else "FAIL - hold is still winning")

asyncio.run(main())
