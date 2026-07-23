"""Unit tests for 5-Brain Ensemble Committee decision system with confidence sizing."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
from src.committee.models import BrainVote, CommitteeResult

from src.committee.committee import run_committee, WINNING_SCORE_THRESHOLD, calculate_confidence_size_multiplier
from src.committee.sentinel_brain import sentinel_brain
from src.committee.momentum_brain import momentum_brain
from src.committee.quant_brain import quant_brain
from src.committee.transformer_brain import transformer_brain
from src.committee.llm_brain import llm_brain

async def test_committee_buy_consensus_and_sizing():
    """Test committee agreement on buy action and dynamic confidence position multiplier."""
    signal = {
        "action": "buy",
        "confidence": 0.92,
        "regime": "uptrend",
        "rsi": 22.0,  # Strongly Oversold (Quant BUY)
        "atr": 50.0
    }
    result = await run_committee("BTC/USD", 50000.0, signal)
    
    print(f"DEBUG: score={result.score}, multiplier={result.size_multiplier}")
    assert result.vetoed is False
    assert result.action == "buy"
    assert result.score >= WINNING_SCORE_THRESHOLD
    assert 0.80 <= result.size_multiplier <= 1.50  # Dynamic sizing multiplier check
    assert len(result.votes) == 5


    print(f"PASS: test_committee_buy_consensus_and_sizing (score={result.score:.2f}, multiplier={result.size_multiplier:.2f}x)")


async def test_committee_sentinel_hard_veto():
    """Test Sentinel brain enforcing hard veto under extreme crash conditions."""
    signal = {
        "action": "buy",
        "confidence": 0.85,
        "regime": "crash",
        "rsi": 15.0,
        "atr": 100.0
    }
    result = await run_committee("BTC/USD", 50000.0, signal)
    
    assert result.vetoed is True
    assert result.action == "stand_aside"
    assert result.size_multiplier == 0.0
    assert "Extreme market volatility" in result.veto_reason
    print("PASS: test_committee_sentinel_hard_veto")

async def test_committee_low_conviction_stand_aside():
    """Test low conviction scoring results in stand_aside."""
    signal = {
        "action": "hold",
        "confidence": 0.45,
        "regime": "neutral",
        "rsi": 50.0,
        "atr": 10.0
    }
    result = await run_committee("ETH/USD", 30000.0, signal)
    
    assert result.action in ["stand_aside", "hold"]
    print("PASS: test_committee_low_conviction_stand_aside")

def test_size_multiplier_logic():
    """Test size multiplier scaling based on confidence scores."""
    # Marginal confidence score (0.60) -> ~0.50x
    m1 = calculate_confidence_size_multiplier(0.60, 0.0)
    assert m1 == 0.50
    
    # Moderate confidence score (0.76) -> ~1.00x
    m2 = calculate_confidence_size_multiplier(0.76, 0.0)
    assert 0.95 <= m2 <= 1.05
    
    # High conviction score (0.92) -> > 1.40x
    m3 = calculate_confidence_size_multiplier(0.92, 0.0)
    assert m3 >= 1.40
    print(f"PASS: test_size_multiplier_logic (0.60->{m1}x, 0.76->{m2}x, 0.92->{m3}x)")

async def main():
    await test_committee_buy_consensus_and_sizing()
    await test_committee_sentinel_hard_veto()
    await test_committee_low_conviction_stand_aside()
    test_size_multiplier_logic()
    print("SUCCESS: All 5-Brain Committee tests passed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
