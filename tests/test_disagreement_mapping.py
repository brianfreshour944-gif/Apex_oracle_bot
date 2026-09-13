"""Tests for committee disagreement mapping (entropy -> LOW/MEDIUM/HIGH).

Regression: brain_disagreement used to stay "LOW" forever (hardcoded
placeholder in strategies.py that the committee never overrode), making the
HIGH-disagreement adversarial veto in bot.py unreachable.
"""
import math

import pytest

from src.committee.models import BrainVote, calculate_vote_entropy, disagreement_from_entropy


def _vote(name: str, action: str) -> BrainVote:
    return BrainVote(name=name, action=action, confidence=0.9, weight=0.2,
                     regime="neutral", reason="test")


def test_entropy_consensus_is_low():
    votes = [_vote(f"b{i}", "buy") for i in range(5)]
    assert calculate_vote_entropy(votes) == pytest.approx(0.0, abs=1e-9)
    assert disagreement_from_entropy(calculate_vote_entropy(votes)) == "LOW"


def test_minority_dissent_is_medium():
    votes = [_vote(f"b{i}", "buy") for i in range(4)] + [_vote("b4", "sell")]
    ent = calculate_vote_entropy(votes)  # 4-1 split -> 0.722
    assert 0.3 < ent <= 0.8
    assert disagreement_from_entropy(ent) == "MEDIUM"


def test_even_split_is_high():
    votes = [_vote("a", "buy"), _vote("b", "buy"), _vote("c", "sell"),
             _vote("d", "sell"), _vote("e", "hold")]
    ent = calculate_vote_entropy(votes)  # 2-2-1 split -> high
    assert ent > 0.8
    assert disagreement_from_entropy(ent) == "HIGH"


def test_stand_aside_votes_excluded():
    votes = [_vote("a", "buy"), _vote("b", "stand_aside")]
    assert calculate_vote_entropy(votes) == pytest.approx(0.0, abs=1e-9)


def test_nan_entropy_degrades_to_low():
    assert disagreement_from_entropy(float("nan")) == "LOW"


def test_max_entropy_bound():
    # Theoretical max with 3 actions is log2(3) -- sanity-check thresholds fit
    assert math.log2(3) > 0.8
