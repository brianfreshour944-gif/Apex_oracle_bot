"""Tests for committee disagreement mapping (entropy -> LOW/MEDIUM/HIGH).

Regression: brain_disagreement used to stay "LOW" forever (hardcoded
placeholder in strategies.py that the committee never overrode), making the
HIGH-disagreement adversarial veto in bot.py unreachable.

Second regression (this file's later tests): making that veto reachable with
the *all-opinions* entropy made it self-locking -- HOLD votes counted as
disagreement while being excluded from the score the veto compares against, so
the ordinary "one directional brain, three HOLDs" pattern read as HIGH
disagreement with a diluted score and every symbol was rejected every cycle
(live log: BTC 0.811/0.230, SOL 0.971/0.391, ETH 1.371/0.195, all
VETO_ADVERSARIAL). The veto now consumes calculate_directional_entropy().
"""
import math

import pytest

from src.committee.models import (
    BrainVote,
    calculate_directional_entropy,
    calculate_vote_entropy,
    disagreement_from_entropy,
)


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


# ── Directional disagreement (what the adversarial veto consumes) ─────────────
# Every case below is a vote set from the live production log.

def _weighted(regime_weights, spec):
    return [BrainVote(name=name, action=action, confidence=conf, weight=regime_weights[name],
                      regime="low_volatility", reason="live-log")
            for name, action, conf in spec]


LOW_VOL_WEIGHTS = {"transformer": 0.30, "quant": 0.30, "momentum": 0.10,
                   "sentinel": 0.05, "llm": 0.25}


def test_abstentions_are_not_directional_disagreement():
    """1 directional vote + 3 HOLDs + 1 PASS is not 'brains disagree'.

    The old metric called this HIGH (0.811) while the committee score was
    diluted to 0.230 -- below the 0.55 veto floor -- which is what silenced the
    bot. It must read LOW now.
    """
    votes = _weighted(LOW_VOL_WEIGHTS, [
        ("transformer", "buy", 0.69), ("quant", "hold", 0.0), ("momentum", "stand_aside", 0.0),
        ("sentinel", "hold", 0.0), ("llm", "hold", 0.0),
    ])
    all_opinions = calculate_vote_entropy(votes)
    assert all_opinions == pytest.approx(0.811, abs=5e-3)
    assert disagreement_from_entropy(all_opinions) == "HIGH"  # the old behaviour

    assert calculate_directional_entropy(votes) == pytest.approx(0.0, abs=1e-9)
    assert disagreement_from_entropy(calculate_directional_entropy(votes)) == "LOW"


def test_two_directional_votes_agreeing_is_not_disagreement():
    votes = _weighted(LOW_VOL_WEIGHTS, [
        ("transformer", "buy", 0.66), ("quant", "hold", 0.0), ("momentum", "hold", 0.0),
        ("sentinel", "hold", 0.0), ("llm", "buy", 0.77),
    ])
    assert calculate_directional_entropy(votes) == pytest.approx(0.0, abs=1e-9)
    assert disagreement_from_entropy(calculate_directional_entropy(votes)) == "LOW"


def test_opposing_directional_votes_stay_high():
    """A real buy-vs-sell fight must still veto (this is the gate's purpose)."""
    votes = _weighted(LOW_VOL_WEIGHTS, [
        ("transformer", "sell", 0.39), ("quant", "hold", 0.0), ("momentum", "hold", 0.0),
        ("sentinel", "hold", 0.0), ("llm", "buy", 0.78),
    ])
    directional = calculate_directional_entropy(votes)
    assert directional == pytest.approx(1.0, abs=1e-9)
    assert disagreement_from_entropy(directional) == "HIGH"


def test_directional_entropy_matches_thresholds_after_abstraction():
    """The directional metric reuses the same LOW/MEDIUM/HIGH thresholds."""
    four_to_one = [_vote(f"b{i}", "buy") for i in range(4)] + [_vote("b4", "sell")]
    assert disagreement_from_entropy(calculate_directional_entropy(four_to_one)) == "MEDIUM"

    two_two = [_vote("a", "buy"), _vote("b", "buy"), _vote("c", "sell"),
               _vote("d", "sell"), _vote("e", "hold")]
    assert disagreement_from_entropy(calculate_directional_entropy(two_two)) == "HIGH"


def test_directional_entropy_all_abstentions_is_zero():
    votes = [_vote("a", "hold"), _vote("b", "stand_aside"), _vote("c", "skip")]
    assert calculate_directional_entropy(votes) == pytest.approx(0.0, abs=1e-9)
    assert calculate_directional_entropy([]) == pytest.approx(0.0, abs=1e-9)


def test_directional_entropy_max_is_log2_of_two():
    """Sanity: with abstentions excluded the max is 1.0, not log2(3)."""
    votes = [_vote("a", "buy"), _vote("b", "sell")]
    assert calculate_directional_entropy(votes) == pytest.approx(math.log2(2), abs=1e-9)
