"""Unit tests for the AdaptiveMetaLearner (self-evolving brain weighting)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# config.py requires Alpaca creds at import time; supply dummies for tests.
os.environ.setdefault("ALPACA_API_KEY", "test")
os.environ.setdefault("ALPACA_SECRET_KEY", "test")

import json

import pytest

from src.committee.adaptive_meta import AdaptiveMetaLearner, BRAINS
from src.committee.models import BrainVote
from src.committee import outcome_tracker


def _vote(name, action, conf=0.8):
    return BrainVote(name=name, action=action, confidence=conf, weight=0.2, regime="uptrend", reason="t")


def _snapshot(regime, final_action, votes):
    return {"regime": regime, "final_action": final_action, "brain_votes": votes}


def _sum(weights):
    return sum(weights.values())


def test_cold_start_equal_weights_normalize(tmp_path):
    learner = AdaptiveMetaLearner(state_path=str(tmp_path / "state.json"))
    decision = learner.combine([_vote(b, "buy") for b in BRAINS], "uptrend")
    weights = decision.weights
    assert set(weights) == set(BRAINS)
    assert _sum(weights) == pytest.approx(1.0, abs=1e-6)
    # Equal weights on cold start
    for w in weights.values():
        assert w == pytest.approx(1.0 / len(BRAINS), abs=1e-6)


def test_aligned_profitable_brain_gains_weight(tmp_path):
    learner = AdaptiveMetaLearner(state_path=str(tmp_path / "s.json"), learning_rate=0.3)
    before = dict(learner._clamp_normalize(learner._regime_weights("uptrend")))
    # transformer voted buy, final action buy, trade profitable -> transformer correct
    snap = _snapshot("uptrend", "buy", {"transformer": "buy", "quant": "hold", "momentum": "sell"})
    learner.update(snap, {"net_pnl": 120.0})
    after = learner.weights["uptrend"]
    assert after["transformer"] > before["transformer"]
    # A brain that voted the opposite (momentum sell) should lose weight
    assert after["momentum"] < before["momentum"]
    # hold voter (quant) unchanged direction-wise; still normalized
    assert _sum(after) == pytest.approx(1.0, abs=1e-6)


def test_opposite_losing_brain_loses_weight(tmp_path):
    learner = AdaptiveMetaLearner(state_path=str(tmp_path / "s.json"), learning_rate=0.3)
    before = dict(learner._clamp_normalize(learner._regime_weights("downtrend")))
    # final action buy but trade LOST -> buy voters were wrong, sell voters right
    snap = _snapshot("downtrend", "buy", {"transformer": "buy", "sentinel": "sell"})
    learner.update(snap, {"net_pnl": -80.0})
    after = learner.weights["downtrend"]
    assert after["transformer"] < before["transformer"]
    assert after["sentinel"] > before["sentinel"]


def test_weights_stay_clamped_and_normalized(tmp_path):
    learner = AdaptiveMetaLearner(
        state_path=str(tmp_path / "s.json"), learning_rate=0.9, min_weight=0.05, max_weight=0.50
    )
    # Hammer transformer as correct many times; it must not exceed max_weight.
    for _ in range(100):
        learner.update(
            _snapshot("uptrend", "buy", {b: "buy" for b in BRAINS}),
            {"net_pnl": 50.0},
        )
    w = learner.weights["uptrend"]
    assert _sum(w) == pytest.approx(1.0, abs=1e-6)
    for val in w.values():
        assert val <= 0.50 + 1e-6
        assert val >= 0.05 - 1e-6


def test_per_regime_updates_do_not_bleed(tmp_path):
    learner = AdaptiveMetaLearner(state_path=str(tmp_path / "s.json"), learning_rate=0.3)
    learner.update(_snapshot("uptrend", "buy", {"transformer": "buy"}), {"net_pnl": 100.0})
    # 'ranging' regime was never touched -> stays at equal weights.
    ranging = learner._clamp_normalize(learner._regime_weights("ranging"))
    for val in ranging.values():
        assert val == pytest.approx(1.0 / len(BRAINS), abs=1e-6)
    # uptrend transformer moved
    assert learner.weights["uptrend"]["transformer"] != pytest.approx(1.0 / len(BRAINS), abs=1e-6)


def test_flat_pnl_no_weight_change(tmp_path):
    learner = AdaptiveMetaLearner(state_path=str(tmp_path / "s.json"), learning_rate=0.3)
    before = dict(learner._clamp_normalize(learner._regime_weights("uptrend")))
    learner.update(_snapshot("uptrend", "buy", {"transformer": "buy"}), {"net_pnl": 0.0})
    after = learner._clamp_normalize(learner._regime_weights("uptrend"))
    assert after == pytest.approx(before)
    assert learner.sample_count == 1  # sample counted, weights untouched


def test_hold_votes_are_conservative(tmp_path):
    learner = AdaptiveMetaLearner(state_path=str(tmp_path / "s.json"), learning_rate=0.5)
    before = dict(learner._clamp_normalize(learner._regime_weights("quiet")))
    # Only hold/stand_aside votes -> no directional signal to grade for them.
    learner.update(
        _snapshot("quiet", "buy", {"quant": "hold", "sentinel": "stand_aside"}),
        {"net_pnl": 40.0},
    )
    after = learner.weights["quiet"]
    assert after["quant"] == pytest.approx(before["quant"], abs=1e-6)
    assert after["sentinel"] == pytest.approx(before["sentinel"], abs=1e-6)


def test_atomic_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "state.json")
    learner = AdaptiveMetaLearner(state_path=path, learning_rate=0.3)
    learner.update(_snapshot("uptrend", "buy", {"transformer": "buy", "momentum": "sell"}), {"net_pnl": 90.0})
    assert os.path.exists(path)

    reloaded = AdaptiveMetaLearner(state_path=path)
    assert reloaded.sample_count == learner.sample_count
    assert reloaded.weights["uptrend"]["transformer"] == pytest.approx(
        learner.weights["uptrend"]["transformer"], abs=1e-9
    )
    # State file is versioned + timestamped.
    with open(path) as fh:
        data = json.load(fh)
    assert data["version"] >= 1
    assert "timestamp" in data


def test_corrupt_state_falls_back_to_equal_weights(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{ this is not valid json ]")
    learner = AdaptiveMetaLearner(state_path=str(path))
    assert learner.sample_count == 0
    w = learner._clamp_normalize(learner._regime_weights("uptrend"))
    for val in w.values():
        assert val == pytest.approx(1.0 / len(BRAINS), abs=1e-6)


def test_missing_state_falls_back_to_equal_weights(tmp_path):
    learner = AdaptiveMetaLearner(state_path=str(tmp_path / "does_not_exist.json"))
    assert learner.sample_count == 0
    decision = learner.combine([_vote(b, "buy") for b in BRAINS], "uptrend")
    assert decision.action == "buy"
    assert _sum(decision.weights) == pytest.approx(1.0, abs=1e-6)


def test_combine_picks_highest_weighted_direction(tmp_path):
    learner = AdaptiveMetaLearner(state_path=str(tmp_path / "s.json"), learning_rate=0.4)
    # Train transformer to dominate in uptrend by repeated correct buys.
    for _ in range(20):
        learner.update(_snapshot("uptrend", "buy", {"transformer": "buy", "quant": "sell"}), {"net_pnl": 100.0})
    votes = [_vote("transformer", "buy", 0.9), _vote("quant", "sell", 0.9)]
    decision = learner.combine(votes, "uptrend")
    assert decision.action == "buy"
    assert decision.explanation.startswith("adaptive[uptrend]")


def test_outcome_tracker_from_snapshot_and_apply(tmp_path):
    learner = AdaptiveMetaLearner(state_path=str(tmp_path / "s.json"), learning_rate=0.3)
    snap = {
        "symbol": "BTC/USD",
        "regime": "uptrend",
        "final_action": "buy",
        "confidence": 0.7,
        "brain_votes": {"transformer": "buy", "momentum": "sell"},
    }
    ex = outcome_tracker.from_decision_snapshot(snap, realized_pnl=150.0, return_pct=3.0)
    assert ex.label == 1
    before = dict(learner._clamp_normalize(learner._regime_weights("uptrend")))
    applied = outcome_tracker.apply_to_learner(learner, [ex])
    assert applied == 1
    assert learner.weights["uptrend"]["transformer"] > before["transformer"]
