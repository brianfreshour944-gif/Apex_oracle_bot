"""Regression tests for regime normalization (M1).

Locks the contract that every incoming regime resolves into the RL-6 one-hot
space so rl_meta/rl_env never silently zero the regime vector for production
(DT-8) labels.
"""

from src.committee.regime_utils import (
    CANONICAL_REGIMES,
    RL_REGIMES,
    is_rl_regime,
    normalize_regime,
)


def test_production_dt8_labels_resolve_into_rl6():
    for label in CANONICAL_REGIMES:
        resolved = normalize_regime(label)
        assert resolved in RL_REGIMES, f"{label} -> {resolved} not in RL_REGIMES"


def test_rl6_labels_passthrough_unchanged():
    for label in RL_REGIMES:
        assert normalize_regime(label) == label


def test_aliases_are_injective_enough_and_valid():
    for label in CANONICAL_REGIMES:
        assert normalize_regime(label) in RL_REGIMES


def test_unknown_and_empty_labels_fall_back_to_default():
    assert normalize_regime("") == "default"
    assert normalize_regime(None) == "default"
    assert normalize_regime("uptrend") == "default"
    assert normalize_regime("does_not_exist") == "default"


def test_rl6_one_hot_space_set():
    assert set(RL_REGIMES) == {
        "trending", "mean_reverting", "volatile", "choppy", "breakout", "default",
    }


def test_is_rl_regime_helper():
    assert is_rl_regime("trending")
    assert is_rl_regime("default")
    assert not is_rl_regime("bull")
    assert not is_rl_regime("uptrend")
