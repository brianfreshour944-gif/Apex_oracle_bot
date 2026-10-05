"""Strategy selector must start each regime from its strategy priors.

Regression (2026-10-04): the strategy learner is an AdaptiveMetaLearner, whose
_regime_weights() lazily seeds a new regime with equal weights for the
COMMITTEE BRAINS (transformer/quant/momentum/sentinel/llm). The weights were
therefore never empty, so select_best_strategy's regime priors (trend for
bull/bear, mean-reversion for sideways, breakout for high-vol...) were dead
code; every real strategy then got min_weight except "momentum" -- the one
name that is both a brain and a strategy -- which kept the brain's weight.
A backtest probe saw "momentum" selected on 4,320/4,320 hours in every regime.
"""
import pytest

import src.strategy_selector as sel


@pytest.fixture
def fresh_learner(monkeypatch):
    learner = sel.StrategyMetaLearner(state_path=None, learning_rate=0.10, min_weight=0.01, max_weight=0.80)
    monkeypatch.setattr(sel, "_STRATEGY_LEARNER", learner)
    return learner


@pytest.mark.parametrize("regime,expected", [
    ("bull", "trend_following"),
    ("bear", "trend_following"),
    ("trending", "trend_following"),
    ("sideways", "mean_reversion"),
    ("high_volatility", "breakout"),
])
def test_regime_priors_drive_selection(fresh_learner, regime, expected):
    assert sel.select_best_strategy(regime, {"close": 100.0, "atr": 2.0}) == expected


def test_momentum_not_selected_in_every_regime(fresh_learner):
    picks = {sel.select_best_strategy(r, {"close": 100.0, "atr": 2.0})
             for r in ("bull", "bear", "sideways", "high_volatility", "low_volatility", "neutral")}
    assert picks != {"momentum"}


def test_learner_weights_never_contain_brain_names(fresh_learner):
    sel.select_best_strategy("bull", {"close": 100.0, "atr": 2.0})
    sel.record_strategy_outcome("bull", "trend_following", "buy", pnl=5.0, return_pct=2.0)
    assert set(fresh_learner.weights["bull"]) <= set(sel.STRATEGIES)


def test_brain_polluted_regime_state_resets_to_priors(fresh_learner):
    # what a pre-fix strategy_meta_state.json holds: brain-seeded weights
    fresh_learner.weights["sideways"] = {"transformer": 0.2, "quant": 0.2, "momentum": 0.2,
                                         "sentinel": 0.2, "llm": 0.2, "mean_reversion": 0.01}
    assert sel.select_best_strategy("sideways", {"close": 100.0, "atr": 2.0}) == "mean_reversion"
    assert set(fresh_learner.weights["sideways"]) <= set(sel.STRATEGIES)


def test_learned_strategy_weights_are_kept(fresh_learner):
    fresh_learner.weights["sideways"] = dict.fromkeys(sel.STRATEGIES, 0.01)
    fresh_learner.weights["sideways"]["breakout"] = 0.9
    assert sel.select_best_strategy("sideways", {"close": 100.0, "atr": 2.0}) == "breakout"
