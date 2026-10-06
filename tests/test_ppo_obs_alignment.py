"""Regression: the PPO training env and the live meta-learner must build the
SAME observation vector.

`rl_env.MetaDecisionEnv` (training) and `rl_meta.RLMetaLearner._build_obs`
(live inference) were built independently and drifted to 17 vs 26 dims. The
weekly promotion gate in scripts/evolutionary_ppo_trainer.py compares
`challenger.observation_space.shape` against the live shape, so a mismatch
permanently blocked every PPO promotion: the meta-learner stayed frozen at
the baked model forever. Both now delegate to
`regime_utils.build_rl_observation`. Found and fixed 2026-10-06.
"""

import numpy as np
import pytest

from src.committee import rl_meta
from src.committee.models import BrainVote
from src.committee.regime_utils import RL_OBS_DIM, build_rl_observation

gym = pytest.importorskip("gymnasium", reason="rl_env imports gymnasium")
from src.committee.rl_env import MetaDecisionEnv  # noqa: E402

SNAPSHOT = {
    "regime": "trending",
    "features": {"rsi": 55.0, "atr": 2.0, "macd": 0.4},
    "votes": {"transformer": "buy", "momentum": "sell", "quant": "hold"},
    "final_action": "buy",
    "realized_pnl": 1.5,
}


def _live_obs(votes, regime, features):
    return rl_meta.RLMetaLearner()._build_obs(votes, regime, features)


def test_observation_dim_constant_is_26():
    assert RL_OBS_DIM == 26


def test_env_observation_space_matches_live_learner():
    env = MetaDecisionEnv([SNAPSHOT])
    live = _live_obs(
        [BrainVote(name="transformer", action="buy", confidence=0.6, weight=1.0, regime="trending", reason="")],
        "trending",
        {"rsi": 55.0, "atr": 2.0, "macd": 0.4},
    )
    assert env.observation_space.shape == live.shape == (RL_OBS_DIM,)


def test_env_get_obs_is_identical_to_shared_builder():
    env = MetaDecisionEnv([SNAPSHOT])
    obs = env.reset()[0]
    expected = build_rl_observation("trending", {"rsi": 55.0, "atr": 2.0, "macd": 0.4}, SNAPSHOT["votes"])
    assert obs.shape == (RL_OBS_DIM,)
    np.testing.assert_allclose(obs, expected, rtol=0, atol=0)


def test_env_terminal_obs_shape_matches_space_not_14():
    """step() past the end used to return np.zeros(14), inconsistent with the
    17-dim space (and now the 26-dim one)."""
    env = MetaDecisionEnv([SNAPSHOT])
    env.reset()
    _obs, _r, done, _trunc, _info = env.step(np.zeros(7, dtype=np.float32))
    assert done is True
    terminal_obs, _r2, done2, _t2, _i2 = env.step(np.zeros(7, dtype=np.float32))
    assert done2 is True
    assert terminal_obs.shape == env.observation_space.shape == (RL_OBS_DIM,)


def test_confidence_threshold_and_size_mappings_are_shared():
    """The env's conf_thresh mapping (0.5-0.8) disagreed with the live
    learner's (0.35-0.55), so a policy learned in training meant something
    different live. Both now bind the same regime_utils helpers."""
    from src.committee import rl_env, rl_meta
    from src.committee.regime_utils import confidence_threshold, position_size_multiplier

    assert rl_env.confidence_threshold is confidence_threshold
    assert rl_meta.confidence_threshold is confidence_threshold
    assert rl_env.position_size_multiplier is position_size_multiplier
    assert rl_meta.position_size_multiplier is position_size_multiplier

    for raw in (-1.0, 0.0, 1.0):
        assert 0.35 <= confidence_threshold(raw) <= 0.55
        assert 0.5 <= position_size_multiplier(raw) <= 1.5


def test_trained_ppo_challenger_accepts_live_observation():
    """End-to-end: a PPO model trained on MetaDecisionEnv must predict on the
    observation the live learner builds -- the exact property the promotion
    gate checks."""
    sb3 = pytest.importorskip("stable_baselines3", reason="PPO training needs stable-baselines3")

    snaps = [dict(SNAPSHOT, realized_pnl=1.5 if i % 2 else -1.0) for i in range(8)]
    env = MetaDecisionEnv(snaps)
    model = sb3.PPO("MlpPolicy", env, verbose=0, learning_rate=0.001)
    model.learn(total_timesteps=len(snaps) * 50)

    live_obs = _live_obs([], "trending", {"rsi": 55.0, "atr": 2.0, "macd": 0.4})
    assert model.observation_space.shape == live_obs.shape == (RL_OBS_DIM,)

    action, _ = model.predict(live_obs, deterministic=True)
    assert action.shape == (7,)
