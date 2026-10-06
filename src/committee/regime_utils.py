"""Regime vocabulary normalization (M1).

Canonicalizes the disparate regime label sets used across the committee of
brains into a single RL-6 one-hot space, WITHOUT changing the observation
dimensionality the trained PPO agent expects.

Background:
  * DT-8 (live classifier in strategies.py + DecisionTransformer brain):
    bull, bear, trending, sideway, high_volatility, low_volatility,
    neutral, default.
  * RL-6 (rl_meta.py / rl_env.py): trending, mean_reverting, volatile,
    choppy, breakout, default.

The live classifier emits DT-8 labels, but the RL one-hot guard is
`if regime in REGIMES` (RL-6). Without normalization, DT-8 regimes the trainer
never saw (bull, high_volatility, ...) silently zero the one-hot vector. We
instead map DT-8 -> RL-6 so every incoming regime lands in a real bucket.

We deliberately do NOT expand REGIMES to 8: the regime one-hot length (6) is
baked into the trained PPO agent's input shape; changing it would desync the
live agent.

Observation layout (26-dim, matches the live agent and the baked
`models/ppo_meta_weights.zip`):
    [regime one-hot (6)] + [numeric features (9)] + [event one-hot (6)] + [votes (5)]

`build_rl_observation()` is the single source of truth for this layout, shared
by the live learner (`rl_meta.RLMetaLearner._build_obs`) and the training env
(`rl_env.MetaDecisionEnv._get_obs`). They used to be built independently and
drifted to 26 vs 17 dims, which permanently blocked every weekly PPO promotion
(2026-10-06).
"""

from __future__ import annotations

import numpy as np

# RL-native one-hot regime space (MUST stay length 6 -- matches trained PPO).
RL_REGIMES: list = ["trending", "mean_reverting", "volatile", "choppy", "breakout", "default"]

# Committee brains whose votes occupy the tail of the observation vector.
RL_BRAINS: list = ["transformer", "quant", "momentum", "sentinel", "llm"]

# News/sentiment event one-hot vocabulary.
RL_EVENT_TYPES: list = ["earnings", "regulation", "macro", "security", "adoption", "none"]

# Numeric feature slots, in order: rsi, atr, macd, funding_rate, open_interest,
# long_short_ratio, bid_ask_imbalance, sentiment_score, sentiment_conf.
_RL_NUMERIC_FEATURE_DIM = 9

RL_OBS_DIM = len(RL_REGIMES) + _RL_NUMERIC_FEATURE_DIM + len(RL_EVENT_TYPES) + len(RL_BRAINS)

# Production canonical vocabulary (live classifier + DecisionTransformer).
CANONICAL_REGIMES: list = [
    "bull", "bear", "trending", "sideways",
    "high_volatility", "low_volatility", "neutral", "default",
]

# DT-8 -> RL-6 alias map. RL-6 labels map to themselves (pass-through).
REGIME_ALIASES: dict[str, str] = {
    "bull": "trending",
    "bear": "trending",
    "sideways": "mean_reverting",
    "high_volatility": "volatile",
    "low_volatility": "choppy",
    "neutral": "default",
    "trending": "trending",
    "mean_reverting": "mean_reverting",
    "volatile": "volatile",
    "choppy": "choppy",
    "breakout": "breakout",
    "default": "default",
}


def normalize_regime(regime) -> str:
    """Map any incoming regime label into the RL-6 one-hot space.

    Unknown/empty labels fall back to "default" instead of leaving the
    one-hot vector zeroed.
    """
    if not regime:
        return "default"
    return REGIME_ALIASES.get(regime, "default")


def is_rl_regime(regime: str) -> bool:
    """True if `regime` is already a valid RL-6 one-hot label."""
    return regime in RL_REGIMES


def _one_hot(value, vocab: list, dtype=np.float32) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=dtype)
    if value in vocab:
        vec[vocab.index(value)] = 1.0
    return vec


def build_rl_observation(regime, features: dict, votes: dict) -> np.ndarray:
    """Build the 26-dim PPO observation shared by live inference and training.

    `features` and `votes` are plain dicts (live callers convert their
    BrainVote objects first). Returns a finite float32 vector of length
    RL_OBS_DIM -- NaN/inf are zeroed so the agent never sees them.
    """
    features = features or {}
    votes = votes or {}

    rsi = (features.get("rsi", 50.0) - 50) / 50.0
    atr = features.get("atr", 0.0) / 100.0
    macd = np.clip(features.get("macd", 0.0), -1.0, 1.0)

    # On-chain features
    fr = np.clip(features.get("funding_rate", 0.0) * 1000, -1.0, 1.0)
    oi = np.clip(features.get("open_interest", 0.0) / 1e9, 0.0, 10.0)
    lsr = np.clip(features.get("long_short_ratio", 1.0) - 1.0, -1.0, 1.0)
    imb = np.clip(features.get("bid_ask_imbalance", 0.0), -1.0, 1.0)

    # Sentiment features
    sent_score = np.clip(features.get("sentiment_score", 0.0), -1.0, 1.0)
    sent_conf = np.clip(features.get("sentiment_conf", 0.0), 0.0, 1.0)

    feature_vec = np.array([rsi, atr, macd, fr, oi, lsr, imb, sent_score, sent_conf], dtype=np.float32)

    regime_vec = _one_hot(normalize_regime(regime), RL_REGIMES)
    event_vec = _one_hot(features.get("event_type", "none"), RL_EVENT_TYPES)

    vote_vec = np.zeros(len(RL_BRAINS), dtype=np.float32)
    for i, brain in enumerate(RL_BRAINS):
        action = votes.get(brain, "hold")
        if action == "buy":
            vote_vec[i] = 1.0
        elif action == "sell":
            vote_vec[i] = -1.0

    obs = np.concatenate([regime_vec, feature_vec, event_vec, vote_vec])
    return np.nan_to_num(obs, 0.0).astype(np.float32)


def position_size_multiplier(action_value: float) -> float:
    """Map raw PPO action[5] to a 0.5-1.5x position-size multiplier."""
    return ((action_value + 1.0) / 2.0) + 0.5


def confidence_threshold(action_value: float) -> float:
    """Map raw PPO action[6] to a 0.35-0.55 confidence threshold.

    Shared by rl_env (training) and rl_meta.combine (live inference) so a
    policy learned in training means the same thing live. They previously
    used different mappings (0.5-0.8 vs 0.35-0.55), silently desyncing the
    learned policy from production (fixed 2026-10-06).
    """
    return ((action_value + 1.0) / 2.0) * 0.2 + 0.35
