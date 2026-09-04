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

We deliberately do NOT expand REGIMES to 8: the RL observation vector is
17-dim (6 = len(RL-6)) and that length is baked into the trained
models/ppo_meta_weights.zip input shape; changing it would desync the live
agent.
"""

from __future__ import annotations

# RL-native one-hot regime space (MUST stay length 6 -- matches trained PPO).
RL_REGIMES: list = ["trending", "mean_reverting", "volatile", "choppy", "breakout", "default"]

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
