"""Integration tests: committee <-> adaptive meta-learner, and risk invariants."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("ALPACA_API_KEY", "test")
os.environ.setdefault("ALPACA_SECRET_KEY", "test")

from unittest.mock import AsyncMock

import pytest

from src.config import settings
from src.committee import committee as committee_mod
from src.committee.committee import run_committee, WINNING_SCORE_THRESHOLD
from src.committee.adaptive_meta import AdaptiveMetaLearner, BRAINS
from src.risk import RiskManager


BUY_SIGNAL = {"action": "buy", "confidence": 0.92, "regime": "uptrend", "rsi": 22.0, "atr": 50.0}
CRASH_SIGNAL = {"action": "buy", "confidence": 0.85, "regime": "crash", "rsi": 15.0, "atr": 100.0}


@pytest.fixture(autouse=True)
def _restore_adaptive_settings():
    """Snapshot/restore adaptive settings + learner singleton around each test."""
    saved = {
        "enabled": settings.ADAPTIVE_ML_ENABLED,
        "path": settings.ADAPTIVE_STATE_PATH,
        "min_trades": settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE,
    }
    yield
    settings.ADAPTIVE_ML_ENABLED = saved["enabled"]
    settings.ADAPTIVE_STATE_PATH = saved["path"]
    settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE = saved["min_trades"]
    committee_mod.reset_meta_learner()


def _seed_state(path, regime="uptrend"):
    """Train a learner so one brain dominates in a regime, then persist state."""
    learner = AdaptiveMetaLearner(state_path=path, learning_rate=0.4)
    # Asymmetric: transformer keeps voting the profitable direction, the rest
    # vote against it -> transformer's weight rises above the others.
    votes = {"transformer": "buy", "quant": "sell", "momentum": "sell", "sentinel": "sell", "llm": "sell"}
    for _ in range(30):
        learner.update(
            {"regime": regime, "final_action": "buy", "brain_votes": votes},
            {"net_pnl": 100.0},
        )
    learner.save()
    return learner


async def test_committee_disabled_is_classic_and_valid(tmp_path):
    """Default (disabled) -> valid decision, adaptive not used, but shadow weights attached."""
    settings.ADAPTIVE_ML_ENABLED = False
    settings.ADAPTIVE_STATE_PATH = str(tmp_path / "state.json")
    committee_mod.reset_meta_learner()

    result = await run_committee("BTC/USD", 50000.0, BUY_SIGNAL)
    assert result.action in ["buy", "sell", "hold", "stand_aside"]
    assert result.adaptive_used is False
    assert len(result.votes) == 5
    assert result.decision_id  # correlation id always present
    # Shadow mode still surfaces the learned weighting for observability.
    assert set(result.adaptive_weights) == set(BRAINS)


async def test_committee_uses_adaptive_when_enabled(tmp_path):
    """Enabled + enough samples -> learned weights drive the decision."""
    path = str(tmp_path / "state.json")
    _seed_state(path, regime="uptrend")

    settings.ADAPTIVE_ML_ENABLED = True
    settings.ADAPTIVE_STATE_PATH = path
    settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE = 0
    committee_mod.reset_meta_learner()

    result = await run_committee("BTC/USD", 50000.0, BUY_SIGNAL)
    assert result.adaptive_used is True
    assert result.explanation and result.explanation.startswith("adaptive[uptrend]")
    # Learned weights are no longer equal (transformer et al. were rewarded).
    weights = result.adaptive_weights
    assert set(weights) == set(BRAINS)
    assert max(weights.values()) > 1.0 / len(BRAINS)


async def test_adaptive_respects_min_trades_gate(tmp_path):
    """Enabled but below MIN_TRADES_BEFORE_LIVE -> shadow only (classic drives)."""
    path = str(tmp_path / "state.json")
    _seed_state(path, regime="uptrend")  # ~30 samples

    settings.ADAPTIVE_ML_ENABLED = True
    settings.ADAPTIVE_STATE_PATH = path
    settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE = 10_000  # far above seeded samples
    settings.PPO_MIN_TRADES_BEFORE_LIVE = 10_000  # PPO gate also high
    committee_mod.reset_meta_learner()

    result = await run_committee("BTC/USD", 50000.0, BUY_SIGNAL)
    assert result.adaptive_used is False
    assert set(result.adaptive_weights) == set(BRAINS)  # still computed for shadow


async def test_adaptive_cannot_bypass_sentinel_veto(tmp_path):
    """ML must never override a hard risk veto (risk stays authoritative)."""
    path = str(tmp_path / "state.json")
    _seed_state(path, regime="crash")

    settings.ADAPTIVE_ML_ENABLED = True
    settings.ADAPTIVE_STATE_PATH = path
    settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE = 0
    committee_mod.reset_meta_learner()

    result = await run_committee("BTC/USD", 50000.0, CRASH_SIGNAL)
    assert result.vetoed is True
    assert result.action == "stand_aside"
    assert result.size_multiplier == 0.0


async def test_missing_state_enabled_falls_back_gracefully(tmp_path):
    """Enabled but no state file yet -> cold start, no crash, valid decision."""
    settings.ADAPTIVE_ML_ENABLED = True
    settings.ADAPTIVE_STATE_PATH = str(tmp_path / "nope.json")
    settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE = 0
    committee_mod.reset_meta_learner()

    result = await run_committee("BTC/USD", 50000.0, BUY_SIGNAL)
    assert result.action in ["buy", "sell", "hold", "stand_aside"]
    assert len(result.votes) == 5


def test_risk_manager_behavior_unchanged_by_adaptive():
    """Position sizing / risk math is independent of the adaptive flag."""
    rm = RiskManager(AsyncMock())

    settings.ADAPTIVE_ML_ENABLED = False
    size_off, status_off = rm.calculate_position_size("BTC/USD", 50000.0, "trending", atr=500.0, confidence=1.0)

    settings.ADAPTIVE_ML_ENABLED = True
    size_on, status_on = rm.calculate_position_size("BTC/USD", 50000.0, "trending", atr=500.0, confidence=1.0)

    assert status_off == status_on == "ok"
    assert size_off == size_on  # adaptive layer does not touch sizing
