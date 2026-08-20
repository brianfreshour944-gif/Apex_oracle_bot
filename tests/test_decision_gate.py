"""
Tests for the shared Decision Source Gate — Step 4 of Foundation Hardening.

Previously the only "verification" of this module was an import-only check
in enforce_capability_freeze.py (`python -c "from ... import ...; print('OK')"`),
which would pass even if the gate's actual logic were broken - e.g. always
returning allowed=True, or skipping a check. These tests exercise the real
fail-closed behavior the gate exists to guarantee.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.committee.decision_gate import check_decision_source_gate, GateResult


def _make_learner(sample_count=0, validated=False):
    learner = MagicMock()
    learner.sample_count_for_regime = MagicMock(return_value=sample_count)
    learner.is_regime_validated = MagicMock(return_value=validated)
    learner.get_regime_validation_metrics = MagicMock(return_value={})
    return learner


class TestDecisionSourceGate:
    def test_fails_closed_when_global_disabled(self):
        """Nothing should go live if ADAPTIVE_ML_ENABLED is False."""
        with patch("src.committee.decision_gate.settings") as mock_settings:
            mock_settings.ADAPTIVE_ML_ENABLED = False
            result = check_decision_source_gate("adaptive_learner", "sideways")
        assert result.allowed is False
        assert "ADAPTIVE_ML_ENABLED" in result.reason

    def test_fails_closed_when_learner_missing(self):
        """No adaptive learner available -> stay in shadow mode."""
        with patch("src.committee.decision_gate._get_learner", return_value=None), \
             patch("src.committee.decision_gate.settings") as mock_settings:
            mock_settings.ADAPTIVE_ML_ENABLED = True
            result = check_decision_source_gate("ppo", "sideways")
        assert result.allowed is False
        assert "not available" in result.reason.lower()

    def test_fails_closed_below_min_trades(self):
        """Insufficient regime samples -> blocked regardless of source."""
        learner = _make_learner(sample_count=2, validated=True)
        with patch("src.committee.decision_gate._get_learner", return_value=learner), \
             patch("src.committee.decision_gate.settings") as mock_settings:
            mock_settings.ADAPTIVE_ML_ENABLED = True
            mock_settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE = 30
            result = check_decision_source_gate("adaptive_learner", "sideways")
        assert result.allowed is False
        assert "Insufficient regime samples" in result.reason

    def test_fails_closed_when_not_validated(self):
        """Enough samples but Sharpe/win-rate validation failed -> still blocked.

        This is the exact gap that let the Decision Transformer go live with
        zero track record before this gate existed: enough trades alone is
        NOT sufficient, validation must also pass.
        """
        learner = _make_learner(sample_count=100, validated=False)
        with patch("src.committee.decision_gate._get_learner", return_value=learner), \
             patch("src.committee.decision_gate.settings") as mock_settings:
            mock_settings.ADAPTIVE_ML_ENABLED = True
            mock_settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE = 30
            result = check_decision_source_gate("decision_transformer", "sideways")
        assert result.allowed is False
        assert "not validated" in result.reason.lower()

    def test_allowed_when_all_gates_pass(self):
        """Enough samples + validated + model loaded -> allowed to go live."""
        learner = _make_learner(sample_count=100, validated=True)
        with patch("src.committee.decision_gate._get_learner", return_value=learner), \
             patch("src.committee.decision_gate.settings") as mock_settings:
            mock_settings.ADAPTIVE_ML_ENABLED = True
            mock_settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE = 30
            result = check_decision_source_gate("adaptive_learner", "sideways")
        assert result.allowed is True
        assert result.reason == "All gates passed"

    def test_ppo_blocked_when_model_not_loaded(self):
        """Even with enough samples and validation, PPO needs its model file loaded."""
        learner = _make_learner(sample_count=100, validated=True)
        fake_rl_learner = MagicMock()
        fake_rl_learner.model = None
        with patch("src.committee.decision_gate._get_learner", return_value=learner), \
             patch("src.committee.rl_meta.RLMetaLearner", return_value=fake_rl_learner), \
             patch("src.committee.decision_gate.settings") as mock_settings:
            mock_settings.ADAPTIVE_ML_ENABLED = True
            mock_settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE = 30
            mock_settings.PPO_MIN_TRADES_BEFORE_LIVE = 10
            result = check_decision_source_gate("ppo", "sideways")
        assert result.allowed is False
        assert "not loaded" in result.reason.lower()

    def test_decision_transformer_uses_adaptive_min_trades(self):
        """DT shares the adaptive learner's min-trades gate, not a separate
        (potentially weaker) one - confirms the fix for the original bug
        where DT had no gate at all."""
        from src.committee.decision_gate import _get_min_trades_for_source
        with patch("src.committee.decision_gate.settings") as mock_settings:
            mock_settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE = 30
            assert _get_min_trades_for_source("decision_transformer") == 30

    def test_result_is_immutable(self):
        """GateResult is frozen - callers can't accidentally mutate a cached result."""
        result = GateResult(allowed=True, source="ppo", regime="sideways", reason="x", details={})
        with pytest.raises(Exception):
            result.allowed = False