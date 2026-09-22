"""Tests for get_ppo_model()'s failure-mode return value.

decision_gate.py's PPO readiness check is `getattr(rl_learner, "model", None)
is not None` -- get_ppo_model() must return None (not False, not any other
falsy-but-not-None value) on any failure, or that check silently reports
"loaded" for a model that never loaded. stable_baselines3 is not in
requirements.txt/pyproject.toml, so an ImportError on load is the default
production state, not an edge case. Found via an external correctness audit,
2026-09-22.
"""
import importlib

import pytest


@pytest.fixture(autouse=True)
def _reset_module_cache():
    """get_ppo_model() caches its result in module-level globals -- reload
    fresh for each test so one test's cached (non-)load doesn't leak into
    the next."""
    import src.committee.rl_meta as rl_meta_mod
    importlib.reload(rl_meta_mod)
    yield
    importlib.reload(rl_meta_mod)


class TestGetPpoModelFailureMode:
    def test_returns_none_when_weights_file_missing(self, monkeypatch):
        import src.committee.rl_meta as rl_meta_mod
        monkeypatch.setattr(rl_meta_mod.os.path, "exists", lambda path: False)

        result = rl_meta_mod.get_ppo_model()

        assert result is None
        assert result is not False, "must be None, not False -- decision_gate's 'is not None' check treats False as loaded"

    def test_returns_none_when_stable_baselines3_import_fails(self, monkeypatch):
        import builtins

        import src.committee.rl_meta as rl_meta_mod
        monkeypatch.setattr(rl_meta_mod.os.path, "exists", lambda path: True)

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "stable_baselines3":
                raise ImportError("simulated: stable_baselines3 not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        result = rl_meta_mod.get_ppo_model()

        assert result is None
        assert result is not False

    def test_decision_gate_reports_not_ready_when_model_failed_to_load(self, monkeypatch):
        """End-to-end: the actual gate check callers rely on, not just the
        raw return value."""
        import src.committee.rl_meta as rl_meta_mod
        monkeypatch.setattr(rl_meta_mod.os.path, "exists", lambda path: False)

        from src.committee.decision_gate import _check_source_specific

        allowed, reason, details = _check_source_specific("ppo", "trending")

        assert allowed is False
        assert "not loaded" in reason.lower()
        assert details["ppo_model_loaded"] is False
