"""Test isolation of the feature-drift monitor state file (2026-10-25).

Same defect class as the performance-tracker leak fixed alongside it:
``src/feature_drift_monitor.py`` hardcoded a RELATIVE default
(``data/feature_drift_state.json``), and ``tests/conftest.py`` did not redirect
it, so a full-suite run wrote the real runtime file via
``strategies.py:470`` -> ``record_features_for_drift`` -> ``_save_state()``.

The fix follows the same precedent: the path is configurable
(``settings.FEATURE_DRIFT_STATE_PATH``), resolved at construction time, and
redirected by ``tests/conftest.py`` to a temp dir.

These tests pin:
  - the default resolves from settings, not a hardcoded constant,
  - an explicit path is still honoured (backwards-compatible signature),
  - the suite path is redirected out of the repo,
  - recording features writes the redirected path, never the real file.
"""

import json
import os

from src import feature_drift_monitor as fdm
from src.config import settings


def test_default_path_comes_from_settings():
    monitor = fdm.FeatureDriftMonitor()
    assert monitor.state_path == settings.FEATURE_DRIFT_STATE_PATH


def test_explicit_path_still_honoured(tmp_path):
    p = str(tmp_path / "custom.json")
    monitor = fdm.FeatureDriftMonitor(state_path=p)
    assert monitor.state_path == p


def test_suite_state_path_is_redirected_out_of_the_repo():
    resolved = os.path.abspath(settings.FEATURE_DRIFT_STATE_PATH)
    repo_real = os.path.abspath(os.path.join("data", "feature_drift_state.json"))
    assert resolved != repo_real, "drift state is NOT isolated from the real runtime file"


def test_record_features_writes_redirected_path_not_real_file():
    real = os.path.abspath(os.path.join("data", "feature_drift_state.json"))
    before = None
    if os.path.exists(real):
        with open(real, "rb") as f:
            before = f.read()

    fdm.record_features_for_drift("BTC/USD", {"rsi": 50.0, "atr": 1.0})

    if before is not None:
        with open(real, "rb") as f:
            assert f.read() == before, "the real drift state file was mutated"
    else:
        assert not os.path.exists(real), "the real drift state file was created"

    redirected = settings.FEATURE_DRIFT_STATE_PATH
    assert os.path.exists(redirected)
    with open(redirected) as f:
        data = json.load(f)
    assert "reference_windows" in data
