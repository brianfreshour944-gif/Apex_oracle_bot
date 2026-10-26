"""Test isolation of the performance-tracker state file (P2, 2026-10-25).

Regression: the tracker's state path was a hardcoded module constant
(``data/performance_tracker_state.json``), so tests that record strategy
outcomes wrote through the process-wide singleton into the REAL runtime file.
A full-suite run appended one synthetic trade (``pnl=5.0, return_pct=2.0``)
every time, contaminating decay-monitoring state with fixture data.

The fix follows the ADAPTIVE_STATE_PATH precedent: the path is configurable
(``settings.PERFORMANCE_TRACKER_STATE_PATH``), resolved at construction time,
and redirected by ``tests/conftest.py`` to a temp dir.

These tests pin:
  - the default resolves from settings, not a hardcoded constant,
  - a test recording an outcome writes to the redirected path, not the real one,
  - the real repo-relative path is never touched by the suite.
"""

import json
import os

from src import performance_tracker as pt
from src.config import settings


def test_default_path_comes_from_settings():
    """A default-constructed tracker uses the configured path, not _STATE_PATH."""
    tracker = pt.PerformanceTracker()
    assert tracker.state_path == settings.PERFORMANCE_TRACKER_STATE_PATH


def test_explicit_path_still_honoured(tmp_path):
    """An explicit path overrides settings (backwards-compatible signature)."""
    p = str(tmp_path / "custom.json")
    tracker = pt.PerformanceTracker(state_path=p)
    assert tracker.state_path == p


def test_suite_state_path_is_redirected_out_of_the_repo():
    """conftest must point the tracker at a temp dir, not data/."""
    resolved = os.path.abspath(settings.PERFORMANCE_TRACKER_STATE_PATH)
    repo_real = os.path.abspath(os.path.join("data", "performance_tracker_state.json"))
    assert resolved != repo_real, "tracker state is NOT isolated from the real runtime file"


def test_record_trade_outcome_writes_redirected_path_not_real_file():
    """Recording an outcome must write the redirected file, never data/."""
    real = os.path.abspath(os.path.join("data", "performance_tracker_state.json"))
    before = None
    if os.path.exists(real):
        with open(real, "rb") as f:
            before = f.read()

    pt.record_trade_outcome("trend_following", "bull", pnl=5.0, return_pct=2.0)

    # The real file is byte-identical after the write.
    if before is not None:
        with open(real, "rb") as f:
            assert f.read() == before, "the real tracker state file was mutated"
    else:
        assert not os.path.exists(real), "the real tracker state file was created"

    # And the write actually landed somewhere (the redirected path).
    redirected = settings.PERFORMANCE_TRACKER_STATE_PATH
    assert os.path.exists(redirected)
    with open(redirected) as f:
        data = json.load(f)
    assert "trend_following|bull" in data
    assert len(data["trend_following|bull"]["trades"]) >= 1
