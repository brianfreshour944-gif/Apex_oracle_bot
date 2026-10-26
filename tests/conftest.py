"""
Ensures the test suite can run without real Alpaca credentials.

src/config.py validates that ALPACA_API_KEY and ALPACA_SECRET_KEY are set,
and raises at import time if they're missing. That's the right behavior for
running the actual bot, but it means every test file that imports anything
touching src.config (directly or transitively, e.g. src.exchange) fails to
even collect without real-looking credentials in the environment.

This conftest sets safe dummy values before any test module is imported, so
tests that have nothing to do with Alpaca (formatting logic, risk math, etc.)
aren't blocked by a credentials check they don't need. This must be set
before pytest imports any test module, which is why it lives here rather
than in a fixture (fixtures run too late — the ValidationError happens at
import time).

If you need to test against real credentials for a specific test, override
these via monkeypatch inside that test instead of relying on these dummies.
"""
import atexit
import os
import shutil
import tempfile

os.environ.setdefault("ALPACA_API_KEY", "test_dummy_key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test_dummy_secret")

# Isolate the suite from the bot's real runtime state. Both defaults are
# RELATIVE paths (sqlite:///data/bot.db, data/adaptive_meta_state.json), the
# same files a paper bot started from the repo root uses, and nothing redirected
# them before -- test_integration_trading_loop.py wrote one fake losing
# "profit_target_reached" BTC/USD trade (entry 50050.0, pnl -5.0) into the real
# DB and fed it to the adaptive learner on every run, which then marked the
# "trending" regime validated from fixture data. A full-suite run in an isolated
# copy showed these are the only two real files it mutates. Environment
# variables beat .env in pydantic-settings, and this runs before any test module
# imports src.config, so forcing them here (not setdefault -- tests must never
# fall back to the real files) covers every test. Tests that need their own DB
# still override settings.DATABASE_URL themselves, as before.
_TEST_STATE_DIR = tempfile.mkdtemp(prefix="apex_test_state_")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TEST_STATE_DIR, "bot.db").replace("\\", "/")
os.environ["ADAPTIVE_STATE_PATH"] = os.path.join(_TEST_STATE_DIR, "adaptive_meta_state.json")
# The performance tracker's state path is likewise a RELATIVE default
# (data/performance_tracker_state.json). tests/test_strategy_selector_seeding.py
# calls record_strategy_outcome(), which writes through the process-wide
# singleton to that real file -- a full-suite run appended one synthetic trade
# (pnl=5.0, return_pct=2.0) every time, corrupting decay-monitoring state with
# fixture data. Redirect it here so the suite can never touch the real file.
os.environ["PERFORMANCE_TRACKER_STATE_PATH"] = os.path.join(_TEST_STATE_DIR, "performance_tracker_state.json")
# Same defect class, separate file: src/feature_drift_monitor.py also hardcoded
# a RELATIVE default (data/feature_drift_state.json), so the suite wrote the real
# runtime file via strategies.py:470 -> record_features_for_drift -> _save_state().
# (Verified: its mtime advanced across full-suite runs.) Redirect it too.
os.environ["FEATURE_DRIFT_STATE_PATH"] = os.path.join(_TEST_STATE_DIR, "feature_drift_state.json")
os.environ["APEX_LIVE_BUFFER_PATH"] = os.path.join(_TEST_STATE_DIR, "live_experiences.jsonl")
os.environ["APEX_MODEL_STORE_DIR"] = os.path.join(_TEST_STATE_DIR, "models")
atexit.register(shutil.rmtree, _TEST_STATE_DIR, ignore_errors=True)