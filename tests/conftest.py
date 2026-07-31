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
import os

os.environ.setdefault("ALPACA_API_KEY", "test_dummy_key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test_dummy_secret")