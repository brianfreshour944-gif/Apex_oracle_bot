"""Regression tests for performance fixes applied in Round 2.

These tests ensure the event-loop-blocking / redundant-call / memory-leak
fixes from the performance audit are not silently regressed by later edits.
"""
import pytest
import asyncio
import time
import os
import sys
import inspect
import ast
from unittest.mock import Mock, patch, AsyncMock

# Must be set BEFORE importing src.bot (which reads settings at import time)
os.environ.setdefault("ALPACA_API_KEY", "test_key_dummy")
os.environ.setdefault("ALPACA_SECRET_KEY", "test_secret_dummy")
os.environ.setdefault("DATABASE_URL", "sqlite:///./tmp_perf_test.db")

from src.db import init_db, save_decision_snapshot, get_open_snapshot, close_decision_snapshot, DecisionSnapshot
from src.risk import RiskManager


@pytest.fixture(scope="module")
def fresh_db():
    init_db()
    yield
    # cleanup
    import os
    for f in ['tmp_perf_test.db', 'tmp_perf_test.db-wal', 'tmp_perf_test.db-shm']:
        if os.path.exists(f):
            os.remove(f)


class TestRedundantAPICalls:
    """Finding 1 & 2: check_and_reserve_exposure must not re-fetch account
    data when current_exposure is passed in, and must not hold the lock
    across network I/O."""

    def test_no_redundant_fetches_when_exposure_provided(self, fresh_db):
        mock_ex = Mock()
        mock_ex.get_account = AsyncMock(return_value={"equity": 100000, "cash": 50000, "portfolio_value": 100000})
        mock_ex.get_positions = AsyncMock(return_value=[])
        rm = RiskManager(mock_ex)

        asyncio.run(rm.check_and_reserve_exposure(100.0, current_exposure=0.0))

        # get_account and get_positions should NOT be called when current_exposure
        # is explicitly provided -- the whole point of the fix.
        assert mock_ex.get_account.call_count == 0, \
            "check_and_reserve_exposure re-fetched account status despite receiving current_exposure"
        assert mock_ex.get_positions.call_count == 0, \
            "check_and_reserve_exposure re-fetched positions despite receiving current_exposure"

    def test_concurrent_reservation_no_lock_serialization(self):
        """15 concurrent check_and_reserve_exposure calls with precomputed
        current_exposure should complete near-instantly (no lock serialization
        of network I/O). Previously this took ~2.85s."""
        mock_ex = Mock()
        mock_ex.get_account = AsyncMock(return_value={"equity": 100000, "cash": 50000, "portfolio_value": 100000})
        mock_ex.get_positions = AsyncMock(return_value=[])
        rm = RiskManager(mock_ex)

        async def run_concurrent():
            tasks = [rm.check_and_reserve_exposure(100.0, current_exposure=float(i * 10)) for i in range(15)]
            return await asyncio.gather(*tasks)

        t0 = time.perf_counter()
        results = asyncio.run(run_concurrent())
        elapsed = time.perf_counter() - t0

        # Should complete in well under 1 second (was ~2.85s with lock serialization)
        assert elapsed < 1.0, \
            f"15 concurrent reservations took {elapsed:.2f}s -- lock serialization regression detected"
        # No network calls at all since all had current_exposure provided
        assert mock_ex.get_account.call_count == 0
        assert mock_ex.get_positions.call_count == 0


class TestPeakPriceCleanup:
    """Finding 5: peak_prices entry must be cleared on ANY position close,
    not only on trailing-stop-triggered closes."""

    def test_peak_prices_cleared_on_signal_close(self, fresh_db):
        import src.bot as bot_mod

        rm = Mock()
        rm.peak_prices = {"BTCUSD": 52000.0}
        original = bot_mod.risk_manager
        bot_mod.risk_manager = rm

        try:
            save_decision_snapshot(
                decision_id="test_sig_close",
                symbol="BTCUSD", regime="trend", final_action="buy",
                confidence=0.7, size_multiplier=1.0, entry_price=50000.0, qty=0.01,
                brain_votes={}, feature_snapshot_json="{}", causal_reasoning_json="{}",
            )

            with patch("src.committee.committee.get_meta_learner", return_value=None):
                asyncio.run(bot_mod._record_committee_outcome("BTCUSD", 51000.0, exit_reason="signal_close"))

            assert "BTCUSD" not in rm.peak_prices, \
                "peak_prices['BTCUSD'] was not cleared on signal-based position close"
        finally:
            bot_mod.risk_manager = original


class TestLoggerEagerFormatting:
    """Finding 6: logger.debug must use kwargs (lazy formatting), not f-strings."""

    def test_scan_signal_debug_uses_lazy_format(self):
        import src.bot as bot_mod

        src, _ = inspect.getsourcelines(bot_mod.process_signal_for_symbol)
        src_text = "".join(src)
        # The audit flagged the [SCAN] debug call. Verify it uses named
        # field placeholders + kwargs, NOT an f-string prefix.
        assert '[SCAN] Signal for' in src_text, "expected the [SCAN] debug call to still exist"
        # Check that no f-string is passed directly to logger.debug
        for line in src_text.split('\n'):
            if 'logger.debug(' in line and '[SCAN]' in line:
                assert 'f"' not in line and "f'" not in line, \
                    f"logger.debug still uses an f-string on line: {line.strip()}"


class TestDBIndex:
    """Finding 3: DecisionSnapshot must have a composite index on
    (symbol, status) so get_open_snapshot does not full-scan."""

    def test_symbol_status_index_exists(self, fresh_db):
        from src.db import get_engine
        from sqlalchemy import text
        with get_engine().connect() as conn:
            indexes = conn.execute(text("PRAGMA index_list('decision_snapshots')")).fetchall()
            names = [row[1] for row in indexes]
            assert any("symbol_status" in n for n in names), \
                f"Expected an index on (symbol, status) on decision_snapshots, got: {names}"

    def test_shadow_trade_index_exists(self, fresh_db):
        from src.db import get_engine
        from sqlalchemy import text
        with get_engine().connect() as conn:
            indexes = conn.execute(text("PRAGMA index_list('shadow_trades')")).fetchall()
            names = [row[1] for row in indexes]
            assert any("candidate_symbol_status" in n for n in names), \
                f"Expected an index on (candidate_name, symbol, status) on shadow_trades, got: {names}"


class TestShadowArenaNoCreateAllPerCall:
    """Finding 4: shadow_arena's process_shadow_signal must NOT call
    Base.metadata.create_all() as a runtime call on every invocation.
    Uses AST parsing to distinguish actual calls from comments/docstrings."""

    def test_no_runtime_create_all(self):
        import src.shadow_arena as sa
        src_path = inspect.getsourcefile(sa)
        with open(src_path, "r") as f:
            tree = ast.parse(f.read())

        create_all_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Match: Base.metadata.create_all(...)
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr == "create_all"
                        and isinstance(func.value, ast.Attribute)
                        and func.value.attr == "metadata"):
                    create_all_calls.append(node.lineno)

        assert not create_all_calls, \
            f"shadow_arena contains runtime create_all() call(s) at lines: {create_all_calls}"

    def test_ensure_tables_used(self):
        import src.shadow_arena as sa
        src, _ = inspect.getsourcelines(sa.process_shadow_signal)
        src_text = "".join(src)
        assert "_ensure_tables()" in src_text, \
            "process_shadow_signal should use _ensure_tables() (one-time cached) not direct create_all()"


class TestDBRetentionCleanup:
    """Finding 7: DB maintenance task must exist and be schedulable."""

    def test_db_maintenance_task_is_created(self):
        import src.bot as bot_mod
        assert hasattr(bot_mod, "run_periodic_db_maintenance"), \
            "run_periodic_db_maintenance must be defined in src.bot"
        assert inspect.iscoroutinefunction(bot_mod.run_periodic_db_maintenance), \
            "run_periodic_db_maintenance must be an async function"

        src, _ = inspect.getsourcelines(bot_mod.run_trading_bot)
        src_text = "".join(src)
        assert "run_periodic_db_maintenance" in src_text, \
            "run_trading_bot must start a periodic DB maintenance task"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])