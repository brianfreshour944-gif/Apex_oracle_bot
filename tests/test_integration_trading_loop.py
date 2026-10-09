"""Integration tests for the full trading loop."""

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import settings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.bot import process_signal_for_symbol
from src.committee.committee import run_committee
from src.committee.models import BrainVote
from src.exchange import AlpacaExchange
from src.risk import RiskManager
from src.strategies import TradingStrategy


def _deterministic_committee_result(action: str = "buy", score: float = 0.62):
    """CommitteeResult with two directional votes so the entry-agreement
    floor (MIN_ENTRY_DIRECTIONAL_VOTES=2) is satisfied deterministically,
    independent of what the real committee brains do in the test env."""
    from src.committee.models import BrainVote, CommitteeResult

    votes = [
        BrainVote(name="momentum", action=action, confidence=0.7, weight=0.3,
                  regime="trending", reason="test"),
        BrainVote(name="quant", action=action, confidence=0.6, weight=0.3,
                  regime="trending", reason="test"),
    ]
    return CommitteeResult(
        action=action, score=score, size_multiplier=1.0, entropy=0.0,
        votes=votes, decision_id="test-decision-id",
    )


class TestFullTradingLoop:
    """Integration tests for the complete trading loop."""

    @pytest.fixture(autouse=True)
    def _reset_last_fill_times(self):
        """Isolate the fill-time tracker between tests: a fill recorded in
        one test must not veto the next test's entry via ENTRY_RACE_GUARD."""
        import src.bot as bot_mod
        bot_mod._state.last_fill_times = {}
        bot_mod._state.protective_stops = {}
        yield
        bot_mod._state.last_fill_times = {}
        bot_mod._state.protective_stops = {}

    async def _run_buy_with_patched_committee(self, mock_exchange, mock_strategy,
                                              mock_risk_manager, action="buy",
                                              score=0.62):
        """Run the buy flow with a deterministic committee verdict so sizing
        and the entry gates don't depend on live brain behaviour."""
        async def _fake_run_committee(symbol, price, signal):
            return _deterministic_committee_result(action=action, score=score)
        with patch("src.committee.committee.run_committee", new=_fake_run_committee):
            await self._run_buy(mock_exchange, mock_strategy, mock_risk_manager)

    @pytest.fixture
    def mock_exchange(self):
        """Create a mock AlpacaExchange."""
        ex = AsyncMock(spec=AlpacaExchange)
        ex.get_positions = AsyncMock(return_value=[])
        ex.get_latest_bar = AsyncMock()
        ex.create_order = AsyncMock(return_value={
            "id": "test_order_123",
            "filled_avg_price": 50000.0,
            "filled_qty": 0.1,
            "commission": 5.0,
            "status": "filled"
        })
        ex.get_account = AsyncMock(return_value={
            "equity": 10000.0,
            "cash": 10000.0,
            "portfolio_value": 10000.0
        })
        ex.load = AsyncMock()
        ex.close = AsyncMock()
        return ex

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock TradingStrategy."""
        strat = AsyncMock(spec=TradingStrategy)
        strat.generate_trading_signal = AsyncMock(return_value={
            "action": "buy",
            "confidence": 0.75,
            "regime": "trending",
            "rsi": 55.0,
            "atr": 500.0,
            "features": {"hurst": 0.65, "atr": 500.0}
        })
        return strat

    @pytest.fixture
    def mock_risk_manager(self):
        """Create a mock RiskManager."""
        rm = AsyncMock(spec=RiskManager)
        rm.update_account_status = AsyncMock(return_value={
            "status": "risk_ok",
            "equity": 10000.0,
            "cash": 10000.0,
            "portfolio_value": 10000.0,
            "drawdown_pct": -2.0,
            "daily_pnl": 100.0,
            "open_positions": 0,
            "current_exposure": 0.0
        })
        rm.is_killswitch_active = MagicMock(return_value=False)
        rm.check_killswitch_conditions = AsyncMock(return_value=False)
        rm.check_trailing_stop = MagicMock(return_value="hold")
        # calculate_position_size is a synchronous function in the real
        # RiskManager (src/risk.py) -- AsyncMock here was masking a real bug
        # where bot.py incorrectly awaited it (confirmed + fixed 2026-09-21:
        # awaiting a sync function raises TypeError on every single trade
        # signal, silently swallowed by the fire-and-forget task dispatch in
        # bot.py's main loop, meaning the bot would place zero trades while
        # appearing to run normally). Must stay MagicMock to actually catch
        # a regression of this exact bug.
        rm.calculate_position_size = MagicMock(return_value=(0.1, "ok"))
        rm.check_and_reserve_exposure = AsyncMock(return_value=(1000.0, "ok"))
        rm.reserve_position_slot = AsyncMock(return_value=(True, "ok"))
        # async in the real RiskManager as of 2026-09-22 (wrapped in
        # _exposure_lock for consistency with release_reserved_exposure,
        # per an external concurrency audit) -- must be AsyncMock or bot.py's
        # `await risk_manager.release_position_slot(...)` raises TypeError.
        rm.release_position_slot = AsyncMock()
        rm.peak_prices = {}
        rm.record_fill_costs = MagicMock()
        return rm

    async def _run_buy(self, mock_exchange, mock_strategy, mock_risk_manager):
        import pandas as pd
        import polars as pl

        mock_exchange.get_latest_bar = AsyncMock(return_value=pl.DataFrame({
            "t": [pd.Timestamp.now(tz="UTC")], "open": [50000.0], "high": [50100.0],
            "low": [49900.0], "close": [50050.0], "volume": [100.0], "vwap": [50025.0],
            "trade_count": [100],
        }))
        await process_signal_for_symbol(
            symbol="BTC/USD", current_price=50050.0, risk_manager=mock_risk_manager,
            strategy=mock_strategy, ex=mock_exchange, positions=[], regime_flag=None,
            banned_symbols=set(),
        )

    @pytest.mark.asyncio
    async def test_unresolved_order_blocks_new_entry(self, mock_exchange, mock_strategy, mock_risk_manager):
        """A stale order still open on the exchange (e.g. submitted before a
        crash) must block a second entry for that symbol -- otherwise the
        next signal re-fetches still-empty positions and doubles exposure.
        Same flow as test_full_buy_signal_flow, which does place an order."""
        import src.bot as bot_mod
        bot_mod._state.symbols_with_unresolved_orders = {"BTCUSD"}
        try:
            await self._run_buy(mock_exchange, mock_strategy, mock_risk_manager)
        finally:
            bot_mod._state.symbols_with_unresolved_orders = set()
        mock_exchange.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_incomplete_reconciliation_blocks_new_entry(self, mock_exchange, mock_strategy, mock_risk_manager):
        """If startup reconciliation couldn't read positions, the exchange
        state is unknown: new entries must wait for a successful retry."""
        import src.bot as bot_mod
        bot_mod._state.reconciliation_incomplete = True
        try:
            await self._run_buy(mock_exchange, mock_strategy, mock_risk_manager)
        finally:
            bot_mod._state.reconciliation_incomplete = False
        mock_exchange.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_order_never_exceeds_max_single_trade(self, mock_exchange, mock_strategy, mock_risk_manager):
        """Sizing asks for ~$500k and the exposure check approves whatever is
        requested, so only the MAX_SINGLE_TRADE_USD cap (including the re-cap
        after the committee confidence multiplier) stands between the signal
        and the order."""
        from src.config import settings
        mock_risk_manager.calculate_position_size = MagicMock(return_value=(10.0, "ok"))
        mock_risk_manager.check_and_reserve_exposure = AsyncMock(
            side_effect=lambda notional, *a, **k: (notional, "ok"))
        await self._run_buy(mock_exchange, mock_strategy, mock_risk_manager)
        mock_exchange.create_order.assert_called_once()
        qty = mock_exchange.create_order.call_args.kwargs["qty"]
        assert qty * 50050.0 <= settings.MAX_SINGLE_TRADE_USD * 1.0001

    @pytest.mark.asyncio
    async def test_filled_order_is_written_to_order_ledger(self, mock_exchange, mock_strategy, mock_risk_manager):
        """Startup reconciliation re-attaches a crash-gap fill instead of
        liquidating it only if the order ledger has a record of it; the ledger
        write had zero callers before 2026-09-22."""
        from src.db import get_recent_order_records
        await self._run_buy(mock_exchange, mock_strategy, mock_risk_manager)
        records = get_recent_order_records("BTCUSD")
        assert any(r["order_id"] == "test_order_123" and r["side"] == "buy" for r in records)

    @pytest.mark.asyncio
    async def test_trailing_stop_uses_fresh_position_not_cycle_snapshot(
        self, mock_exchange, mock_strategy, mock_risk_manager
    ):
        """The main loop passes a cycle-start positions snapshot (up to one
        full loop interval old). The trailing-stop check must decide on a
        fresh fetch, like the buy path does."""
        stale = [{"symbol": "BTCUSD", "qty": "0.1", "avg_entry_price": 40000.0}]
        fresh = [{"symbol": "BTCUSD", "qty": "0.2", "avg_entry_price": 45000.0}]
        mock_exchange.get_positions = AsyncMock(return_value=fresh)
        mock_strategy.generate_trading_signal = AsyncMock(return_value={
            "action": "hold", "confidence": 0.5, "regime": "trending", "rsi": 50.0, "atr": 500.0,
        })
        await process_signal_for_symbol(
            symbol="BTC/USD", current_price=50050.0, risk_manager=mock_risk_manager,
            strategy=mock_strategy, ex=mock_exchange, positions=stale, regime_flag=None,
            banned_symbols=set(),
        )
        args = mock_risk_manager.check_trailing_stop.call_args.args
        assert args[2] == 45000.0 and args[3] == 0.2

    @pytest.mark.asyncio
    async def test_full_buy_signal_flow(self, mock_exchange, mock_strategy, mock_risk_manager):
        """Test complete buy signal processing flow."""
        
        # Setup mock bar data
        import pandas as pd
        import polars as pl
        
        mock_exchange.get_latest_bar = AsyncMock(return_value=pl.DataFrame({
            "t": [pd.Timestamp.now(tz="UTC")],
            "open": [50000.0],
            "high": [50100.0],
            "low": [49900.0],
            "close": [50050.0],
            "volume": [100.0],
            "vwap": [50025.0],
            "trade_count": [100]
        }))

        # Process signal
        await process_signal_for_symbol(
            symbol="BTC/USD",
            current_price=50050.0,
            risk_manager=mock_risk_manager,
            strategy=mock_strategy,
            ex=mock_exchange,
            positions=[],
            regime_flag=None,
            banned_symbols=set()
        )

        # Verify order was placed
        mock_exchange.create_order.assert_called_once()
        call_args = mock_exchange.create_order.call_args
        assert call_args.kwargs["symbol"] == "BTC/USD"
        assert call_args.kwargs["side"] == "buy"
        assert call_args.kwargs["type"] == "market"

        # A successful new-entry fill must NOT release its position-slot OR
        # its exposure reservation -- doing so reopens the exact race each
        # exists to prevent for any sibling symbol evaluated later in the
        # same cycle (see risk.py reserve_position_slot /
        # check_and_reserve_exposure docstrings). Both should instead be
        # left to expire via their own 30s TTL.
        mock_risk_manager.release_position_slot.assert_not_called()
        mock_risk_manager.release_reserved_exposure.assert_not_called()

    @pytest.mark.asyncio
    async def test_recent_fill_blocks_immediate_reentry(self, mock_exchange, mock_strategy, mock_risk_manager):
        """ENTRY_RACE_GUARD: a fill recorded seconds ago (not yet visible in
        the cycle-start positions snapshot) must veto a fresh BUY. This is
        the 2026-10-05 double-entry race: two equal-size ETH buys 59s apart."""
        import src.bot as bot_mod

        bot_mod._state.last_fill_times["BTC/USD"] = time.time() - 5.0
        try:
            await self._run_buy_with_patched_committee(
                mock_exchange, mock_strategy, mock_risk_manager
            )
        finally:
            bot_mod._state.last_fill_times.pop("BTC/USD", None)
        mock_exchange.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_old_fill_does_not_block_new_entry(self, mock_exchange, mock_strategy, mock_risk_manager):
        """A fill older than ENTRY_RACE_GUARD_SECONDS must not block a fresh
        BUY, and the successful fill must record a new fill time."""
        import src.bot as bot_mod

        bot_mod._state.last_fill_times["BTC/USD"] = time.time() - (
            settings.ENTRY_RACE_GUARD_SECONDS + 60.0
        )
        fill_recorded: list[float] = []
        try:
            await self._run_buy_with_patched_committee(
                mock_exchange, mock_strategy, mock_risk_manager
            )
            # Read inside the try: the finally below clears the tracker.
            fill_recorded.append(bot_mod._state.last_fill_times.get("BTC/USD", 0.0))
        finally:
            bot_mod._state.last_fill_times.pop("BTC/USD", None)
        mock_exchange.create_order.assert_called_once()
        assert fill_recorded and fill_recorded[0] > 0.0

    @pytest.mark.asyncio
    async def test_small_buy_bumped_to_equity_notional_floor(self, mock_exchange, mock_strategy, mock_risk_manager):
        """MIN_ENTRY_NOTIONAL_EQUITY_PCT: a ~$10 entry on a $100 account is
        bumped to the equity-based floor (15% of equity = $15) rather than
        trading a dust-size fill whose fees dominate the round trip."""
        mock_risk_manager.update_account_status = AsyncMock(return_value={
            "status": "risk_ok", "equity": 100.0, "cash": 100.0,
            "portfolio_value": 100.0, "drawdown_pct": 0.0, "daily_pnl": 0.0,
            "open_positions": 0, "current_exposure": 0.0,
        })
        mock_risk_manager.calculate_position_size = MagicMock(return_value=(0.0002, "ok"))
        await self._run_buy_with_patched_committee(
            mock_exchange, mock_strategy, mock_risk_manager
        )
        mock_exchange.create_order.assert_called_once()
        qty = mock_exchange.create_order.call_args.kwargs["qty"]
        notional = qty * 50050.0
        floor = max(10.0, settings.MIN_ENTRY_NOTIONAL_EQUITY_PCT * 100.0)
        assert notional >= floor * 0.99
        assert notional < floor * 2.0

    @pytest.mark.asyncio
    async def test_tiny_buy_below_half_floor_vetoed(self, mock_exchange, mock_strategy, mock_risk_manager):
        """An entry less than half the effective notional floor is vetoed
        (bumping it >2x would be too large a deviation) and must release the
        exposure reservation it made before the veto."""
        mock_risk_manager.update_account_status = AsyncMock(return_value={
            "status": "risk_ok", "equity": 100.0, "cash": 100.0,
            "portfolio_value": 100.0, "drawdown_pct": 0.0, "daily_pnl": 0.0,
            "open_positions": 0, "current_exposure": 0.0,
        })
        mock_risk_manager.calculate_position_size = MagicMock(return_value=(0.00005, "ok"))
        await self._run_buy_with_patched_committee(
            mock_exchange, mock_strategy, mock_risk_manager
        )
        mock_exchange.create_order.assert_not_called()
        mock_risk_manager.release_reserved_exposure.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_position_flow(self, mock_exchange, mock_strategy, mock_risk_manager):
        """Test position close flow."""
        
        # Setup existing position
        
        mock_exchange.get_positions = AsyncMock(return_value=[
            {"symbol": "BTC/USD", "qty": "0.1", "side": "long", "avg_entry_price": 50000.0, "market_value": 5000.0}
        ])
        
        # Signal to close
        mock_strategy.generate_trading_signal = AsyncMock(return_value={
            "action": "close",
            "confidence": 1.0,
            "regime": "trending",
            "reason": "profit_target_reached",
            "atr": 500.0
        })
        
        mock_exchange.get_latest_bar = AsyncMock(return_value=__import__("polars").DataFrame({
            "t": [__import__("pandas").Timestamp.now(tz="UTC")],
            "open": [51000.0],
            "high": [51100.0],
            "low": [50900.0],
            "close": [51050.0],
            "volume": [100.0],
            "vwap": [51025.0],
            "trade_count": [100]
        }))
        
        await process_signal_for_symbol(
            symbol="BTC/USD",
            current_price=51050.0,
            risk_manager=mock_risk_manager,
            strategy=mock_strategy,
            ex=mock_exchange,
            positions=[{"symbol": "BTC/USD", "qty": "0.1", "side": "long", "avg_entry_price": 50000.0}],
            regime_flag=None,
            banned_symbols=set()
        )
        
        # Verify close order was placed
        mock_exchange.create_order.assert_called_once()
        call_args = mock_exchange.create_order.call_args
        assert call_args.kwargs["side"] == "sell"
        assert call_args.kwargs["qty"] == 0.1

    @pytest.mark.asyncio
    async def test_trailing_stop_trigger(self, mock_exchange, mock_strategy, mock_risk_manager):
        """Test trailing stop triggers position close."""
        
        # Setup position with profit
        mock_exchange.get_positions = AsyncMock(return_value=[
            {"symbol": "BTC/USD", "qty": "0.1", "side": "long", "avg_entry_price": 50000.0, "market_value": 5500.0}
        ])
        
        mock_risk_manager.check_trailing_stop = MagicMock(return_value="close")
        
        mock_exchange.get_latest_bar = AsyncMock(return_value=__import__("polars").DataFrame({
            "t": [__import__("pandas").Timestamp.now(tz="UTC")],
            "open": [54000.0],
            "high": [51100.0],
            "low": [51900.0],
            "close": [54050.0],
            "volume": [100.0],
            "vwap": [54025.0],
            "trade_count": [100]
        }))
        
        await process_signal_for_symbol(
            symbol="BTC/USD",
            current_price=54050.0,
            risk_manager=mock_risk_manager,
            strategy=mock_strategy,
            ex=mock_exchange,
            positions=[{"symbol": "BTC/USD", "qty": "0.1", "side": "long", "avg_entry_price": 50000.0, "market_value": 54000.0}],
            regime_flag=None,
            banned_symbols=set()
        )
        
        # Should have triggered trailing stop close
        mock_exchange.create_order.assert_called_once()
        call_args = mock_exchange.create_order.call_args
        assert call_args.kwargs["side"] == "sell"
        assert call_args.kwargs["qty"] == 0.1

    @pytest.mark.asyncio
    async def test_scale_in_gates_enforced_without_exposure_scaling(self, mock_exchange, mock_strategy, mock_risk_manager):
        """
        Regression test for the position-pyramid safety gates being nested
        one level too deep -- inside `if approved_notional < notional:` --
        so they only ran when the order was scaled down due to exposure
        headroom. In the normal case (full notional approved, no scaling)
        the bot could pyramid into an existing position with none of
        MAX_POSITION_ADDS / POSITION_ADD_MIN_SECONDS /
        POSITION_ADD_MIN_SCORE_INCREASE / POSITION_ADD_SIZE_DECAY applied.

        This constructs exactly the previously-broken case: approved_notional
        == notional (echoed back by check_and_reserve_exposure regardless of
        the exact position_size, so this doesn't depend on predicting the
        committee/regime size multipliers upstream), with the symbol already
        at MAX_POSITION_ADDS -- and asserts the gate still vetoes the add.
        """
        from src.bot import _state

        symbol = "BTC/USD"
        current_price = 50050.0

        # Echo back whatever notional was requested -> approved_notional ==
        # notional always, i.e. no exposure-headroom scaling ever occurs.
        async def _approve_in_full(notional, current_exposure=None):
            return (notional, "ok")
        mock_risk_manager.check_and_reserve_exposure = AsyncMock(side_effect=_approve_in_full)

        # Already at the MAX_POSITION_ADDS cap (default 2) for this symbol --
        # a further add must be vetoed regardless of exposure scaling.
        _state.position_adds[symbol] = {"count": 2, "last_add_time": 0.0, "last_add_score": 0.0}
        try:
            existing_position = {"symbol": symbol, "qty": "0.1", "avg_entry_price": 49000.0, "market_value": 5000.0}
            mock_exchange.get_positions = AsyncMock(return_value=[existing_position])

            await process_signal_for_symbol(
                symbol=symbol,
                current_price=current_price,
                risk_manager=mock_risk_manager,
                strategy=mock_strategy,
                ex=mock_exchange,
                positions=[existing_position],
                regime_flag=None,
                banned_symbols=set(),
            )

            # The MAX_POSITION_ADDS gate must have vetoed this add -- no
            # order should have been placed.
            mock_exchange.create_order.assert_not_called()
        finally:
            _state.position_adds.pop(symbol, None)

    @pytest.mark.asyncio
    async def test_scale_in_entry_fee_is_summed_via_open_snapshot_decision_id(
        self, mock_exchange, mock_strategy, mock_risk_manager
    ):
        """A scale-in add must be persisted under the OPEN snapshot's
        decision_id -- not the add's throwaway committee id -- so the snapshot
        close (db.get_entry_fee_estimate, keyed on decision_id) sums BOTH buys'
        fees. Driven through the real process_signal_for_symbol buy path rather
        than hand-built ledger rows, because the defect is in WHICH id the buy
        path records the row under, which hand-built rows can't exercise.
        """
        import pandas as pd
        import polars as pl

        import src.bot as bot_mod
        from src.committee.models import BrainVote, CommitteeResult
        from src.db import get_entry_fee_estimate, get_recent_order_records

        # Two distinct committee decision_ids: the fresh entry mints its own,
        # the scale-in add mints a throwaway one that must NOT key its row.
        ids = ["entry-dec", "add-dec"]

        async def _committee(symbol, price, signal):
            return CommitteeResult(
                action="buy", score=0.62, size_multiplier=1.0, entropy=0.0,
                votes=[
                    BrainVote(name="momentum", action="buy", confidence=0.7, weight=0.3,
                              regime="trending", reason="test"),
                    BrainVote(name="quant", action="buy", confidence=0.6, weight=0.3,
                              regime="trending", reason="test"),
                ],
                decision_id=ids.pop(0),
            )

        mock_exchange.get_latest_bar = AsyncMock(return_value=pl.DataFrame({
            "t": [pd.Timestamp.now(tz="UTC")], "open": [50000.0], "high": [50100.0],
            "low": [49900.0], "close": [50050.0], "volume": [100.0], "vwap": [50025.0],
            "trade_count": [100],
        }))

        # Clear any entry cooldown / add-tracking left by earlier tests in this
        # class (the class fixture resets fill times and stops, not these), or
        # the first buy is skipped as "on entry cooldown".
        bot_mod._state.cooldowns.clear()
        bot_mod._state.position_adds.clear()
        # Drop any open BTC/USD snapshot a prior test left behind: the suite
        # shares one DB, and a leftover open snapshot would make entry 1 fold
        # into it (creating no "entry-dec") and defeat the assertion. Clear the
        # in-memory snapshot cache too, or a stale cached dict is returned.
        import src.db as _db
        from src.db import DecisionSnapshot, get_db_session
        _db._open_snapshot_cache.clear()
        with get_db_session() as _s:
            _s.query(DecisionSnapshot).filter_by(symbol="BTC/USD", status="open").delete()
            _s.commit()

        # Distinct order ids per fill, or the ledger upserts both writes onto
        # one row and the sum can't be observed.
        order_ids = iter(["entry-ord", "add-ord"])

        async def _create_order(**_kwargs):
            return {"id": next(order_ids), "filled_avg_price": 50000.0,
                    "filled_qty": 0.1, "commission": 5.0, "status": "filled"}
        mock_exchange.create_order = AsyncMock(side_effect=_create_order)

        # ENTRY_RACE_GUARD would veto the second (scale-in) buy seconds after
        # the first fill; disable it so both buys go out in this one test.
        with patch.object(settings, "ENTRY_RACE_GUARD_SECONDS", 0), \
             patch("src.committee.committee.run_committee", new=_committee):
            # Entry 1 -- fresh buy, no position yet.
            mock_exchange.get_positions = AsyncMock(return_value=[])
            await process_signal_for_symbol(
                symbol="BTC/USD", current_price=50050.0, risk_manager=mock_risk_manager,
                strategy=mock_strategy, ex=mock_exchange, positions=[],
                regime_flag=None, banned_symbols=set(),
            )

            # Entry 2 -- scale-in add; a position is now held.
            held = [{"symbol": "BTC/USD", "qty": "0.1", "side": "long",
                     "avg_entry_price": 50050.0, "market_value": 5005.0}]
            mock_exchange.get_positions = AsyncMock(return_value=held)
            await process_signal_for_symbol(
                symbol="BTC/USD", current_price=50050.0, risk_manager=mock_risk_manager,
                strategy=mock_strategy, ex=mock_exchange, positions=held,
                regime_flag=None, banned_symbols=set(),
            )

        assert mock_exchange.create_order.await_count == 2
        # Scope to THIS test's two ledger rows; the suite shares one DB and
        # earlier tests leave their own BTC/USD buy rows behind.
        buys = [r for r in get_recent_order_records("BTCUSD")
                if r["order_id"] in {"entry-ord", "add-ord"}]
        assert len(buys) == 2
        # The scale-in row carries the OPEN snapshot's id, not "add-dec".
        assert {r["decision_id"] for r in buys} == {"entry-dec"}
        # Both legs' fees (5.0 each) are summed for the snapshot the position
        # closes on; the add's throwaway id keys nothing.
        assert get_entry_fee_estimate("BTCUSD", 50050.0, "entry-dec") == pytest.approx(10.0)
        assert get_entry_fee_estimate("BTCUSD", 50050.0, "add-dec") == 0.0

    @pytest.mark.asyncio
    async def test_rolling_loss_limit_does_not_block_exit(self, mock_exchange, mock_strategy, mock_risk_manager):
        """Regression (reproduced 2026-10-05): the rolling soft loss-limit was
        a top-of-function early return, so on a losing day (exactly when it
        trips) it also skipped the trailing-stop / SL / TP exit checks -- an
        already-open loser was left unmanaged and bled further as the market
        fell. It must gate NEW entries only; an open position must still exit.
        """
        losing = [{"symbol": "BTCUSD", "qty": 0.1, "avg_entry_price": 40000.0}]
        mock_exchange.get_positions = AsyncMock(return_value=losing)
        mock_risk_manager.check_trailing_stop = MagicMock(return_value="close")
        mock_risk_manager.peak_equity = 10000.0

        with patch.object(settings, "ROLLING_LOSS_LIMIT_PCT", 1.0), \
             patch.object(settings, "LOSS_LIMIT_WINDOW_HOURS", 6.0), \
             patch("src.bot.get_recent_realized_pnl", return_value=-500.0):
            await process_signal_for_symbol(
                symbol="BTC/USD", current_price=40000.0, risk_manager=mock_risk_manager,
                strategy=mock_strategy, ex=mock_exchange, positions=losing,
                regime_flag=None, banned_symbols=set(),
            )

        mock_exchange.create_order.assert_called_once()
        assert mock_exchange.create_order.call_args.kwargs["side"] == "sell"

    @pytest.mark.asyncio
    async def test_rolling_loss_limit_still_blocks_new_entry(self, mock_exchange, mock_strategy, mock_risk_manager):
        """The limit must still refuse NEW entries while it is tripped -- the
        fix relocates the gate, it does not disable it."""
        mock_risk_manager.peak_equity = 10000.0
        with patch.object(settings, "ROLLING_LOSS_LIMIT_PCT", 1.0), \
             patch.object(settings, "LOSS_LIMIT_WINDOW_HOURS", 6.0), \
             patch("src.bot.get_recent_realized_pnl", return_value=-500.0):
            await self._run_buy(mock_exchange, mock_strategy, mock_risk_manager)

        mock_exchange.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_rolling_loss_limit_does_not_block_sell_exit(self, mock_exchange, mock_strategy, mock_risk_manager):
        """A committee-overridden 'sell' while holding a long is an EXIT (the
        backtester closes the held long on sell/close, applying entries_blocked
        only to new buys). It must not be trapped by the rolling loss-limit."""
        from src.committee.models import CommitteeResult
        losing = [{"symbol": "BTCUSD", "qty": 0.1, "avg_entry_price": 40000.0}]
        mock_exchange.get_positions = AsyncMock(return_value=losing)
        mock_strategy.generate_trading_signal = AsyncMock(return_value={
            "action": "hold", "confidence": 0.5, "regime": "trending",
            "rsi": 50.0, "atr": 500.0, "features": {},
        })
        mock_risk_manager.peak_equity = 10000.0

        async def _sell_committee(symbol, price, signal):
            return CommitteeResult(
                action="sell", score=0.62, size_multiplier=1.0, entropy=0.0,
                votes=[], decision_id="sell-exit",
            )

        with patch.object(settings, "ROLLING_LOSS_LIMIT_PCT", 1.0), \
             patch.object(settings, "LOSS_LIMIT_WINDOW_HOURS", 6.0), \
             patch("src.bot.get_recent_realized_pnl", return_value=-500.0), \
             patch("src.committee.committee.run_committee", new=_sell_committee):
            await process_signal_for_symbol(
                symbol="BTC/USD", current_price=40000.0, risk_manager=mock_risk_manager,
                strategy=mock_strategy, ex=mock_exchange, positions=losing,
                regime_flag=None, banned_symbols=set(),
            )

        mock_exchange.create_order.assert_called_once()
        assert mock_exchange.create_order.call_args.kwargs["side"] == "sell"


class TestProtectiveStops:
    """Exchange-side protective stop_limit orders.

    Alpaca crypto has no plain `stop`, no trailing stop and no bracket/OCO
    (simple order class only), so the bot arms a standalone stop_limit sell
    at -STOP_LOSS_PCT from the real fill. Gated behind the default-OFF
    PROTECTIVE_STOPS_ENABLED flag.
    """

    @pytest.fixture(autouse=True)
    def _reset_protective_stops(self):
        import src.bot as bot_mod
        bot_mod._state.last_fill_times = {}
        bot_mod._state.protective_stops = {}
        # Earlier trailing-stop tests leave a BTCUSD entry cooldown behind;
        # without clearing it the entry path returns before placing anything.
        bot_mod._state.cooldowns = {}
        bot_mod._state.position_adds = {}
        yield
        bot_mod._state.last_fill_times = {}
        bot_mod._state.protective_stops = {}
        bot_mod._state.cooldowns = {}
        bot_mod._state.position_adds = {}

    @pytest.fixture
    def ex(self):
        from src.exchange import AlpacaExchange
        ex = AsyncMock(spec=AlpacaExchange)
        ex.get_positions = AsyncMock(return_value=[])
        ex.create_order = AsyncMock(return_value={
            "id": "entry_1", "filled_avg_price": 50000.0, "filled_qty": 0.1,
            "commission": 5.0, "status": "filled",
        })
        ex.submit_protective_stop = AsyncMock(return_value={"id": "stop_1", "type": "stop_limit"})
        ex.cancel_order = AsyncMock(return_value=True)
        ex.get_account = AsyncMock(return_value={"equity": 10000.0, "cash": 10000.0, "portfolio_value": 10000.0})
        ex.load = AsyncMock()
        ex.close = AsyncMock()
        return ex

    @pytest.fixture
    def strategy(self):
        from src.strategies import TradingStrategy
        s = AsyncMock(spec=TradingStrategy)
        s.generate_trading_signal = AsyncMock(return_value={
            "action": "buy", "confidence": 0.75, "regime": "trending",
            "rsi": 55.0, "atr": 500.0, "features": {"hurst": 0.65, "atr": 500.0},
        })
        return s

    @pytest.fixture
    def rm(self):
        from src.risk import RiskManager
        rm = AsyncMock(spec=RiskManager)
        rm.update_account_status = AsyncMock(return_value={
            "status": "risk_ok", "equity": 10000.0, "cash": 10000.0,
            "portfolio_value": 10000.0, "drawdown_pct": -2.0, "daily_pnl": 100.0,
            "open_positions": 0, "current_exposure": 0.0,
        })
        rm.is_killswitch_active = MagicMock(return_value=False)
        rm.check_killswitch_conditions = AsyncMock(return_value=False)
        rm.check_trailing_stop = MagicMock(return_value="hold")
        rm.calculate_position_size = MagicMock(return_value=(0.1, "ok"))
        rm.check_and_reserve_exposure = AsyncMock(return_value=(1000.0, "ok"))
        rm.reserve_position_slot = AsyncMock(return_value=(True, "ok"))
        rm.release_position_slot = AsyncMock()
        rm.peak_prices = {}
        rm.record_fill_costs = MagicMock()
        return rm

    async def _buy(self, ex, strategy, rm):
        import pandas as pd
        import polars as pl
        ex.get_latest_bar = AsyncMock(return_value=pl.DataFrame({
            "t": [pd.Timestamp.now(tz="UTC")], "open": [50000.0], "high": [50100.0],
            "low": [49900.0], "close": [50050.0], "volume": [100.0], "vwap": [50025.0],
            "trade_count": [100],
        }))
        await process_signal_for_symbol(
            symbol="BTC/USD", current_price=50050.0, risk_manager=rm,
            strategy=strategy, ex=ex, positions=[], regime_flag=None, banned_symbols=set(),
        )

    @pytest.mark.asyncio
    async def test_stop_not_armed_when_flag_disabled(self, ex, strategy, rm):
        """Default OFF: no resting order, so existing behaviour is unchanged."""
        await self._buy(ex, strategy, rm)
        ex.submit_protective_stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_armed_at_stop_loss_pct_from_real_fill(self, ex, strategy, rm):
        with patch.object(settings, "PROTECTIVE_STOPS_ENABLED", True):
            await self._buy(ex, strategy, rm)

        import src.bot as bot_mod
        ex.submit_protective_stop.assert_called_once()
        kwargs = ex.submit_protective_stop.call_args.kwargs
        assert kwargs["symbol"] == "BTC/USD"
        # Covers exactly what the entry actually ordered (sizing multipliers
        # may shrink the requested size).
        assert kwargs["qty"] == pytest.approx(ex.create_order.call_args.kwargs["qty"])
        # 4% below the REAL fill (50000), not the signal-time price (50050).
        assert kwargs["stop_price"] == pytest.approx(50000.0 * (1 - settings.STOP_LOSS_PCT))
        assert kwargs["limit_price"] < kwargs["stop_price"]
        assert bot_mod._state.protective_stops["BTCUSD"] == "stop_1"

    @pytest.mark.asyncio
    async def test_stop_failure_does_not_undo_entry(self, ex, strategy, rm):
        """A rejected stop must not raise out of the entry path -- the fill
        already happened and the bot's polled exits still cover the position."""
        ex.submit_protective_stop = AsyncMock(side_effect=RuntimeError("insufficient qty"))
        with patch.object(settings, "PROTECTIVE_STOPS_ENABLED", True):
            await self._buy(ex, strategy, rm)
        ex.create_order.assert_called_once()  # the buy still went through

    @pytest.mark.asyncio
    async def test_stop_cancelled_before_committee_close(self, ex, strategy, rm):
        """A close must cancel the resting stop first, or a dangling sell
        lingers after the position is flat."""
        import src.bot as bot_mod
        bot_mod._state.protective_stops["BTCUSD"] = "stop_1"
        ex.get_positions = AsyncMock(return_value=[
            {"symbol": "BTC/USD", "qty": "0.1", "side": "long", "avg_entry_price": 50000.0, "market_value": 5000.0}
        ])
        strategy.generate_trading_signal = AsyncMock(return_value={
            "action": "close", "confidence": 1.0, "regime": "trending", "atr": 500.0,
        })
        import pandas as pd
        import polars as pl
        ex.get_latest_bar = AsyncMock(return_value=pl.DataFrame({
            "t": [pd.Timestamp.now(tz="UTC")], "open": [51000.0], "high": [51100.0],
            "low": [50900.0], "close": [51050.0], "volume": [100.0], "vwap": [51025.0],
            "trade_count": [100],
        }))

        await process_signal_for_symbol(
            symbol="BTC/USD", current_price=51050.0, risk_manager=rm, strategy=strategy,
            ex=ex, positions=[{"symbol": "BTC/USD", "qty": "0.1", "side": "long", "avg_entry_price": 50000.0}],
            regime_flag=None, banned_symbols=set(),
        )

        ex.cancel_order.assert_awaited_once_with("stop_1")
        assert "BTCUSD" not in bot_mod._state.protective_stops

    @pytest.mark.asyncio
    async def test_stop_records_decision_id_and_client_order_id_in_ledger(self, ex, strategy, rm):
        """The resting stop must land in the order ledger tagged with the
        entry's decision_id and its own client_order_id, so a crash can
        correlate the stop's later fill back to the position it protected."""
        import src.bot as bot_mod
        from src.db import get_recent_order_records

        async def _fake_committee(symbol, price, signal):
            return _deterministic_committee_result(action="buy", score=0.62)

        with patch.object(settings, "PROTECTIVE_STOPS_ENABLED", True), \
             patch("src.committee.committee.run_committee", new=_fake_committee):
            await self._buy(ex, strategy, rm)

        cid = ex.submit_protective_stop.call_args.kwargs["client_order_id"]
        # Deterministic from the entry's decision_id, so reconcile can recompute
        # it after a restart and match the stop's fill by client_order_id.
        assert cid == bot_mod._protstop_client_order_id("BTC/USD", "test-decision-id")
        assert cid.startswith("BTCUSD_ps_")
        assert len(cid) <= 48  # Alpaca's client_order_id limit
        records = get_recent_order_records("BTCUSD")
        stops = [r for r in records if r["type"] == "stop_limit"]
        assert len(stops) == 1
        assert stops[0]["decision_id"] == "test-decision-id"
        assert stops[0]["client_order_id"] == cid
        assert stops[0]["side"] == "sell"
        assert stops[0]["time_in_force"] == "gtc"

    @pytest.mark.asyncio
    async def test_stop_records_real_fill_price_and_filled_at(self, ex, strategy, rm):
        """If the exchange reports the stop already triggered (a fast gap), the
        ledger must store its real fill price/time -- that is the exit price a
        restart reconciles the snapshot against."""
        from src.db import get_recent_order_records
        ex.submit_protective_stop = AsyncMock(return_value={
            "id": "stop_filled", "type": "stop_limit", "status": "filled",
            "filled_avg_price": 48000.0, "filled_qty": 0.1, "qty": 0.1,
            "filled_at": "2026-10-07T12:00:00Z",
        })
        with patch.object(settings, "PROTECTIVE_STOPS_ENABLED", True):
            await self._buy(ex, strategy, rm)

        stops = [r for r in get_recent_order_records("BTCUSD") if r["type"] == "stop_limit"]
        assert stops[0]["filled_avg_price"] == pytest.approx(48000.0)
        assert stops[0]["filled_at"] is not None

    @pytest.mark.asyncio
    async def test_stop_cancelled_before_trailing_close(self, ex, strategy, rm):
        import src.bot as bot_mod
        bot_mod._state.protective_stops["BTCUSD"] = "stop_2"
        ex.get_positions = AsyncMock(return_value=[
            {"symbol": "BTC/USD", "qty": "0.1", "side": "long", "avg_entry_price": 50000.0, "market_value": 5400.0}
        ])
        rm.check_trailing_stop = MagicMock(return_value="close")
        import pandas as pd
        import polars as pl
        ex.get_latest_bar = AsyncMock(return_value=pl.DataFrame({
            "t": [pd.Timestamp.now(tz="UTC")], "open": [54000.0], "high": [54100.0],
            "low": [53900.0], "close": [54050.0], "volume": [100.0], "vwap": [54025.0],
            "trade_count": [100],
        }))

        await process_signal_for_symbol(
            symbol="BTC/USD", current_price=54050.0, risk_manager=rm, strategy=strategy,
            ex=ex, positions=[{"symbol": "BTC/USD", "qty": "0.1", "side": "long", "avg_entry_price": 50000.0}],
            regime_flag=None, banned_symbols=set(),
        )

        ex.cancel_order.assert_awaited_once_with("stop_2")
        assert "BTCUSD" not in bot_mod._state.protective_stops


class TestCommitteeErrorHandling:
    """Tests for committee error handling."""

    @pytest.mark.asyncio
    async def test_committee_single_brain_failure(self):
        """Test committee handles single brain failure gracefully."""
        
        signal = {
            "action": "buy",
            "regime": "trending",
            "rsi": 60.0,
            "atr": 500.0,
            "features": {"hurst": 0.65}
        }
        
        # Mock one brain to fail
        with patch("src.committee.committee.transformer_brain", side_effect=Exception("Transformer failed")):
            with patch("src.committee.committee.quant_brain", return_value=AsyncMock(return_value=BrainVote(
                name="quant", action="buy", confidence=0.7, weight=0.25, regime="trending", reason="test"
            ))):
                with patch("src.committee.committee.momentum_brain", return_value=AsyncMock(return_value=BrainVote(
                    name="momentum", action="buy", confidence=0.6, weight=0.2, regime="trending", reason="test"
                ))):
                    with patch("src.committee.committee.sentinel_brain", return_value=AsyncMock(return_value=BrainVote(
                        name="sentinel", action="hold", confidence=0.5, weight=0.05, regime="trending", reason="test"
                    ))):
                        with patch("src.committee.committee.llm_brain", return_value=AsyncMock(return_value=BrainVote(
                            name="llm", action="hold", confidence=0.5, weight=0.1, regime="trending", reason="test"
                        ))):
                            result = await run_committee("BTC/USD", 50000.0, signal)
                            
                            # Should still produce result despite one brain failing
                            assert result.action in ["buy", "sell", "hold", "stand_aside"]


class TestRiskManagerRaceConditions:
    """Tests for race conditions in risk manager."""

    @pytest.mark.asyncio
    async def test_concurrent_exposure_reservation(self):
        """Test concurrent exposure reservations don't exceed limit."""
        from src.exchange import AlpacaExchange
        from src.risk import RiskManager
        
        mock_exchange = AsyncMock(spec=AlpacaExchange)
        mock_exchange.get_account = AsyncMock(return_value={
            "equity": 10000.0, "cash": 10000.0, "portfolio_value": 10000.0
        })
        mock_exchange.get_positions = AsyncMock(return_value=[])
        
        risk = RiskManager(AlpacaExchange())
        risk.exchange = mock_exchange
        risk.peak_equity = 10000.0
        
        # Simulate concurrent exposure checks
        async def check_and_reserve():
            return await risk.check_and_reserve_exposure(2000.0)
        
        # Run 10 concurrent requests
        results = await asyncio.gather(*[check_and_reserve() for _ in range(10)])
        
        # Total approved should not exceed max portfolio value
        total_approved = sum(r[0] for r in results if r[1] == "ok")
        assert total_approved <= 5000.0  # ACCOUNT_BASE * MAX_PORTFOLIO_PCT (10000 * 0.5)


class TestGracefulShutdown:
    """Tests for graceful shutdown handling."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_background_tasks(self):
        """Verify shutdown cancels all registered background tasks."""
        import src.bot as bot_mod

        state = bot_mod.BotState()
        cancelled = []

        async def dummy_task():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        state.add_background_task(asyncio.create_task(dummy_task()))
        state.add_background_task(asyncio.create_task(dummy_task()))
        assert len(state._background_tasks) == 2

        # Let both tasks actually start (and park in sleep) before shutdown,
        # otherwise they're cancelled before their first step and the
        # CancelledError handler never runs.
        await asyncio.sleep(0.01)

        await state.shutdown()

        # Both tasks must have observed cancellation and been drained.
        assert len(cancelled) == 2
        assert not state._background_tasks


class TestSameRegimePositionCounting:
    """Tests for _count_same_regime_open_positions, the regime-clustering
    proxy used to cap correlated exposure (real return-series correlation
    isn't wired up anywhere live -- see calculate_position_size's unused
    returns_matrix parameter)."""

    def _make_strategy(self, regime_by_symbol):
        import time
        strat = MagicMock()
        strat._regime_cache = {
            sym: (time.monotonic(), {"regime": regime})
            for sym, regime in regime_by_symbol.items()
        }
        return strat

    def test_counts_matching_regime_only(self):
        from src.bot import _count_same_regime_open_positions
        strat = self._make_strategy({
            "BTC/USD": "high_volatility",
            "ETH/USD": "high_volatility",
            "SOL/USD": "sideways",
            "DOGE/USD": "high_volatility",
        })
        positions = [
            {"symbol": "BTCUSD"}, {"symbol": "ETHUSD"}, {"symbol": "SOLUSD"},
        ]
        count = _count_same_regime_open_positions("DOGE/USD", positions, strat)
        assert count == 2  # BTC + ETH, not SOL

    def test_excludes_candidate_itself(self):
        from src.bot import _count_same_regime_open_positions
        strat = self._make_strategy({"BTC/USD": "trending", "ETH/USD": "trending"})
        positions = [{"symbol": "BTCUSD"}, {"symbol": "ETHUSD"}]
        # BTC/USD is itself already open and is the "candidate" here -- must
        # not count itself, only the other trending position (ETH).
        assert _count_same_regime_open_positions("BTC/USD", positions, strat) == 1

    def test_no_cached_regime_for_candidate_returns_zero(self):
        from src.bot import _count_same_regime_open_positions
        strat = self._make_strategy({"BTC/USD": "trending"})
        positions = [{"symbol": "BTCUSD"}]
        # DOGE/USD has no cache entry -> candidate_regime is None -> can't
        # meaningfully compare, must return 0 rather than error or guess.
        assert _count_same_regime_open_positions("DOGE/USD", positions, strat) == 0

    def test_empty_positions_or_missing_strategy_is_safe(self):
        from src.bot import _count_same_regime_open_positions
        strat = self._make_strategy({"BTC/USD": "trending"})
        assert _count_same_regime_open_positions("BTC/USD", [], strat) == 0
        assert _count_same_regime_open_positions("BTC/USD", [{"symbol": "BTCUSD"}], None) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
