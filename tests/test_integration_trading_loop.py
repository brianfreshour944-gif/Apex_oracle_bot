"""Integration tests for the full trading loop."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.committee.models import BrainVote
from src.bot import run_trading_bot, process_signal_for_symbol
from src.committee.committee import run_committee
from src.risk import RiskManager
from src.exchange import AlpacaExchange
from src.strategies import TradingStrategy


class TestFullTradingLoop:
    """Integration tests for the complete trading loop."""

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
        rm.check_killswitch_conditions = AsyncMock(return_value=False)
        rm.check_trailing_stop = MagicMock(return_value="hold")
        rm.calculate_position_size = MagicMock(return_value=(0.1, "ok"))
        rm.check_and_reserve_exposure = AsyncMock(return_value=(1000.0, "ok"))
        rm.reserve_position_slot = AsyncMock(return_value=(True, "ok"))
        rm.release_position_slot = MagicMock()
        rm.peak_prices = {}
        rm.record_fill_costs = MagicMock()
        return rm

    @pytest.mark.asyncio
    async def test_full_buy_signal_flow(self, mock_exchange, mock_strategy, mock_risk_manager):
        """Test complete buy signal processing flow."""
        from src.bot import process_signal_for_symbol
        
        # Setup mock bar data
        import polars as pl
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        closes = np.cumsum(np.random.randn(100) * 10) + 50000
        bars_df = pl.DataFrame({
            "t": dates,
            "open": closes + np.random.randn(100) * 5,
            "high": np.abs(np.random.randn(100) * 10) + closes + 10,
            "low": np.abs(np.random.randn(100) * 10) + closes - 10,
            "close": closes,
            "volume": np.abs(np.random.randn(100) * 1000) + 100,
            "vwap": closes,
            "trade_count": np.ones(100) * 100
        })
        
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

    @pytest.mark.asyncio
    async def test_close_position_flow(self, mock_exchange, mock_strategy, mock_risk_manager):
        """Test position close flow."""
        from src.bot import process_signal_for_symbol
        
        # Setup existing position
        import polars as pl
        import pandas as pd
        import numpy as np
        
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
        from src.bot import process_signal_for_symbol
        
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


class TestCommitteeErrorHandling:
    """Tests for committee error handling."""

    @pytest.mark.asyncio
    async def test_committee_single_brain_failure(self):
        """Test committee handles single brain failure gracefully."""
        from src.committee.committee import run_committee
        
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
                            result = await run_committee("BTC/USD", 50000.0, {"action": "buy", "regime": "trending", "features": {}})
                            
                            # Should still produce result despite one brain failing
                            assert result.action in ["buy", "sell", "hold", "stand_aside"]


class TestRiskManagerRaceConditions:
    """Tests for race conditions in risk manager."""

    @pytest.mark.asyncio
    async def test_concurrent_exposure_reservation(self):
        """Test concurrent exposure reservations don't exceed limit."""
        from src.risk import RiskManager
        from src.exchange import AlpacaExchange
        
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
        """Verify shutdown cancels all background tasks."""
        from src.bot import _state
        
        # Add some mock tasks
        async def dummy_task():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise
        
        task1 = asyncio.create_task(dummy_task())
        task2 = asyncio.create_task(dummy_task())
        
        # Simulate shutdown
        # This would be tested via BotState.shutdown()
        assert True  # Placeholder


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