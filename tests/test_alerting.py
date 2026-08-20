"""
Tests for Alerting Engine — Step 7 of Foundation Hardening.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.alerting import (
    AlertingEngine,
    AlertCategory,
    AlertSeverity,
    get_alerting_engine,
    reset_alerting_engine,
)


class TestAlertingEngine:
    """Tests for the AlertingEngine class."""

    @pytest.fixture
    def engine(self):
        reset_alerting_engine()
        return AlertingEngine()

    @pytest.mark.asyncio
    async def test_fire_alert_returns_true_first_time(self, engine):
        """First alert should be sent."""
        result = await engine.fire(
            category=AlertCategory.EXPOSURE,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message",
            key="test_alert",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_fire_alert_returns_false_on_cooldown(self, engine):
        """Second alert within cooldown should be suppressed."""
        await engine.fire(
            category=AlertCategory.EXPOSURE,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message",
            key="test_alert",
        )
        
        result = await engine.fire(
            category=AlertCategory.EXPOSURE,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message",
            key="test_alert",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_different_keys_not_suppressed(self, engine):
        """Different alert keys should not suppress each other."""
        await engine.fire(
            category=AlertCategory.EXPOSURE,
            severity=AlertSeverity.WARNING,
            title="Alert 1",
            message="Message 1",
            key="alert_1",
        )
        
        result = await engine.fire(
            category=AlertCategory.EXPOSURE,
            severity=AlertSeverity.WARNING,
            title="Alert 2",
            message="Message 2",
            key="alert_2",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_alert_stored_in_history(self, engine):
        """Alerts should be stored in history."""
        await engine.fire(
            category=AlertCategory.VETO,
            severity=AlertSeverity.WARNING,
            title="Veto Alert",
            message="Veto message",
        )
        
        history = engine.get_recent_alerts()
        assert len(history) == 1
        assert history[0]["title"] == "Veto Alert"
        assert history[0]["category"] == "veto"

    @pytest.mark.asyncio
    async def test_history_limited_to_maxlen(self, engine):
        """History should be limited to maxlen."""
        # Fire 1100 alerts (maxlen is 1000)
        for i in range(1100):
            await engine.fire(
                category=AlertCategory.SYSTEM,
                severity=AlertSeverity.INFO,
                title=f"Alert {i}",
                message=f"Message {i}",
                key=f"alert_{i}",
            )
        
        history = engine.get_recent_alerts(limit=1000)
        assert len(history) == 1000

    @pytest.mark.asyncio
    async def test_convenience_methods(self, engine):
        """Test convenience alert methods."""
        with patch("src.alerting.send_alert", new_callable=AsyncMock) as mock_send:
            await engine.alert_exposure_saturation(4500, 5000, 0.9)
            # May trigger escalation since 90% >= threshold
            assert mock_send.call_count >= 1
            
            await engine.alert_repeated_vetoes(10, 300, "sideways", ["reason1", "reason2"])
            # May trigger escalation since 10 >= threshold
            assert mock_send.call_count >= 2
            
            await engine.alert_drawdown_approaching_killswitch(-8.0, -10.0, 0.8)
            assert mock_send.call_count >= 3
            
            await engine.alert_model_load_failure("DT", "File not found", "baseline")
            assert mock_send.call_count >= 4
            
            await engine.alert_churn_detected(20, 300, ["BTC/USD"], -5.0)
            assert mock_send.call_count >= 5
            
            await engine.alert_circuit_breaker_trip("API", 5, "Timeout")
            assert mock_send.call_count >= 6
            
            await engine.alert_data_integrity_failure("price_check", 3, {"mismatches": 3})
            assert mock_send.call_count >= 7
            
            await engine.alert_killswitch_activated("drawdown", "liquidate", {"dd": -10})
            assert mock_send.call_count >= 8

    @pytest.mark.asyncio
    async def test_escalation_on_veto_threshold(self, engine):
        """Test escalation when veto count exceeds threshold."""
        with patch("src.alerting.send_alert", new_callable=AsyncMock) as mock_send:
            # Fire 10 veto alerts (threshold is 10)
            for i in range(10):
                await engine.fire(
                    category=AlertCategory.VETO,
                    severity=AlertSeverity.WARNING,
                    title="Veto",
                    message=f"Veto {i}",
                    details={"veto_count_5min": i + 1},
                    key=f"veto_{i}",
                )
            
            # Should have sent escalation (10 regular + 1 escalation)
            assert mock_send.call_count >= 11

    def test_get_alert_stats(self, engine):
        """Test alert statistics."""
        import asyncio
        
        async def _fire_some():
            await engine.fire(AlertCategory.EXPOSURE, AlertSeverity.WARNING, "A", "a", key="a")
            await engine.fire(AlertCategory.EXPOSURE, AlertSeverity.CRITICAL, "B", "b", key="b")
            await engine.fire(AlertCategory.VETO, AlertSeverity.WARNING, "C", "c", key="c")
        
        asyncio.run(_fire_some())
        
        stats = engine.get_alert_stats()
        assert stats["total_alerts"] == 3
        assert stats["by_category"]["exposure"] == 2
        assert stats["by_category"]["veto"] == 1
        assert stats["by_severity"]["warning"] == 2
        assert stats["by_severity"]["critical"] == 1


class TestGlobalEngine:
    """Test global engine singleton."""

    def test_get_returns_same_instance(self):
        reset_alerting_engine()
        engine1 = get_alerting_engine()
        engine2 = get_alerting_engine()
        assert engine1 is engine2

    def test_reset_creates_new_instance(self):
        reset_alerting_engine()
        engine1 = get_alerting_engine()
        reset_alerting_engine()
        engine2 = get_alerting_engine()
        assert engine1 is not engine2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])