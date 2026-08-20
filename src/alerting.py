"""
Real Alerting System — Step 7 of Foundation Hardening.

Provides strong "tell me when something's actually wrong" instrumentation:
- Exposure saturation alerts
- Repeated veto alerts
- Drawdown approaching killswitch
- Model load failures
- Abnormal trade frequency (churn detection)
- Circuit breaker trips
- Data integrity failures

Run: python -m src.alerting
"""

import asyncio
import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from src.config import settings
from src.logging_config import get_logger
from src.alerts import send_alert

logger = get_logger("alerting")


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCategory(Enum):
    EXPOSURE = "exposure"
    VETO = "veto"
    DRAWDOWN = "drawdown"
    MODEL = "model"
    CHURN = "churn"
    CIRCUIT_BREAKER = "circuit_breaker"
    DATA_INTEGRITY = "data_integrity"
    KILLSWITCH = "killswitch"
    SYSTEM = "system"


@dataclass
class Alert:
    category: AlertCategory
    severity: AlertSeverity
    title: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    key: str = ""  # For deduplication


class AlertingEngine:
    """
    Central alerting engine with:
    - Per-category cooldowns
    - Escalation paths
    - Structured alert logging
    - Prometheus metrics integration
    """
    
    def __init__(self):
        self._alert_history: deque = deque(maxlen=1000)
        self._category_cooldowns: Dict[AlertCategory, float] = {}
        self._alert_counts: Dict[str, int] = defaultdict(int)
        self._last_alert_time: Dict[str, float] = {}
        
        # Default cooldowns per category (seconds)
        self._default_cooldowns = {
            AlertCategory.EXPOSURE: 300,
            AlertCategory.VETO: 60,
            AlertCategory.DRAWDOWN: 300,
            AlertCategory.MODEL: 600,
            AlertCategory.CHURN: 300,
            AlertCategory.CIRCUIT_BREAKER: 60,
            AlertCategory.DATA_INTEGRITY: 300,
            AlertCategory.KILLSWITCH: 0,  # No cooldown for killswitch
            AlertCategory.SYSTEM: 300,
        }
        
        # Escalation thresholds
        self._escalation_thresholds = {
            AlertCategory.VETO: {"count": 10, "window_sec": 300},  # 10 vetoes in 5 min
            AlertCategory.CHURN: {"count": 20, "window_sec": 300},  # 20 trades in 5 min
            AlertCategory.EXPOSURE: {"pct": 0.9},  # 90% of cap
            AlertCategory.DRAWDOWN: {"pct": 0.8},  # 80% of max drawdown
        }
    
    def _get_cooldown(self, category: AlertCategory) -> float:
        return self._category_cooldowns.get(category, self._default_cooldowns.get(category, 300))
    
    def _in_cooldown(self, alert_key: str, category: AlertCategory) -> bool:
        now = time.monotonic()
        cooldown = self._get_cooldown(category)
        last = self._last_alert_time.get(alert_key, float("-inf"))
        if now - last < cooldown:
            return True
        self._last_alert_time[alert_key] = now
        return False
    
    def _should_escalate(self, category: AlertCategory, details: Dict[str, Any]) -> bool:
        """Check if alert should trigger escalation."""
        if category == AlertCategory.VETO:
            threshold = self._escalation_thresholds.get(category, {})
            count = details.get("veto_count_5min", 0)
            return count >= threshold.get("count", 10)
        elif category == AlertCategory.CHURN:
            threshold = self._escalation_thresholds.get(category, {})
            count = details.get("trade_count_5min", 0)
            return count >= threshold.get("count", 20)
        elif category == AlertCategory.EXPOSURE:
            threshold = self._escalation_thresholds.get(category, {})
            pct = details.get("exposure_pct", 0)
            return pct >= threshold.get("pct", 0.9)
        elif category == AlertCategory.DRAWDOWN:
            threshold = self._escalation_thresholds.get(category, {})
            pct = details.get("drawdown_pct_of_max", 0)
            return pct >= threshold.get("pct", 0.8)
        return False
    
    async def fire(
        self,
        category: AlertCategory,
        severity: AlertSeverity,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        key: Optional[str] = None,
    ) -> bool:
        """
        Fire an alert. Returns True if sent, False if suppressed.
        """
        alert_key = key or f"{category.value}:{title}"
        
        if self._in_cooldown(alert_key, category):
            logger.debug(f"Alert suppressed (cooldown): {alert_key}")
            return False
        
        alert = Alert(
            category=category,
            severity=severity,
            title=title,
            message=message,
            details=details or {},
            key=alert_key,
        )
        
        # Track for escalation
        self._alert_counts[alert_key] += 1
        self._alert_history.append(alert)
        
        # Log locally
        log_method = getattr(logger, severity.value.lower(), logger.info)
        log_method(f"ALERT [{category.value}/{severity.value}] {title}: {message}")
        
        # Send via Telegram/email
        full_message = self._format_alert(alert)
        await send_alert(full_message, key=alert_key)
        
        # Check escalation
        if self._should_escalate(category, details or {}):
            await self._escalate(alert)
        
        return True
    
    def _format_alert(self, alert: Alert) -> str:
        """Format alert for Telegram/email."""
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
        }
        emoji = severity_emoji.get(alert.severity, "")
        
        lines = [
            f"{emoji} <b>{alert.title}</b>",
            f"Category: {alert.category.value}",
            f"Severity: {alert.severity.value.upper()}",
            f"Time: {alert.timestamp}",
            "",
            alert.message,
        ]
        
        if alert.details:
            lines.append("")
            lines.append("<b>Details:</b>")
            for k, v in alert.details.items():
                lines.append(f"  {k}: {v}")
        
        return "\n".join(lines)
    
    async def _escalate(self, alert: Alert):
        """Send escalation alert."""
        escalation_msg = (
            f"🚨 <b>ESCALATION: {alert.title}</b>\n"
            f"Category: {alert.category.value}\n"
            f"Alert count: {self._alert_counts[alert.key]}\n"
            f"Original: {alert.message}\n"
            f"Time: {alert.timestamp}"
        )
        await send_alert(escalation_msg, key=f"escalation:{alert.key}")
        logger.critical(f"ESCALATION fired for {alert.key}")
    
    # ─── Convenience Methods ───
    
    async def alert_exposure_saturation(self, current_exposure: float, max_exposure: float, pct: float):
        """Alert when portfolio exposure approaches cap."""
        severity = AlertSeverity.CRITICAL if pct >= 0.95 else AlertSeverity.WARNING
        await self.fire(
            category=AlertCategory.EXPOSURE,
            severity=severity,
            title="Portfolio Exposure Saturation",
            message=f"Exposure at {pct*100:.1f}% of cap (${current_exposure:,.0f}/${max_exposure:,.0f})",
            details={"exposure_pct": pct, "current": current_exposure, "max": max_exposure},
            key="exposure_saturation",
        )
    
    async def alert_repeated_vetoes(self, veto_count: int, window_sec: int, regime: str, reasons: list):
        """Alert when vetoes are frequent."""
        await self.fire(
            category=AlertCategory.VETO,
            severity=AlertSeverity.WARNING,
            title="Repeated Vetoes Detected",
            message=f"{veto_count} vetoes in {window_sec}s (regime: {regime})",
            details={
                "veto_count_5min": veto_count,
                "window_sec": window_sec,
                "regime": regime,
                "reasons": reasons[:5],  # Top 5
            },
            key="repeated_vetoes",
        )
    
    async def alert_drawdown_approaching_killswitch(self, current_drawdown: float, max_drawdown: float, pct_of_max: float):
        """Alert when drawdown approaches killswitch."""
        severity = AlertSeverity.CRITICAL if pct_of_max >= 0.9 else AlertSeverity.WARNING
        await self.fire(
            category=AlertCategory.DRAWDOWN,
            severity=severity,
            title="Drawdown Approaching Killswitch",
            message=f"Drawdown at {current_drawdown:.2f}% ({pct_of_max*100:.1f}% of killswitch at {max_drawdown}%)",
            details={
                "drawdown_pct": current_drawdown,
                "max_drawdown": max_drawdown,
                "drawdown_pct_of_max": pct_of_max,
            },
            key="drawdown_killswitch",
        )
    
    async def alert_model_load_failure(self, model_name: str, error: str, fallback: str = ""):
        """Alert when a model fails to load."""
        await self.fire(
            category=AlertCategory.MODEL,
            severity=AlertSeverity.CRITICAL,
            title=f"Model Load Failure: {model_name}",
            message=f"Failed to load {model_name}: {error}" + (f" | Fallback: {fallback}" if fallback else ""),
            details={"model": model_name, "error": error, "fallback": fallback},
            key=f"model_load:{model_name}",
        )
    
    async def alert_churn_detected(self, trade_count: int, window_sec: int, symbols: list, pnl_per_trade: float = 0):
        """Alert on abnormal trade frequency (churn)."""
        await self.fire(
            category=AlertCategory.CHURN,
            severity=AlertSeverity.WARNING,
            title="Trade Churn Detected",
            message=f"{trade_count} trades in {window_sec}s across {len(symbols)} symbols (avg PnL/trade: ${pnl_per_trade:.2f})",
            details={
                "trade_count_5min": trade_count,
                "window_sec": window_sec,
                "symbols": symbols,
                "avg_pnl_per_trade": pnl_per_trade,
            },
            key="churn_detected",
        )
    
    async def alert_circuit_breaker_trip(self, circuit_name: str, failure_count: int, reason: str):
        """Alert when circuit breaker trips."""
        await self.fire(
            category=AlertCategory.CIRCUIT_BREAKER,
            severity=AlertSeverity.CRITICAL,
            title=f"Circuit Breaker Tripped: {circuit_name}",
            message=f"Circuit '{circuit_name}' opened after {failure_count} failures: {reason}",
            details={"circuit": circuit_name, "failures": failure_count, "reason": reason},
            key=f"circuit_breaker:{circuit_name}",
        )
    
    async def alert_data_integrity_failure(self, check_name: str, mismatches: int, details: Dict[str, Any]):
        """Alert on data integrity verification failure."""
        await self.fire(
            category=AlertCategory.DATA_INTEGRITY,
            severity=AlertSeverity.CRITICAL,
            title=f"Data Integrity Failure: {check_name}",
            message=f"{mismatches} data mismatches found in {check_name}",
            details={"check": check_name, "mismatches": mismatches, **details},
            key=f"data_integrity:{check_name}",
        )
    
    async def alert_killswitch_activated(self, reason: str, action: str, details: Dict[str, Any]):
        """Alert when killswitch activates (NO cooldown)."""
        await self.fire(
            category=AlertCategory.KILLSWITCH,
            severity=AlertSeverity.CRITICAL,
            title="KILLSWITCH ACTIVATED",
            message=f"Reason: {reason} | Action: {action}",
            details={"reason": reason, "action": action, **details},
            key="killswitch_activated",
        )
    
    async def alert_system_health(self, component: str, status: str, details: Dict[str, Any]):
        """General system health alert."""
        severity = AlertSeverity.CRITICAL if status in ("down", "failed") else AlertSeverity.WARNING
        await self.fire(
            category=AlertCategory.SYSTEM,
            severity=severity,
            title=f"System Health: {component}",
            message=f"{component} is {status}",
            details={"component": component, "status": status, **details},
            key=f"system:{component}",
        )
    
    def get_recent_alerts(self, limit: int = 50) -> list:
        """Get recent alert history."""
        return [
            {
                "timestamp": a.timestamp,
                "category": a.category.value,
                "severity": a.severity.value,
                "title": a.title,
                "message": a.message,
                "details": a.details,
            }
            for a in list(self._alert_history)[-limit:]
        ]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        by_category = defaultdict(int)
        by_severity = defaultdict(int)
        for alert in self._alert_history:
            by_category[alert.category.value] += 1
            by_severity[alert.severity.value] += 1
        return {
            "total_alerts": len(self._alert_history),
            "by_category": dict(by_category),
            "by_severity": dict(by_severity),
            "top_keys": dict(sorted(self._alert_counts.items(), key=lambda x: -x[1])[:10]),
        }


# Global singleton
_alerting_engine: Optional[AlertingEngine] = None


def get_alerting_engine() -> AlertingEngine:
    """Get the global alerting engine instance."""
    global _alerting_engine
    if _alerting_engine is None:
        _alerting_engine = AlertingEngine()
    return _alerting_engine


def reset_alerting_engine():
    """Reset the alerting engine (for testing)."""
    global _alerting_engine
    _alerting_engine = None


# Convenience functions for common alerts
async def alert_exposure_saturation(current: float, max_exp: float, pct: float):
    await get_alerting_engine().alert_exposure_saturation(current, max_exp, pct)

async def alert_repeated_vetoes(count: int, window: int, regime: str, reasons: list):
    await get_alerting_engine().alert_repeated_vetoes(count, window, regime, reasons)

async def alert_drawdown_approaching(current: float, max_dd: float, pct: float):
    await get_alerting_engine().alert_drawdown_approaching_killswitch(current, max_dd, pct)

async def alert_model_load_failure(model: str, error: str, fallback: str = ""):
    await get_alerting_engine().alert_model_load_failure(model, error, fallback)

async def alert_churn_detected(count: int, window: int, symbols: list, avg_pnl: float = 0):
    await get_alerting_engine().alert_churn_detected(count, window, symbols, avg_pnl)

async def alert_circuit_breaker_trip(name: str, failures: int, reason: str):
    await get_alerting_engine().alert_circuit_breaker_trip(name, failures, reason)

async def alert_data_integrity_failure(check: str, mismatches: int, details: Dict[str, Any]):
    await get_alerting_engine().alert_data_integrity_failure(check, mismatches, details)

async def alert_killswitch_activated(reason: str, action: str, details: Dict[str, Any]):
    await get_alerting_engine().alert_killswitch_activated(reason, action, details)


# ─── Background Monitoring Task ───
async def monitoring_loop(
    get_risk_status: Callable,
    get_committee_stats: Callable,
    get_model_status: Callable,
    interval_sec: int = 30,
):
    """
    Background task that monitors system health and fires alerts.
    
    Args:
        get_risk_status: Callable returning risk status dict
        get_committee_stats: Callable returning committee stats dict  
        get_model_status: Callable returning model status dict
        interval_sec: Check interval
    """
    engine = get_alerting_engine()
    veto_history = deque(maxlen=100)
    trade_history = deque(maxlen=1000)
    
    logger.info("Starting alerting monitoring loop")
    
    while True:
        try:
            # 1. Check exposure
            risk_status = await get_risk_status()
            if risk_status.get("status") == "risk_ok":
                exposure = risk_status.get("current_exposure", 0)
                max_exp = risk_status.get("max_portfolio_value", settings.ACCOUNT_BASE * settings.MAX_PORTFOLIO_PCT)
                if max_exp > 0:
                    pct = exposure / max_exp
                    if pct >= 0.8:
                        await engine.alert_exposure_saturation(exposure, max_exp, pct)
            
            # 2. Check drawdown
            if risk_status.get("status") == "risk_ok":
                drawdown = risk_status.get("drawdown_pct", 0)
                max_dd = abs(settings.MAX_DRAWDOWN_STOP)
                if max_dd > 0 and drawdown < 0:
                    pct_of_max = abs(drawdown) / max_dd
                    if pct_of_max >= 0.7:
                        await engine.alert_drawdown_approaching(drawdown, -max_dd, pct_of_max)
            
            # 3. Check committee vetoes
            committee_stats = await get_committee_stats()
            recent_vetoes = committee_stats.get("recent_vetoes", [])
            if len(recent_vetoes) >= 5:
                reasons = [v.get("reason", "unknown") for v in recent_vetoes]
                regime = committee_stats.get("current_regime", "unknown")
                await engine.alert_repeated_vetoes(len(recent_vetoes), 300, regime, reasons)
            
            # 4. Check churn
            recent_trades = committee_stats.get("recent_trades", [])
            if len(recent_trades) >= 15:
                symbols = list(set(t.get("symbol") for t in recent_trades))
                avg_pnl = sum(t.get("pnl", 0) for t in recent_trades) / len(recent_trades)
                await engine.alert_churn_detected(len(recent_trades), 300, symbols, avg_pnl)
            
            # 5. Check model status
            model_status = await get_model_status()
            for model_name, status in model_status.items():
                if status.get("status") == "failed":
                    await engine.alert_model_load_failure(
                        model_name, 
                        status.get("error", "unknown"),
                        status.get("fallback", "")
                    )
            
        except Exception as e:
            logger.error(f"Alerting monitoring loop error: {e}")
        
        await asyncio.sleep(interval_sec)


if __name__ == "__main__":
    # Test the alerting engine
    async def test():
        engine = get_alerting_engine()
        
        print("Testing alerting engine...")
        
        # Test exposure alert
        await engine.alert_exposure_saturation(4500, 5000, 0.9)
        
        # Test veto alert
        await engine.alert_repeated_vetoes(12, 300, "sideways", ["low confidence", "sentinel veto"])
        
        # Test drawdown alert
        await engine.alert_drawdown_approaching(-8.0, -10.0, 0.8)
        
        # Test model failure
        await engine.alert_model_load_failure("DecisionTransformer", "File not found", "Using baseline")
        
        # Test churn
        await engine.alert_churn_detected(25, 300, ["BTC/USD", "ETH/USD"], -5.0)
        
        # Test circuit breaker
        await engine.alert_circuit_breaker_trip("AlpacaAPI", 5, "Timeout")
        
        # Test data integrity
        await engine.alert_data_integrity_failure("fill_price_check", 3, {"price_mismatches": 2, "commission_mismatches": 1})
        
        # Test killswitch
        await engine.alert_killswitch_activated("max_drawdown_exceeded", "liquidate_all", {"drawdown_pct": -10.5})
        
        print("\nAlert stats:", engine.get_alert_stats())
        print("Done!")
    
    asyncio.run(test())