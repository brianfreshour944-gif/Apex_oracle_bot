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
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.alerts import send_alert
from src.config import settings
from src.logging_config import get_logger

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
    PERFORMANCE_DECAY = "performance_decay"
    FEATURE_DRIFT = "feature_drift"
    STALE_PRICE = "stale_price"
    DUPLICATE_ORDER = "duplicate_order"
    POSITION_DESYNC = "position_desync"


@dataclass
class Alert:
    category: AlertCategory
    severity: AlertSeverity
    title: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    key: str = ""  # For deduplication


class AlertingEngine:
    """
    Central alerting engine with:
    - Per-category cooldowns
    - Escalation paths
    - Structured alert logging
    - Prometheus metrics integration
    """
    
    def __init__(self, risk_manager: Any | None = None, exchange: Any | None = None):
        # Optional refs used by run_monitoring_cycle() for periodic exposure/
        # drawdown checks. Alerts can still be fired manually via the
        # alert_*() convenience methods without these being set.
        self.risk_manager = risk_manager
        self.exchange = exchange
        self._alert_history: deque = deque(maxlen=1000)
        self._category_cooldowns: dict[AlertCategory, float] = {}
        self._alert_counts: dict[str, int] = defaultdict(int)
        self._last_alert_time: dict[str, float] = {}
        
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
            AlertCategory.PERFORMANCE_DECAY: 3600,  # slow-moving signal, hourly is plenty
            AlertCategory.FEATURE_DRIFT: 3600,
            AlertCategory.STALE_PRICE: 300,
            AlertCategory.DUPLICATE_ORDER: 0,  # never suppress -- always investigate
            AlertCategory.POSITION_DESYNC: 300,
        }
        # run_monitoring_cycle() gates its own (expensive, DB/state-file-backed)
        # decay and drift checks on top of the per-alert cooldowns above, since
        # those checks themselves are only worth running periodically, not on
        # every ~30s monitoring tick.
        self._last_decay_check: float = 0.0
        self._last_drift_check: float = 0.0
        self._slow_check_interval_sec: float = 1800.0  # 30 min
        
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
        return False

    def _record_sent(self, alert_key: str):
        now = time.monotonic()
        self._last_alert_time[alert_key] = now
        # Prune _last_alert_time entries older than 2x the max cooldown
        # (default max is 600s for MODEL) to prevent unbounded growth.
        max_cooldown = max(self._default_cooldowns.values()) if self._default_cooldowns else 600.0
        cutoff = now - max_cooldown * 2
        expired = [k for k, v in self._last_alert_time.items() if v < cutoff]
        for k in expired:
            del self._last_alert_time[k]
    
    def _should_escalate(self, category: AlertCategory, details: dict[str, Any]) -> bool:
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
        details: dict[str, Any] | None = None,
        key: str | None = None,
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
        self._record_sent(alert_key)
        
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
    
    async def alert_data_integrity_failure(self, check_name: str, mismatches: int, details: dict[str, Any]):
        """Alert on data integrity verification failure."""
        await self.fire(
            category=AlertCategory.DATA_INTEGRITY,
            severity=AlertSeverity.CRITICAL,
            title=f"Data Integrity Failure: {check_name}",
            message=f"{mismatches} data mismatches found in {check_name}",
            details={"check": check_name, "mismatches": mismatches, **details},
            key=f"data_integrity:{check_name}",
        )
    
    async def alert_killswitch_activated(self, reason: str, action: str, details: dict[str, Any]):
        """Alert when killswitch activates (NO cooldown)."""
        await self.fire(
            category=AlertCategory.KILLSWITCH,
            severity=AlertSeverity.CRITICAL,
            title="KILLSWITCH ACTIVATED",
            message=f"Reason: {reason} | Action: {action}",
            details={"reason": reason, "action": action, **details},
            key="killswitch_activated",
        )
    
    async def alert_system_health(self, component: str, status: str, details: dict[str, Any]):
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
    
    async def check_brain_failure_alert(self, brain_name: str, error: str) -> None:
        """Alert when a committee brain raises during evaluation."""
        await self.alert_system_health(brain_name, "failed", {"error": error})

    async def alert_performance_decay(self, decay_alert: dict[str, Any]) -> None:
        """Alert on a strategy/regime performance-decay signal (Sharpe/win-rate/
        profit-factor dropping below threshold). Source: src.performance_tracker
        .get_decay_alerts() -- computed since inception but never wired to any
        consumer until 2026-09-20 (see ADVERSARIAL_AUDIT_2026-09-20.md).

        Alert-only for now: this does NOT pause trading or resize positions.
        With likely <30 real closed trades at time of writing, Sharpe/win-rate
        computed on a small sample is noise-prone -- auto-acting on it before
        the sample size supports the statistic would be worse than not
        checking at all. Escalate to automatic risk-reduction only once you've
        watched this fire on real data and trust it isn't a false alarm.
        """
        pair = decay_alert.get("pair", "unknown")
        alert_type = decay_alert.get("type", "decay")
        val = decay_alert.get("sharpe", decay_alert.get("win_rate", decay_alert.get("profit_factor", 0)))
        threshold = decay_alert.get("threshold", 0)
        trades = decay_alert.get("trades", 0)
        await self.fire(
            category=AlertCategory.PERFORMANCE_DECAY,
            severity=AlertSeverity.WARNING,
            title=f"Performance Decay: {pair}",
            message=f"{alert_type} for {pair}: {val:.3f} < threshold {threshold:.3f} (n={trades} trades, 30d window)",
            details=decay_alert,
            key=f"performance_decay:{pair}:{alert_type}",
        )

    async def alert_feature_drift(self, drift_alert: dict[str, Any]) -> None:
        """Alert on feature-distribution drift (PSI/KS vs. a 30d reference
        window). Source: src.feature_drift_monitor.get_feature_drift_alerts()
        -- PSI/KS were computed since inception but the report/alert path had
        zero callers anywhere in the codebase until 2026-09-20 (see
        ADVERSARIAL_AUDIT_2026-09-20.md). Alert-only -- does not change
        trading behavior.
        """
        feature = drift_alert.get("feature", "unknown")
        severity_str = drift_alert.get("severity", "low")
        severity = AlertSeverity.CRITICAL if severity_str == "high" else AlertSeverity.WARNING
        psi = drift_alert.get("psi", 0.0)
        ks_pvalue = drift_alert.get("ks_pvalue", 1.0)
        await self.fire(
            category=AlertCategory.FEATURE_DRIFT,
            severity=severity,
            title=f"Feature Drift: {feature}",
            message=f"{feature} drift={severity_str} (PSI={psi:.3f}, KS p={ks_pvalue:.4f}, "
                    f"mean_shift={drift_alert.get('mean_shift', 0):.4f})",
            details=drift_alert,
            key=f"feature_drift:{feature}:{severity_str}",
        )

    async def alert_stale_price(self, symbol: str, age_sec: float, max_age_sec: float) -> None:
        """Alert when the last fetched bar/quote for a symbol is older than
        expected -- a stale price silently feeding position sizing / stop
        checks was flagged as a missing alert in ADVERSARIAL_AUDIT_2026-09-20.md
        (§16, ranked highest-priority of the missing alerts: hard to notice,
        can produce a bad fill or a stop that never fires)."""
        await self.fire(
            category=AlertCategory.STALE_PRICE,
            severity=AlertSeverity.WARNING,
            title=f"Stale Price: {symbol}",
            message=f"{symbol} last price is {age_sec:.0f}s old (max expected: {max_age_sec:.0f}s)",
            details={"symbol": symbol, "age_sec": age_sec, "max_age_sec": max_age_sec},
            key=f"stale_price:{symbol}",
        )

    async def alert_duplicate_order(self, symbol: str, client_order_id: str, detail: str) -> None:
        """Alert on a detected duplicate order submission for the same
        symbol/client_order_id. No cooldown -- always worth a fresh look."""
        await self.fire(
            category=AlertCategory.DUPLICATE_ORDER,
            severity=AlertSeverity.CRITICAL,
            title=f"Duplicate Order Detected: {symbol}",
            message=f"{symbol}: {detail} (client_order_id={client_order_id})",
            details={"symbol": symbol, "client_order_id": client_order_id, "detail": detail},
            key=f"duplicate_order:{symbol}:{client_order_id}",
        )

    async def alert_position_desync(self, symbol: str, expected_qty: float, actual_qty: float) -> None:
        """Alert when a reconciliation pass finds the exchange's reported
        position for a symbol doesn't match what the bot's internal
        state/DB expected (ghost snapshot, orphan position, etc.)."""
        await self.fire(
            category=AlertCategory.POSITION_DESYNC,
            severity=AlertSeverity.CRITICAL,
            title=f"Position Desync: {symbol}",
            message=f"{symbol}: expected qty={expected_qty}, exchange reports qty={actual_qty}",
            details={"symbol": symbol, "expected_qty": expected_qty, "actual_qty": actual_qty},
            key=f"position_desync:{symbol}",
        )

    async def run_monitoring_cycle(self) -> None:
        """Periodic check of exposure and drawdown against risk_manager state.

        Called every 30s from bot.py's alerting_monitoring_loop. No-op if
        this engine wasn't constructed with a risk_manager.
        """
        if self.risk_manager is None:
            return
        status = await self.risk_manager.update_account_status()
        if status.get("status") != "risk_ok":
            return

        max_exposure = self.risk_manager._get_max_portfolio_cap()
        current_exposure = status.get("current_exposure", 0.0)
        if not math.isfinite(current_exposure):
            current_exposure = 0.0
        if max_exposure > 0:
            exposure_pct = current_exposure / max_exposure
            if exposure_pct >= 0.8:
                await self.alert_exposure_saturation(current_exposure, max_exposure, exposure_pct)

        max_drawdown = settings.MAX_DRAWDOWN_STOP  # negative, e.g. -10.0
        drawdown_pct = status.get("drawdown_pct", 0.0)  # negative or 0
        if not math.isfinite(drawdown_pct):
            drawdown_pct = 0.0
        if max_drawdown < 0:
            pct_of_max = drawdown_pct / max_drawdown  # both negative -> positive ratio
            if pct_of_max >= 0.5:
                await self.alert_drawdown_approaching_killswitch(drawdown_pct, max_drawdown, pct_of_max)

        # Performance decay and feature drift are slow-moving signals backed
        # by a DB query / state-file read respectively -- not worth the cost
        # on every ~30s tick, so gate them to self._slow_check_interval_sec.
        # Wired in 2026-09-20 (previously computed, never surfaced -- see
        # ADVERSARIAL_AUDIT_2026-09-20.md §0/§19).
        now = time.monotonic()
        if now - self._last_decay_check >= self._slow_check_interval_sec:
            self._last_decay_check = now
            try:
                from src.performance_tracker import get_decay_alerts
                for decay_alert in get_decay_alerts():
                    await self.alert_performance_decay(decay_alert)
            except Exception as e:
                logger.debug(f"Performance decay check skipped (non-fatal): {e}")

        if now - self._last_drift_check >= self._slow_check_interval_sec:
            self._last_drift_check = now
            try:
                from src.feature_drift_monitor import get_feature_drift_alerts
                for drift_alert in get_feature_drift_alerts():
                    await self.alert_feature_drift(drift_alert)
            except Exception as e:
                logger.debug(f"Feature drift check skipped (non-fatal): {e}")

    async def check_churn_alert(self, trades_last_hour: int, threshold: int = 20) -> None:
        """Fire a churn alert when trade frequency over the last hour is abnormally high."""
        if trades_last_hour >= threshold:
            await self.alert_churn_detected(trades_last_hour, 3600, list(settings.SYMBOLS))

    async def check_exchange_failure_alert(self, message: str, failure_count: int) -> None:
        """Fire a system-health alert on consecutive exchange failures."""
        threshold = getattr(settings, "CIRCUIT_FAILURE_THRESHOLD", 5)
        status = "down" if failure_count >= threshold else "degraded"
        await self.alert_system_health("exchange", status, {"message": message, "failure_count": failure_count})

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
    
    def get_alert_stats(self) -> dict[str, Any]:
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
_alerting_engine: AlertingEngine | None = None


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
        await engine.alert_drawdown_approaching_killswitch(-8.0, -10.0, 0.8)
        
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