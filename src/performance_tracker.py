"""Performance Tracker for Strategy/Regime Decay Monitoring.

Tracks out-of-sample rolling Sharpe and win-rate per (strategy, regime) pair
to detect edge decay early. Provides weekly decay reports.

Extended metrics: Avg Win/Loss, Profit Factor, Payoff Ratio, Expectancy, Turnover,
Cost-adjusted returns, Top trade contribution.
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from src.config import settings
from src.db import Trade, get_db_session
from src.logging_config import get_logger

logger = get_logger("performance_tracker")

_STATE_PATH = os.path.join("data", "performance_tracker_state.json")

# Decay detection thresholds
MIN_TRADES_FOR_STATS = 10
SHARPE_DECAY_THRESHOLD = 0.3  # Alert if rolling Sharpe drops below this
WIN_RATE_DECAY_THRESHOLD = 0.48  # Alert if win rate drops below this
PROFIT_FACTOR_DECAY_THRESHOLD = 1.2  # Alert if profit factor drops below this
LOOKBACK_WINDOWS = [30, 90, 180]  # Days for rolling windows


def _compute_extended_stats(returns: list[float], pnls: list[float]) -> dict[str, float]:
    """Compute extended statistics from returns and PnLs."""
    if len(returns) < MIN_TRADES_FOR_STATS:
        return {
            "sharpe": 0.0, "sortino": 0.0, "win_rate": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "payoff_ratio": 0.0,
            "profit_factor": 0.0, "expectancy": 0.0,
            "n": len(returns),
        }

    arr = np.array(returns)
    pnl_arr = np.array(pnls)
    
    # Basic stats
    sharpe = float(arr.mean() / (arr.std() + 1e-8) * np.sqrt(252)) if arr.std() > 0 else 0.0
    win_rate = float((arr > 0).mean())
    
    # Win/Loss separation
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_pnls = pnl_arr[pnl_arr > 0]
    loss_pnls = pnl_arr[pnl_arr < 0]
    
    # Avg win/loss
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    avg_win_usd = float(win_pnls.mean()) if len(win_pnls) else 0.0
    avg_loss_usd = float(loss_pnls.mean()) if len(loss_pnls) else 0.0
    
    # Payoff ratio
    if len(losses) > 0 and losses.mean() != 0:
        payoff_ratio = float(wins.mean() / abs(losses.mean()))
    else:
        payoff_ratio = float('inf') if len(wins) > 0 else 0.0
    
    # Profit factor
    if len(loss_pnls) > 0 and loss_pnls.sum() != 0:
        profit_factor = float(win_pnls.sum() / abs(loss_pnls.sum()))
    else:
        profit_factor = float('inf') if len(win_pnls) > 0 else 0.0
    
    # Expectancy
    expectancy = float(arr.mean())
    expectancy_usd = float(pnl_arr.mean())
    
    # Sortino
    downside = arr[arr < 0]
    sortino = float(arr.mean() / (downside.std() + 1e-8) * np.sqrt(252)) if len(downside) > 1 and downside.std() > 0 else (float('inf') if arr.mean() > 0 else 0.0)
    
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_win_usd": avg_win_usd,
        "avg_loss_usd": avg_loss_usd,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "expectancy_usd": expectancy_usd,
        "n": len(arr),
    }


class PerformanceTracker:
    """Tracks and reports strategy/regime performance for decay detection."""

    def __init__(self, state_path: str = _STATE_PATH):
        self.state_path = state_path
        self._state: dict[str, Any] = {}
        self._load_state()

    def _load_state(self) -> None:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    self._state = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load performance tracker state: {e}")
                self._state = {}
        else:
            self._state = {}

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        try:
            with open(self.state_path, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save performance tracker state: {e}")

    def record_trade(
        self,
        strategy: str,
        regime: str,
        pnl: float,
        return_pct: float,
        timestamp: datetime | None = None,
        notional_usd: float | None = None,
    ) -> None:
        """Record a completed trade for performance tracking."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        key = f"{strategy}|{regime}"
        if key not in self._state:
            self._state[key] = {"trades": []}

        trade_record = {
            "timestamp": timestamp.isoformat(),
            "pnl": float(pnl),
            "return_pct": float(return_pct),
        }
        if notional_usd is not None:
            trade_record["notional_usd"] = float(notional_usd)

        self._state[key]["trades"].append(trade_record)

        # Keep only last 500 trades per strategy/regime
        if len(self._state[key]["trades"]) > 500:
            self._state[key]["trades"] = self._state[key]["trades"][-500:]

        self._save_state()

    def _compute_window_stats(self, window_trades: list[dict]) -> dict[str, float]:
        """Compute all stats for a window of trades."""
        if not window_trades:
            return {}
        
        returns = [t["return_pct"] / 100.0 for t in window_trades]
        pnls = [t["pnl"] for t in window_trades]
        
        return _compute_extended_stats(returns, pnls)

    def get_performance_report(self) -> dict[str, Any]:
        """Generate performance report with rolling windows for all strategy/regime pairs."""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "pairs": {},
            "alerts": [],
        }

        for key, data in self._state.items():
            trades = data.get("trades", [])
            if len(trades) < MIN_TRADES_FOR_STATS:
                continue

            # Parse timestamps and returns
            trade_data = []
            for t in trades:
                try:
                    ts = datetime.fromisoformat(t["timestamp"])
                    ret = t["return_pct"] / 100.0
                    pnl = t["pnl"]
                    notional = t.get("notional_usd", 0.0)
                    trade_data.append((ts, ret, pnl, notional))
                except Exception:
                    continue

            if not trade_data:
                continue

            trade_data.sort(key=lambda x: x[0])
            now = datetime.utcnow()

            pair_report = {"total_trades": len(trade_data), "windows": {}}

            for window_days in LOOKBACK_WINDOWS:
                cutoff = now - timedelta(days=window_days)
                window_trades = [
                    {"return_pct": ret * 100, "pnl": pnl, "notional_usd": notional}
                    for ts, ret, pnl, notional in trade_data if ts >= cutoff
                ]

                if len(window_trades) >= MIN_TRADES_FOR_STATS:
                    stats = self._compute_window_stats(window_trades)
                    stats["total_notional"] = sum(t["notional_usd"] for t in window_trades)
                    stats["turnover_annualized"] = float(len(window_trades) * 252 / window_days) if window_days > 0 else 0.0
                    pair_report["windows"][f"{window_days}d"] = stats

                    # Check for decay alerts (only on 30d window for responsiveness)
                    if window_days == 30:
                        if stats["sharpe"] < SHARPE_DECAY_THRESHOLD:
                            report["alerts"].append({
                                "type": "sharpe_decay",
                                "pair": key,
                                "sharpe": stats["sharpe"],
                                "threshold": SHARPE_DECAY_THRESHOLD,
                                "trades": stats["n"],
                            })
                        if stats["win_rate"] < WIN_RATE_DECAY_THRESHOLD:
                            report["alerts"].append({
                                "type": "win_rate_decay",
                                "pair": key,
                                "win_rate": stats["win_rate"],
                                "threshold": WIN_RATE_DECAY_THRESHOLD,
                                "trades": stats["n"],
                            })
                        if stats["profit_factor"] < PROFIT_FACTOR_DECAY_THRESHOLD:
                            report["alerts"].append({
                                "type": "profit_factor_decay",
                                "pair": key,
                                "profit_factor": stats["profit_factor"],
                                "threshold": PROFIT_FACTOR_DECAY_THRESHOLD,
                                "trades": stats["n"],
                            })

            report["pairs"][key] = pair_report

        return report

    def log_weekly_report(self) -> None:
        """Log a weekly performance summary with decay alerts."""
        report = self.get_performance_report()

        logger.info("=" * 70)
        logger.info("WEEKLY STRATEGY/REGIME PERFORMANCE REPORT")
        logger.info("=" * 70)

        if not report["pairs"]:
            logger.info("Insufficient data for performance report.")
            return

        for key, data in sorted(report["pairs"].items()):
            logger.info(f"\n{key}: {data['total_trades']} total trades")
            for window, stats in data["windows"].items():
                logger.info(
                    f"  {window}: Sharpe={stats['sharpe']:.2f}, Sortino={stats['sortino']:.2f}, "
                    f"WinRate={stats['win_rate']:.1%}, ProfitFactor={stats['profit_factor']:.2f}, "
                    f"PayoffRatio={stats['payoff_ratio']:.2f}, Expectancy={stats['expectancy']:.4f}, "
                    f"AvgWin={stats['avg_win']:.4f}, AvgLoss={stats['avg_loss']:.4f}, "
                    f"N={stats['n']}, Turnover={stats['turnover_annualized']:.1f}x/yr"
                )

        if report["alerts"]:
            logger.warning("\n⚠️  DECAY ALERTS:")
            for alert in report["alerts"]:
                val = alert.get('sharpe', alert.get('win_rate', alert.get('profit_factor', 0)))
                logger.warning(
                    f"  {alert['pair']}: {alert['type'].upper()} "
                    f"({val:.3f} < {alert['threshold']:.3f}) "
                    f"N={alert['trades']}"
                )
        else:
            logger.info("\n✅ No decay alerts.")

        logger.info("=" * 70)


# Singleton instance
_TRACKER = None


def get_performance_tracker() -> PerformanceTracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = PerformanceTracker()
    return _TRACKER


def record_trade_outcome(
    strategy: str,
    regime: str,
    pnl: float,
    return_pct: float,
    timestamp: datetime | None = None,
    notional_usd: float | None = None,
) -> None:
    """Convenience function to record a trade outcome."""
    tracker = get_performance_tracker()
    tracker.record_trade(strategy, regime, pnl, return_pct, timestamp, notional_usd)


def log_weekly_performance() -> None:
    """Convenience function to log weekly report."""
    tracker = get_performance_tracker()
    tracker.log_weekly_report()


def get_decay_alerts() -> list[dict[str, Any]]:
    """Get current decay alerts for external monitoring."""
    tracker = get_performance_tracker()
    report = tracker.get_performance_report()
    return report["alerts"]