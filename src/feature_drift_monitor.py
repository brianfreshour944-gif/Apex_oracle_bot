"""Feature Drift Monitor.

Tracks Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistic
for core features to detect distributional shift over time.
Logs weekly drift reports and alerts when features drift beyond thresholds.
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.config import settings
from src.logging_config import get_logger

logger = get_logger("feature_drift")

_STATE_PATH = os.path.join("data", "feature_drift_state.json")

# Drift detection thresholds (per feature)
PSI_THRESHOLDS = {
    "low": 0.1,      # Minor drift - log only
    "medium": 0.2,   # Moderate drift - alert
    "high": 0.5,     # Severe drift - alert + consider retraining
}

KS_THRESHOLDS = {
    "low": 0.05,     # p-value threshold for low drift
    "medium": 0.01,  # p-value threshold for moderate drift
    "high": 0.001,   # p-value threshold for severe drift
}

# Core features to monitor (must match CORE_FEATURE_COLS)
CORE_FEATURES = [
    "z_return", "parkinson_vol", "garman_klass_vol", "kyle_lambda",
    "signed_flow", "vwap_z", "vol_of_vol", "amihud_z",
    "trade_size_proxy", "roll_autocorr", "range_position_z"
]

# Reference window for baseline distribution (days)
REFERENCE_WINDOW_DAYS = 30
# Monitoring window for current distribution (days)
MONITOR_WINDOW_DAYS = 7
# Minimum samples needed
MIN_SAMPLES = 100


def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between reference and current distributions.
    
    PSI = sum((current_pct - reference_pct) * log(current_pct / reference_pct))
    
    Args:
        reference: Reference distribution samples
        current: Current distribution samples
        bins: Number of quantile bins
        
    Returns:
        PSI value (0 = no drift, >0.2 = concerning, >0.5 = severe)
    """
    if len(reference) < MIN_SAMPLES or len(current) < MIN_SAMPLES:
        return 0.0
    
    # Use quantile bins for equal-frequency binning
    try:
        _, bin_edges = pd.qcut(reference, q=bins, retbins=True, duplicates='drop')
        # Ensure bin edges cover full range
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        cur_hist, _ = np.histogram(current, bins=bin_edges)
        
        ref_pct = ref_hist / len(reference)
        cur_pct = cur_hist / len(current)
        
        # Avoid division by zero
        ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
        cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
        
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)
    except Exception:
        return 0.0


def _compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """
    Compute Kolmogorov-Smirnov statistic and p-value.
    
    Returns:
        (ks_statistic, p_value)
    """
    if len(reference) < MIN_SAMPLES or len(current) < MIN_SAMPLES:
        return 0.0, 1.0
    
    try:
        from scipy import stats
        ks_stat, p_value = stats.ks_2samp(reference, current)
        return float(ks_stat), float(p_value)
    except ImportError:
        # Fallback: manual KS test approximation
        ref_sorted = np.sort(reference)
        cur_sorted = np.sort(current)
        
        # Evaluate CDF difference at all points
        all_vals = np.sort(np.concatenate([ref_sorted, cur_sorted]))
        ref_cdf = np.searchsorted(ref_sorted, all_vals, side='right') / len(ref_sorted)
        cur_cdf = np.searchsorted(cur_sorted, all_vals, side='right') / len(cur_sorted)
        ks_stat = float(np.max(np.abs(ref_cdf - cur_cdf)))
        
        # Approximate p-value (Dvoretzky-Kiefer-Wolfowitz)
        n_eff = len(reference) * len(current) / (len(reference) + len(current))
        p_value = float(2 * np.exp(-2 * n_eff * ks_stat ** 2))
        return ks_stat, p_value


def _classify_drift(psi: float, ks_pvalue: float) -> str:
    """Classify drift severity based on PSI and KS p-value."""
    if psi >= PSI_THRESHOLDS["high"] or ks_pvalue <= KS_THRESHOLDS["high"]:
        return "high"
    elif psi >= PSI_THRESHOLDS["medium"] or ks_pvalue <= KS_THRESHOLDS["medium"]:
        return "medium"
    elif psi >= PSI_THRESHOLDS["low"] or ks_pvalue <= KS_THRESHOLDS["low"]:
        return "low"
    return "none"


class FeatureDriftMonitor:
    """Monitors feature distributional drift using PSI and KS tests."""
    
    def __init__(self, state_path: str = _STATE_PATH):
        self.state_path = state_path
        self._state: dict[str, Any] = {
            "reference_windows": {},      # feature -> { "values": [], "window_start": iso_ts }
            "monitor_windows": {},        # feature -> { "values": [], "window_start": iso_ts }
            "last_report": None,
            "alerts": [],
        }
        self._load_state()
    
    def _load_state(self) -> None:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    self._state = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load feature drift state: {e}")
                self._state = {
                    "reference_windows": {},
                    "monitor_windows": {},
                    "last_report": None,
                    "alerts": [],
                }
    
    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        try:
            with open(self.state_path, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feature drift state: {e}")
    
    def record_features(self, symbol: str, features: dict[str, float], timestamp: datetime | None = None) -> None:
        """
        Record feature values for drift monitoring.
        
        Args:
            symbol: Trading symbol
            features: Dict of feature_name -> value
            timestamp: Timestamp of the observation (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        ts_iso = timestamp.isoformat()
        
        for feature_name in CORE_FEATURES:
            if feature_name not in features:
                continue
            
            value = features[feature_name]
            if not np.isfinite(value):
                continue
            
            # Initialize windows if needed
            for window_dict in [self._state["reference_windows"], self._state["monitor_windows"]]:
                if feature_name not in window_dict:
                    window_dict[feature_name] = {"values": [], "window_start": ts_iso}
            
            # Add to monitor window (rolling 7 days)
            self._state["monitor_windows"][feature_name]["values"].append({
                "value": float(value),
                "timestamp": ts_iso,
                "symbol": symbol,
            })
            
            # Also add to reference window (rolling 30 days)
            self._state["reference_windows"][feature_name]["values"].append({
                "value": float(value),
                "timestamp": ts_iso,
                "symbol": symbol,
            })
        
        # Prune old data
        self._prune_windows(timestamp)
        self._save_state()
    
    def _prune_windows(self, now: datetime) -> None:
        """Remove data older than window sizes."""
        ref_cutoff = now - timedelta(days=REFERENCE_WINDOW_DAYS)
        mon_cutoff = now - timedelta(days=MONITOR_WINDOW_DAYS)
        
        for window_dict, cutoff in [
            (self._state["reference_windows"], ref_cutoff),
            (self._state["monitor_windows"], mon_cutoff),
        ]:
            for feature_name, data in window_dict.items():
                if "values" in data:
                    data["values"] = [
                        v for v in data["values"]
                        if datetime.fromisoformat(v["timestamp"]) >= cutoff
                    ]
    
    def compute_drift_report(self) -> dict[str, Any]:
        """Compute drift metrics for all core features."""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "features": {},
            "alerts": [],
            "summary": {"low": 0, "medium": 0, "high": 0},
        }
        
        for feature_name in CORE_FEATURES:
            ref_data = self._state["reference_windows"].get(feature_name, {}).get("values", [])
            mon_data = self._state["monitor_windows"].get(feature_name, {}).get("values", [])
            
            if len(ref_data) < MIN_SAMPLES or len(mon_data) < MIN_SAMPLES:
                report["features"][feature_name] = {
                    "status": "insufficient_data",
                    "ref_samples": len(ref_data),
                    "mon_samples": len(mon_data),
                }
                continue
            
            ref_values = np.array([v["value"] for v in ref_data])
            mon_values = np.array([v["value"] for v in mon_data])
            
            psi = _compute_psi(ref_values, mon_values)
            ks_stat, ks_pvalue = _compute_ks(ref_values, mon_values)
            severity = _classify_drift(psi, ks_pvalue)
            
            feature_report = {
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
                "severity": severity,
                "ref_samples": len(ref_values),
                "mon_samples": len(mon_values),
                "ref_mean": float(np.mean(ref_values)),
                "ref_std": float(np.std(ref_values)),
                "mon_mean": float(np.mean(mon_values)),
                "mon_std": float(np.std(mon_values)),
                "mean_shift": float(np.mean(mon_values) - np.mean(ref_values)),
                "std_ratio": float(np.std(mon_values) / (np.std(ref_values) + 1e-8)),
            }
            
            report["features"][feature_name] = feature_report
            
            if severity != "none":
                report["summary"][severity] += 1
                alert = {
                    "feature": feature_name,
                    "severity": severity,
                    "psi": psi,
                    "ks_pvalue": ks_pvalue,
                    "mean_shift": feature_report["mean_shift"],
                    "std_ratio": feature_report["std_ratio"],
                }
                report["alerts"].append(alert)
        
        return report
    
    def log_weekly_report(self) -> None:
        """Log a weekly drift summary."""
        report = self.compute_drift_report()
        
        logger.info("=" * 70)
        logger.info("WEEKLY FEATURE DRIFT REPORT")
        logger.info("=" * 70)
        
        if not report["features"]:
            logger.info("Insufficient data for drift report.")
            return
        
        for feature_name, data in sorted(report["features"].items()):
            if data.get("status") == "insufficient_data":
                logger.info(f"  {feature_name}: INSUFFICIENT DATA (ref={data['ref_samples']}, mon={data['mon_samples']})")
                continue
            
            severity = data["severity"]
            severity_marker = {"low": "⚠️", "medium": "🟠", "high": "🔴", "none": "✅"}.get(severity, "")
            
            logger.info(
                f"  {feature_name}: PSI={data['psi']:.4f}, KS_p={data['ks_pvalue']:.4f}, "
                f"mean_shift={data['mean_shift']:+.4f}, std_ratio={data['std_ratio']:.2f} {severity_marker}"
            )
        
        if report["alerts"]:
            logger.warning(f"\n  DRIFT ALERTS: {report['summary']['high']} high, {report['summary']['medium']} medium, {report['summary']['low']} low")
            for alert in report["alerts"]:
                logger.warning(
                    f"    {alert['feature']}: {alert['severity'].upper()} "
                    f"(PSI={alert['psi']:.3f}, KS_p={alert['ks_pvalue']:.3f}, "
                    f"mean_shift={alert['mean_shift']:+.3f})"
                )
        else:
            logger.info("\n  ✅ No significant feature drift detected.")
        
        logger.info("=" * 70)
        
        # Update last report timestamp
        self._state["last_report"] = datetime.utcnow().isoformat()
        self._state["alerts"] = report["alerts"]
        self._save_state()
    
    def get_current_alerts(self) -> list[dict[str, Any]]:
        """Get current drift alerts for external monitoring."""
        report = self.compute_drift_report()
        return report["alerts"]


# Singleton instance
_MONITOR = None


def get_feature_drift_monitor() -> FeatureDriftMonitor:
    global _MONITOR
    if _MONITOR is None:
        _MONITOR = FeatureDriftMonitor()
    return _MONITOR


def record_features_for_drift(symbol: str, features: dict[str, float], timestamp: datetime | None = None) -> None:
    """Convenience function to record features for drift monitoring."""
    monitor = get_feature_drift_monitor()
    monitor.record_features(symbol, features, timestamp)


def log_feature_drift_report() -> None:
    """Convenience function to log weekly drift report."""
    monitor = get_feature_drift_monitor()
    monitor.log_weekly_report()


def get_feature_drift_alerts() -> list[dict[str, Any]]:
    """Get current feature drift alerts."""
    monitor = get_feature_drift_monitor()
    return monitor.get_current_alerts()