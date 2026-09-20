"""Feature Usage Verification for LEGACY_FEATURE_COLS.

Answers, per legacy feature column: is it actually consumed by anything in
the LIVE trading path outside the ML model's own input (CORE_FEATURE_COLS),
based on direct code-usage inspection (VERIFIED_LEGACY_USAGE below) rather
than domain-knowledge speculation? "Not part of the transformer's input" is
NOT the same claim as "safe to remove" -- see VERIFIED_LEGACY_USAGE's
docstring for why rsi/atr specifically are load-bearing elsewhere and must
NOT be removed despite not being in CORE_FEATURE_COLS.

Only applies to LEGACY_FEATURE_COLS — never touches CORE_FEATURE_COLS (the
model's actual training/inference input).

Does not run a full comparative-retrain ablation study (removing a feature,
retraining the transformer, and measuring the OOS delta) -- that is a
materially larger undertaking (real GPU time, a full retrain per candidate)
that these functions do not attempt. What's here is code-verified, not
retrain-verified.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from src.config import settings
from src.feature_engineering import CORE_FEATURE_COLS, LEGACY_FEATURE_COLS, add_multi_timeframe_features, compute_cross_asset_returns, compute_correlation_matrix
from src.logging_config import get_logger

logger = get_logger("feature_ablation")

_STATE_PATH = os.path.join("data", "feature_ablation_state.json")
_MIN_STABILITY_WINDOWS = 3
_MAX_CORR_CLUSTER = 0.85
_MAX_OOS_DROP_PCT = 1.0  # % drop acceptable to remove cluster

# Pre-defined clusters for the 11-core + legacy pool (derived from domain knowledge)
FEATURE_CLUSTERS = {
    "core_vol": ["parkinson_vol", "garman_klass_vol", "vol_of_vol"],
    "core_trend_memory": ["roll_autocorr", "z_return", "range_position_z"],
    "core_microstructure": ["kyle_lambda", "signed_flow", "trade_size_proxy"],
    "core_value": ["vwap_z", "amihud_z"],
    "legacy_retail": ["rsi", "macd", "atr", "volume_spike", "bollinger_width"],
    "legacy_derivatives": ["funding_rate", "open_interest", "long_short_ratio", "bid_ask_imbalance"],
}


def compute_feature_correlation_matrix(feature_df: "pd.DataFrame") -> tuple:  # noqa: F821
    """Compute feature correlation matrix for ablation analysis."""
    if feature_df is None or feature_df.empty:
        return np.array([]), []
    
    cols = [c for c in feature_df.columns if c in CORE_FEATURE_COLS + LEGACY_FEATURE_COLS]
    if len(cols) < 2:
        return np.array([]), cols
    
    corr = feature_df[cols].corr().values
    return corr, cols


def identify_high_corr_clusters(corr_matrix: np.ndarray, features: list[str], threshold: float = 0.85) -> list[set[str]]:
    """Find feature clusters with pairwise correlation above threshold."""
    clusters: list[set[str]] = []
    visited = set()
    
    for i, f1 in enumerate(features):
        if f1 in visited:
            continue
        cluster = {f1}
        for j, f2 in enumerate(features):
            if i == j or f2 in visited:
                continue
            if corr_matrix[i][j] >= threshold:
                cluster.add(f2)
        # Only keep clusters with 2+ members (true redundancy)
        if len(cluster) >= 2:
            clusters.append(cluster)
            visited.update(cluster)
    
    return clusters


# VERIFIED usage of each LEGACY_FEATURE_COLS column OUTSIDE feature_engineering.py
# itself, established by direct code inspection (repo-wide grep, 2026-09-20) --
# NOT domain-knowledge guesses. This matters because "not part of CORE_FEATURE_COLS
# (the transformer's input)" is NOT the same claim as "safe to remove" -- a column
# can be irrelevant to the ML model but still be load-bearing elsewhere.
#
# - rsi, atr: LOAD-BEARING, NOT SAFE TO REMOVE. src/strategies.py's
#   analyze_market_regime() (lines ~210-227) reads bars_df_features['atr']/['rsi']
#   (the add_features() output, i.e. exactly these legacy columns) as the
#   PRIMARY source for regime classification and the signal dict's atr/rsi
#   fields -- which quant_brain.py's RSI-threshold votes, the high-volatility
#   regime stand-aside check, and strategies.py's own profit-target/stop-loss/
#   trailing-stop exit logic all consume directly. A separate _calculate_rsi/
#   _calculate_atr fallback only fires if the feature value is missing/NaN, so
#   in normal operation these ARE the values driving trading decisions.
#   The previous version of this file's recommendations called this cluster
#   "safe to test... without affecting model input" -- true only for the
#   transformer's input, false for the rest of the system, and dangerously
#   misleading if acted on literally.
# - macd: genuinely dead in the live signal path. Repo-wide grep found no
#   `signal["macd"] = ...` assignment anywhere in strategies.py; the only
#   reader (committee.py:344, feeding rl_meta.py's PPO observation) always
#   gets committee.py's own `signal.get("macd", 0.0)` default. Computing it in
#   add_features() is wasted work, but "ablating" it changes nothing live
#   since nothing live currently reads a real value for it.
# - volume_spike, bollinger_width: genuinely unreferenced anywhere outside
#   feature_engineering.py/feature_ablation.py (repo-wide grep). Dead columns.
# - funding_rate, open_interest, long_short_ratio, bid_ask_imbalance: this
#   file's original claim was ACCURATE -- these are populated in strategies.py
#   from a separate live source (src.onchain_data's deriv_data, not from
#   add_features()'s legacy columns), so add_features() computing placeholder
#   versions of them is genuinely redundant with the real values used
#   elsewhere.
VERIFIED_LEGACY_USAGE = {
    "rsi": {"safe_to_remove": False, "reason": "primary source for regime classification, quant_brain RSI votes, and price-based exit logic via strategies.py's analyze_market_regime"},
    "atr": {"safe_to_remove": False, "reason": "primary source for regime classification (high-volatility stand-aside threshold) and ATR-based stop-distance sizing via strategies.py's analyze_market_regime"},
    "macd": {"safe_to_remove": True, "reason": "never assigned into the live signal dict anywhere in strategies.py; only reader always falls back to a hardcoded 0.0"},
    "volume_spike": {"safe_to_remove": True, "reason": "no reader found anywhere outside feature_engineering.py/feature_ablation.py"},
    "bollinger_width": {"safe_to_remove": True, "reason": "no reader found anywhere outside feature_engineering.py/feature_ablation.py"},
    "funding_rate": {"safe_to_remove": True, "reason": "live value comes from src.onchain_data's deriv_data instead; this column is an unused placeholder"},
    "open_interest": {"safe_to_remove": True, "reason": "live value comes from src.onchain_data's deriv_data instead; this column is an unused placeholder"},
    "long_short_ratio": {"safe_to_remove": True, "reason": "live value comes from src.onchain_data's deriv_data instead; this column is an unused placeholder"},
    "bid_ask_imbalance": {"safe_to_remove": True, "reason": "live value comes from src.onchain_data's deriv_data instead; this column is an unused placeholder"},
}


def propose_ablation_candidates(
    feature_dfs: dict[str, "pd.DataFrame"],  # noqa: F821
    symbol: str = "BTC/USD",
    lookback: int = 50,
) -> list[dict[str, Any]]:
    """
    Propose feature clusters that could be removed safely, based on
    VERIFIED_LEGACY_USAGE above (direct code inspection of actual consumers),
    not domain-knowledge speculation. Also computes REAL correlations between
    core and legacy features from the feature_dfs actually passed in, when
    available (previously this function ignored feature_dfs entirely and
    returned static text regardless of input -- fixed 2026-09-20, see
    ADVERSARIAL_AUDIT_2026-09-20.md §0/§11).

    Returns recommendations with evidence (verified usage + real correlation
    where computable). Genuinely safe-to-remove columns (macd, volume_spike,
    bollinger_width, and the 4 unused derivatives placeholders) are flagged
    "safe_removal"; rsi/atr are explicitly flagged "DO NOT REMOVE" since they
    are load-bearing for live trading logic outside the ML model.
    """
    recommendations = []

    for feature, usage in VERIFIED_LEGACY_USAGE.items():
        rec = {
            "feature": feature,
            "safe_to_remove": usage["safe_to_remove"],
            "action": "safe_removal" if usage["safe_to_remove"] else "DO_NOT_REMOVE",
            "reason": usage["reason"],
            "priority": "low" if usage["safe_to_remove"] else "n/a (load-bearing)",
        }
        # Attach real correlation with each core feature when we have actual
        # data to compute it from (not required -- verified usage above is
        # the primary evidence; this is a secondary corroborating signal).
        for sym, df in (feature_dfs or {}).items():
            corr_matrix, cols = compute_feature_correlation_matrix(df)
            if corr_matrix.size == 0 or feature not in cols:
                continue
            idx = cols.index(feature)
            core_corrs = {c: float(corr_matrix[idx][cols.index(c)]) for c in CORE_FEATURE_COLS if c in cols}
            if core_corrs:
                best_core_match = max(core_corrs, key=lambda k: abs(core_corrs[k]))
                rec["measured_correlation"] = {
                    "symbol": sym,
                    "most_correlated_core_feature": best_core_match,
                    "correlation": core_corrs[best_core_match],
                }
            break  # one symbol's measurement is enough evidence to attach
        recommendations.append(rec)

    return recommendations


def run_ablation_shadow(
    symbol: str,
    feature_df: "pd.DataFrame",  # noqa: F821
    base_result: Any,  # BacktestResult
    chart_interval_days: int = 30,
) -> dict[str, Any]:
    """Report on which LEGACY_FEATURE_COLS columns are actually safe to stop
    computing, based on verified live-code usage (VERIFIED_LEGACY_USAGE) plus
    real measured correlation with core features where feature_df is supplied.

    This does NOT run a full comparative retrain/backtest ablation (that
    would require re-fitting the transformer per candidate removal, which is
    out of scope for a lightweight shadow report) -- it answers a narrower
    but directly actionable and CORRECT question: "is this column consumed by
    anything outside the ML model's input, such that removing it would change
    live trading behavior?" The previous version of this function answered a
    different, incorrect question (treated "not fed to the transformer" as
    equivalent to "safe to remove everywhere"), which is false for rsi/atr --
    see VERIFIED_LEGACY_USAGE and ADVERSARIAL_AUDIT_2026-09-20.md §11.
    """
    recommendations = propose_ablation_candidates({symbol: feature_df}, symbol)
    safe = [r for r in recommendations if r["safe_to_remove"]]
    unsafe = [r for r in recommendations if not r["safe_to_remove"]]

    report = {
        "symbol": symbol,
        "recommendations": recommendations,
        "safe_to_remove": [r["feature"] for r in safe],
        "do_not_remove": [r["feature"] for r in unsafe],
        "methodology": "Verified live-code usage inspection (not domain-knowledge speculation) "
                       "+ measured correlation with core features where data is available. "
                       "Does not include a full comparative-retrain ablation.",
        "status": "verified",
    }

    logger.info(
        f"Feature ablation report for {symbol}: {len(safe)} columns safe to stop computing "
        f"({[r['feature'] for r in safe]}), {len(unsafe)} load-bearing outside the ML model "
        f"and must NOT be removed ({[r['feature'] for r in unsafe]})"
    )
    return report


def save_ablation_report(report: dict[str, Any], path: str = "data/feature_ablation_report.json") -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
