"""
Shared Decision Source Gating — Step 4 of Foundation Hardening.

Single authoritative gate that ALL decision sources (adaptive learner, PPO, DT,
hierarchical skills, future sources) must pass before going live.
Eliminates the class of bug where a new fallback source bypasses validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.config import settings
from src.logging_config import get_logger

logger = get_logger("decision_gate")

DecisionSource = Literal[
    "adaptive_learner", "ppo", "decision_transformer", "hierarchical_skills"
]


@dataclass(frozen=True)
class GateResult:
    """Result of the decision source gate check."""
    allowed: bool
    source: DecisionSource
    regime: str
    reason: str
    details: dict


def _get_learner():
    """Lazy import to avoid circular dependency."""
    from src.committee.committee import get_meta_learner
    return get_meta_learner()


def check_decision_source_gate(
    source: DecisionSource,
    regime: str,
    *,
    require_validation: bool = True,
    require_min_trades: bool = True,
) -> GateResult:
    """
    THE single gate all decision sources must pass to go live.
    
    Args:
        source: Which decision source is requesting live status
        regime: Current market regime
        require_validation: If True, require regime validation (Sharpe/win-rate)
        require_min_trades: If True, require minimum trade count
        
    Returns:
        GateResult with allowed=True only if ALL applicable gates pass
        
    Design principle: fail closed. If in doubt, the source stays in shadow mode.
    """
    learner = _get_learner()
    details = {"source": source, "regime": regime}
    
    # 1. Global enable flag
    if not settings.ADAPTIVE_ML_ENABLED:
        return GateResult(
            allowed=False,
            source=source,
            regime=regime,
            reason="ADAPTIVE_ML_ENABLED is False",
            details={**details, "global_enabled": False}
        )
    details["global_enabled"] = True
    
    # 2. Learner must exist
    if learner is None:
        return GateResult(
            allowed=False,
            source=source,
            regime=regime,
            reason="Adaptive meta-learner not available",
            details={**details, "learner_available": False}
        )
    details["learner_available"] = True
    
    # 3. Per-regime sample count gate
    regime_samples = learner.sample_count_for_regime(regime)
    details["regime_samples"] = regime_samples
    
    if require_min_trades:
        min_trades = _get_min_trades_for_source(source)
        if regime_samples < min_trades:
            return GateResult(
                allowed=False,
                source=source,
                regime=regime,
                reason=f"Insufficient regime samples: {regime_samples} < {min_trades}",
                details={**details, "min_trades_required": min_trades, "min_trades_gate": False}
            )
    details["min_trades_gate"] = True
    details["min_trades_required"] = _get_min_trades_for_source(source)
    
    # 4. Regime validation gate (Sharpe + win-rate on holdout)
    if require_validation:
        validated = learner.is_regime_validated(regime)
        details["regime_validated"] = validated
        details["validation_metrics"] = learner.get_regime_validation_metrics(regime)
        
        if not validated:
            return GateResult(
                allowed=False,
                source=source,
                regime=regime,
                reason=f"Regime '{regime}' not validated (Sharpe/win-rate gate failed)",
                details=details
            )
    details["validation_gate"] = True
    
    # 5. Source-specific readiness checks
    source_ready, source_reason, source_details = _check_source_specific(source, regime)
    details.update(source_details)
    
    if not source_ready:
        return GateResult(
            allowed=False,
            source=source,
            regime=regime,
            reason=f"Source-specific check failed: {source_reason}",
            details=details
        )
    details["source_ready"] = True
    
    # All gates passed
    logger.info(
        f"DECISION GATE PASSED: source={source} regime={regime} "
        f"samples={regime_samples} validated={details.get('regime_validated', 'N/A')}"
    )
    return GateResult(
        allowed=True,
        source=source,
        regime=regime,
        reason="All gates passed",
        details=details
    )


def _get_min_trades_for_source(source: DecisionSource) -> int:
    """Minimum trades required per source type."""
    if source == "adaptive_learner":
        return settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE
    elif source == "ppo":
        return settings.PPO_MIN_TRADES_BEFORE_LIVE
    elif source == "decision_transformer":
        # DT uses the same gate as adaptive learner by default
        return settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE
    elif source == "hierarchical_skills":
        return settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE
    return settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE


def _check_source_specific(source: DecisionSource, regime: str) -> tuple[bool, str, dict]:
    """Source-specific readiness checks (model loaded, etc.)."""
    if source == "adaptive_learner":
        # Mathematical learner is ready if learner exists (checked above)
        return True, "OK", {"adaptive_learner_ready": True}
    
    elif source == "ppo":
        from src.committee.rl_meta import RLMetaLearner
        rl_learner = RLMetaLearner()
        model_loaded = getattr(rl_learner, "model", None) is not None
        return model_loaded, "PPO model loaded" if model_loaded else "PPO model not loaded", {
            "ppo_model_loaded": model_loaded
        }
    
    elif source == "decision_transformer":
        from src.committee.decision_transformer import get_decision_transformer
        dt = get_decision_transformer()
        model_loaded = dt is not None and hasattr(dt, "model") and dt.model is not None
        return model_loaded, "DT model loaded" if model_loaded else "DT model not loaded", {
            "dt_model_loaded": model_loaded
        }
    
    elif source == "hierarchical_skills":
        # Check if hierarchical skills trainer/model is available
        try:
            from src.committee.hierarchical_skills import get_hierarchical_skills
            skills = get_hierarchical_skills()
            ready = skills is not None
            return ready, "Skills model loaded" if ready else "Skills model not loaded", {
                "hierarchical_skills_ready": ready
            }
        except Exception as e:
            return False, f"Hierarchical skills error: {e}", {"hierarchical_skills_ready": False}
    
    return False, f"Unknown source: {source}", {}


def get_gate_status_summary(regime: str) -> dict:
    """Get gate status for all sources for a regime (for observability)."""
    sources: list[DecisionSource] = [
        "adaptive_learner", "ppo", "decision_transformer", "hierarchical_skills"
    ]
    
    summary = {"regime": regime, "sources": {}}
    for source in sources:
        result = check_decision_source_gate(source, regime)
        summary["sources"][source] = {
            "allowed": result.allowed,
            "reason": result.reason,
            "details": result.details
        }
    return summary


def log_gate_status(regime: str) -> None:
    """Log gate status for all sources (call periodically for observability)."""
    summary = get_gate_status_summary(regime)
    for source, info in summary["sources"].items():
        status = "LIVE" if info["allowed"] else "SHADOW"
        logger.info(f"Gate {source} [{status}] for regime '{regime}': {info['reason']}")