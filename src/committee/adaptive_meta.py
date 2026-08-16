"""Adaptive meta-learner that sits on top of the committee of brains.

Learns *which brain to trust* per market regime from realized trade outcomes,
using a simple, transparent exponentially-weighted reward rule. It never touches
signal generation, order sizing, stop-loss or the drawdown/daily-loss killswitch
(those live in ``risk.py`` and remain authoritative). It only re-weights the
existing brain votes when combining them into a final action.

Design goals: pragmatic, production-safe, stdlib-only, fail-safe.
- Cold-start = equal weights per regime.
- Weights are clamped to [min, max] and normalized per regime.
- Per-regime updates never bleed into other regimes.
- State persists atomically as versioned JSON; corrupt/missing state falls back
  to equal weights instead of crashing.
- Validation gate: per-regime weights only go live after positive Sharpe & win-rate on holdout.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.logging_config import get_logger
from src.config import settings as _settings

logger = get_logger("adaptive_meta")

# Canonical brain roster. Kept in sync with the committee's five brains.
BRAINS: List[str] = ["transformer", "quant", "momentum", "sentinel", "llm"]

STATE_VERSION = 1
_DIRECTIONAL = {"buy", "sell"}

# Validation gate thresholds — sourced from settings, with fallbacks for tests
VALIDATION_MIN_TRADES = getattr(_settings, "ADAPTIVE_MIN_VALIDATION_TRADES", 10)
VALIDATION_MIN_SHARPE = getattr(_settings, "ADAPTIVE_MIN_SHARPE", 0.5)
VALIDATION_MIN_WIN_RATE = getattr(_settings, "ADAPTIVE_MIN_WIN_RATE", 0.52)
VALIDATION_HOLDOUT_FRACTION = getattr(_settings, "ADAPTIVE_HOLDOUT_FRACTION", 0.3)


@dataclass
class BrainScore:
    """Per-brain contribution to a single combined decision."""

    name: str
    action: str
    confidence: float
    weight: float

    @property
    def contribution(self) -> float:
        return self.confidence * self.weight


@dataclass
class BrainPerformance:
    """Running performance stats for a brain within one regime (observability)."""

    name: str
    regime: str
    n_updates: int = 0
    n_correct: int = 0
    n_incorrect: int = 0
    cumulative_reward: float = 0.0

    @property
    def hit_rate(self) -> float:
        graded = self.n_correct + self.n_incorrect
        return (self.n_correct / graded) if graded else 0.0


@dataclass
class AdaptiveDecision:
    """Result of combining brain votes with learned weights."""

    action: str
    confidence: float
    regime: str
    weights: Dict[str, float]
    scores: List[BrainScore] = field(default_factory=list)
    explanation: str = ""


@dataclass
class UpdateReport:
    """Summary of a single ``update`` call (used for alerting/observability)."""

    regime: str
    label: int
    material_change: bool = False
    max_delta: float = 0.0
    old_weights: Dict[str, float] = field(default_factory=dict)
    new_weights: Dict[str, float] = field(default_factory=dict)
    # Weight drift vs equal-weight baseline
    drift_vs_equal: Dict[str, float] = field(default_factory=dict)
    drift_l2_norm: float = 0.0


def _equal_weights() -> Dict[str, float]:
    w = 1.0 / len(BRAINS)
    return {b: w for b in BRAINS}


def _vote_direction(action: str) -> Optional[str]:
    """Map a brain action to a directional stance, or None for hold/stand-aside."""
    return action if action in _DIRECTIONAL else None


class AdaptiveMetaLearner:
    """Stores per-brain, per-regime weights and evolves them from realized PnL."""

    def __init__(
        self,
        state_path: Optional[str] = None,
        learning_rate: float = 0.10,
        min_weight: float = 0.02,
        max_weight: float = 0.60,
    ) -> None:
        self.state_path = state_path
        self.learning_rate = float(learning_rate)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

        self.version = STATE_VERSION
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.last_update: Optional[str] = None
        self.sample_count = 0
        # Per-regime sample counts. `sample_count` above is the aggregate across
        # all regimes/symbols and must NOT be used to gate whether a specific
        # regime's weights are "live-ready" -- a regime with almost no samples
        # of its own could otherwise ride on volume accumulated by other,
        # unrelated regimes. Use `sample_count_for_regime()` for that gate.
        self.regime_sample_count: Dict[str, int] = {}
        self.weights: Dict[str, Dict[str, float]] = {}
        self.performance: Dict[str, Dict[str, BrainPerformance]] = {}
        # Per-regime returns history for validation (Sharpe, win-rate)
        self.regime_returns: Dict[str, List[float]] = {}
        # Validation status per regime
        self.regime_validated: Dict[str, bool] = {}

        if self.state_path:
            self.load()

    # ---- weight helpers -------------------------------------------------

    def _regime_weights(self, regime: str) -> Dict[str, float]:
        """Return (lazily initialising) the weight vector for a regime."""
        if regime not in self.weights:
            self.weights[regime] = _equal_weights()
        # Guard against stale state missing a brain (roster changes).
        for b in BRAINS:
            self.weights[regime].setdefault(b, self.min_weight)
        return self.weights[regime]

    def _clamp_normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Clamp each weight to [min, max] and normalize to sum 1 (per regime).

        Iterates clamp+normalize to a fixed point so both constraints hold
        simultaneously (feasible while ``min*n <= 1 <= max*n``).
        """
        w = dict(weights)
        for _ in range(50):
            w = {k: min(max(v, self.min_weight), self.max_weight) for k, v in w.items()}
            total = sum(w.values())
            if total <= 0:
                return _equal_weights()
            w = {k: v / total for k, v in w.items()}
            if all(self.min_weight - 1e-9 <= v <= self.max_weight + 1e-9 for v in w.values()):
                break
        return w

    def _compute_drift_vs_equal(self, weights: Dict[str, float]) -> tuple[Dict[str, float], float]:
        """Compute weight drift vs equal-weight baseline.
        
        Returns:
            - per-brain drift (weight - equal_weight)
            - L2 norm of drift vector
        """
        equal_w = 1.0 / len(BRAINS)
        drift = {b: weights.get(b, equal_w) - equal_w for b in BRAINS}
        l2_norm = math.sqrt(sum(d * d for d in drift.values()))
        return drift, l2_norm

    # ---- combine --------------------------------------------------------

    def combine(self, brain_outputs: List[Any], regime: str) -> AdaptiveDecision:
        """Combine brain votes into a final weighted action + confidence.

        ``brain_outputs`` is a list of BrainVote-like objects exposing
        ``name``, ``action`` and ``confidence``. Weights come from the learned
        per-regime vector (cold-start equal). Mirrors the committee's scoring
        semantics: directional votes accrue ``confidence * weight`` for their
        action; stand-aside/skip contribute nothing.
        """
        weights = self._clamp_normalize(self._regime_weights(regime))

        scores: List[BrainScore] = []
        action_scores: Dict[str, float] = {}
        for v in brain_outputs:
            w = weights.get(v.name, self.min_weight)
            scores.append(BrainScore(name=v.name, action=v.action, confidence=float(v.confidence), weight=w))
            if v.action in _DIRECTIONAL:
                action_scores[v.action] = action_scores.get(v.action, 0.0) + float(v.confidence) * w

        if not action_scores:
            action, confidence = "stand_aside", 0.0
        else:
            action = max(action_scores, key=action_scores.get)
            confidence = action_scores[action]

        explanation = "adaptive[{}] {} | {}".format(
            regime,
            f"{action}={confidence:.3f}",
            " ".join(f"{b}={weights.get(b, 0.0):.2f}" for b in BRAINS),
        )
        return AdaptiveDecision(
            action=action,
            confidence=round(confidence, 4),
            regime=regime,
            weights=weights,
            scores=scores,
            explanation=explanation,
        )

    # ---- update ---------------------------------------------------------

    @staticmethod
    def _extract_votes(decision_snapshot: Dict[str, Any]) -> Dict[str, str]:
        """Normalize a decision snapshot into {brain_name: action}."""
        if "brain_votes" in decision_snapshot and isinstance(decision_snapshot["brain_votes"], dict):
            return {str(k): str(v) for k, v in decision_snapshot["brain_votes"].items()}
        votes: Dict[str, str] = {}
        for v in decision_snapshot.get("votes", []) or []:
            if isinstance(v, dict):
                name, action = v.get("name"), v.get("action")
            else:  # BrainVote-like
                name, action = getattr(v, "name", None), getattr(v, "action", None)
            if name and action:
                votes[str(name)] = str(action)
        return votes

    @staticmethod
    def _extract_pnl(realized_outcome: Any) -> float:
        """Pull a signed net-PnL/return figure out of a flexible outcome payload."""
        if isinstance(realized_outcome, (int, float)):
            return float(realized_outcome)
        if isinstance(realized_outcome, dict):
            for key in ("net_pnl", "pnl", "realized_pnl", "return_pct", "pnl_pct", "return"):
                if realized_outcome.get(key) is not None:
                    return float(realized_outcome[key])
        return 0.0

    def _profitable_direction(self, final_action: str, pnl: float) -> Optional[str]:
        """Given the taken action and net PnL sign, which direction *was* right."""
        if final_action not in _DIRECTIONAL or pnl == 0.0:
            return None
        opposite = "sell" if final_action == "buy" else "buy"
        return final_action if pnl > 0 else opposite

    def update(self, decision_snapshot: Dict[str, Any], realized_outcome: Any) -> UpdateReport:
        """Evolve per-regime weights from one realized trade outcome.

        Increases a brain's weight when its directional vote matched the
        profitable direction, decreases it otherwise (exponential reward).
        Hold/stand-aside/skip votes are treated conservatively (no change).
        A flat (zero-PnL) outcome produces no weight change.
        """
        regime = str(decision_snapshot.get("regime", "default"))
        final_action = str(decision_snapshot.get("final_action", decision_snapshot.get("action", "hold")))
        pnl = self._extract_pnl(realized_outcome)
        # Extract return_pct for validation tracking
        return_pct = 0.0
        if isinstance(realized_outcome, dict):
            return_pct = float(realized_outcome.get("return_pct", realized_outcome.get("return", 0.0)))
        label = 1 if pnl > 0 else (0 if pnl < 0 else -1)

        report = UpdateReport(regime=regime, label=label)
        profitable_dir = self._profitable_direction(final_action, pnl)
        votes = self._extract_votes(decision_snapshot)

        old = dict(self._clamp_normalize(self._regime_weights(regime)))
        report.old_weights = old

        # No gradable signal (flat pnl, or non-directional final action): count
        # the sample but leave weights untouched.
        if profitable_dir is None:
            self.sample_count += 1
            self.regime_sample_count[regime] = self.regime_sample_count.get(regime, 0) + 1
            self.last_update = datetime.now(timezone.utc).isoformat()
            report.new_weights = old
            self._save_safely()
            return report

        # Record return for validation gate (even if we don't update weights)
        if return_pct != 0.0:
            self._record_return(regime, return_pct)

        raw = dict(self._regime_weights(regime))
        perf_regime = self.performance.setdefault(regime, {})
        for brain, action in votes.items():
            if brain not in raw:
                continue
            direction = _vote_direction(action)
            if direction is None:
                continue  # conservative: hold/stand-aside/skip do not move weights
            correct = direction == profitable_dir
            reward = 1.0 if correct else -1.0
            raw[brain] = raw[brain] * math.exp(self.learning_rate * reward)

            perf = perf_regime.setdefault(brain, BrainPerformance(name=brain, regime=regime))
            perf.n_updates += 1
            perf.cumulative_reward += reward
            if correct:
                perf.n_correct += 1
            else:
                perf.n_incorrect += 1

        new = self._clamp_normalize(raw)
        self.weights[regime] = new
        report.new_weights = new
        report.max_delta = max((abs(new[b] - old.get(b, 0.0)) for b in new), default=0.0)
        
        # Compute drift vs equal-weight baseline
        report.drift_vs_equal, report.drift_l2_norm = self._compute_drift_vs_equal(new)
        
        # Log significant drift
        if report.drift_l2_norm > 0.15:
            logger.warning(f"Regime '{regime}' weight drift L2={report.drift_l2_norm:.3f} vs equal: {report.drift_vs_equal}")

        self.sample_count += 1
        self.regime_sample_count[regime] = self.regime_sample_count.get(regime, 0) + 1
        self.last_update = datetime.now(timezone.utc).isoformat()
        self._save_safely()
        return report

    def sample_count_for_regime(self, regime: str) -> int:
        """Per-regime sample count. Use this (not the aggregate `sample_count`)
        to gate whether a given regime's learned weights are trustworthy --
        see the constructor comment on `regime_sample_count`."""
        return self.regime_sample_count.get(regime, 0)

    # ---- validation gate --------------------------------------------------

    def _record_return(self, regime: str, return_pct: float) -> None:
        """Record a trade return for validation metrics."""
        if regime not in self.regime_returns:
            self.regime_returns[regime] = []
        self.regime_returns[regime].append(return_pct)
        # Keep only recent returns (max 500 per regime)
        if len(self.regime_returns[regime]) > 500:
            self.regime_returns[regime] = self.regime_returns[regime][-500:]

    def _compute_sharpe(self, returns: List[float]) -> float:
        """Compute Sharpe ratio from returns (simple, not annualized)."""
        if len(returns) < 2:
            return 0.0
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 0.0
        if std_r == 0:
            return 0.0
        return mean_r / std_r

    def _compute_win_rate(self, returns: List[float]) -> float:
        """Compute win rate from returns."""
        if not returns:
            return 0.0
        wins = sum(1 for r in returns if r > 0)
        return wins / len(returns)

    def is_regime_validated(self, regime: str) -> bool:
        """Check if a regime has passed the validation gate.
        
        Uses holdout validation: last VALIDATION_HOLDOUT_FRACTION of trades
        must have positive Sharpe and win-rate above threshold.
        """
        returns = self.regime_returns.get(regime, [])
        min_trades = VALIDATION_MIN_TRADES
        
        if len(returns) < min_trades:
            return False
        
        # Use holdout portion for validation
        holdout_start = int(len(returns) * (1 - VALIDATION_HOLDOUT_FRACTION))
        holdout_returns = returns[holdout_start:]
        
        if len(holdout_returns) < 5:  # Need at least 5 holdout trades
            return False
        
        sharpe = self._compute_sharpe(holdout_returns)
        win_rate = self._compute_win_rate(holdout_returns)
        
        validated = sharpe >= VALIDATION_MIN_SHARPE and win_rate >= VALIDATION_MIN_WIN_RATE
        self.regime_validated[regime] = validated
        
        if validated:
            logger.info(f"Regime '{regime}' PASSED validation: Sharpe={sharpe:.3f}, WinRate={win_rate:.3f} (n={len(holdout_returns)})")
        else:
            logger.debug(f"Regime '{regime}' not validated: Sharpe={sharpe:.3f}, WinRate={win_rate:.3f} (n={len(holdout_returns)})")
        
        return validated

    def get_regime_validation_metrics(self, regime: str) -> Dict[str, Any]:
        """Get validation metrics for a regime (for observability)."""
        returns = self.regime_returns.get(regime, [])
        if not returns:
            return {"validated": False, "sharpe": 0.0, "win_rate": 0.0, "n_trades": 0, "n_holdout": 0}
        
        holdout_start = int(len(returns) * (1 - VALIDATION_HOLDOUT_FRACTION))
        holdout_returns = returns[holdout_start:]
        
        return {
            "validated": self.regime_validated.get(regime, False),
            "sharpe": self._compute_sharpe(holdout_returns),
            "win_rate": self._compute_win_rate(holdout_returns),
            "n_trades": len(returns),
            "n_holdout": len(holdout_returns),
            "all_sharpe": self._compute_sharpe(returns),
            "all_win_rate": self._compute_win_rate(returns),
        }

    # ---- persistence ----------------------------------------------------

    def to_state(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "last_update": self.last_update,
            "sample_count": self.sample_count,
            "regime_sample_count": self.regime_sample_count,
            "learning_rate": self.learning_rate,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "weights": self.weights,
            "regime_returns": self.regime_returns,
            "regime_validated": self.regime_validated,
            "performance": {
                regime: {
                    name: {
                        "n_updates": p.n_updates,
                        "n_correct": p.n_correct,
                        "n_incorrect": p.n_incorrect,
                        "cumulative_reward": p.cumulative_reward,
                    }
                    for name, p in brains.items()
                }
                for regime, brains in self.performance.items()
            },
        }

    def save(self, path: Optional[str] = None) -> None:
        target = path or self.state_path
        if not target:
            return
        self.timestamp = datetime.now(timezone.utc).isoformat()
        directory = os.path.dirname(os.path.abspath(target))
        os.makedirs(directory, exist_ok=True)
        # Atomic write: temp file in the same dir, then os.replace.
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".adaptive_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.to_state(), fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, target)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def _save_safely(self) -> None:
        if not self.state_path:
            return
        try:
            self.save()
        except Exception as e:  # never let persistence break the learner
            logger.warning(f"Adaptive state save failed (continuing in-memory): {e}")

    def load(self, path: Optional[str] = None) -> None:
        """Load state from JSON; fall back to equal weights on missing/corrupt."""
        target = path or self.state_path
        if not target or not os.path.exists(target):
            self._reset_to_cold_start()
            return
        try:
            with open(target, encoding="utf-8") as fh:
                state = json.load(fh)
            if not isinstance(state, dict) or "weights" not in state:
                raise ValueError("state missing required fields")
            self.version = int(state.get("version", STATE_VERSION))
            self.timestamp = str(state.get("timestamp", self.timestamp))
            self.last_update = state.get("last_update")
            self.sample_count = int(state.get("sample_count", 0))
            raw_regime_counts = state.get("regime_sample_count", {})
            self.regime_sample_count = {
                str(k): int(v) for k, v in raw_regime_counts.items()
            } if isinstance(raw_regime_counts, dict) else {}
            self.learning_rate = float(state.get("learning_rate", self.learning_rate))
            self.min_weight = float(state.get("min_weight", self.min_weight))
            self.max_weight = float(state.get("max_weight", self.max_weight))
            raw_weights = state.get("weights", {})
            if not isinstance(raw_weights, dict):
                raise ValueError("weights is not a mapping")
            self.weights = {
                str(regime): {str(b): float(w) for b, w in vec.items()}
                for regime, vec in raw_weights.items()
                if isinstance(vec, dict)
            }
            # Load validation tracking fields (backward compatible)
            raw_returns = state.get("regime_returns", {})
            if isinstance(raw_returns, dict):
                self.regime_returns = {
                    str(regime): [float(r) for r in rets if isinstance(r, (int, float))]
                    for regime, rets in raw_returns.items()
                    if isinstance(rets, list)
                }
            else:
                self.regime_returns = {}
            
            raw_validated = state.get("regime_validated", {})
            if isinstance(raw_validated, dict):
                self.regime_validated = {
                    str(regime): bool(v) for regime, v in raw_validated.items()
                }
            else:
                self.regime_validated = {}
            
            self.performance = {}
            for regime, brains in (state.get("performance", {}) or {}).items():
                if not isinstance(brains, dict):
                    continue
                self.performance[regime] = {
                    name: BrainPerformance(
                        name=name,
                        regime=regime,
                        n_updates=int(p.get("n_updates", 0)),
                        n_correct=int(p.get("n_correct", 0)),
                        n_incorrect=int(p.get("n_incorrect", 0)),
                        cumulative_reward=float(p.get("cumulative_reward", 0.0)),
                    )
                    for name, p in brains.items()
                    if isinstance(p, dict)
                }
        except Exception as e:
            logger.warning(f"Adaptive state load failed ({e}); falling back to equal weights.")
            self._reset_to_cold_start()

    def _reset_to_cold_start(self) -> None:
        self.version = STATE_VERSION
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.last_update = None
        self.sample_count = 0
        self.regime_sample_count = {}
        self.weights = {}
        self.performance = {}
        self.regime_returns = {}
        self.regime_validated = {}

    # ---- observability --------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Lightweight view for metrics/telemetry."""
        drift_info = {}
        for regime, weights in self.weights.items():
            drift, l2 = self._compute_drift_vs_equal(weights)
            drift_info[regime] = {"drift": drift, "l2_norm": l2}
        
        return {
            "version": self.version,
            "sample_count": self.sample_count,
            "last_update": self.last_update,
            "weights": {r: dict(v) for r, v in self.weights.items()},
            "validation": {
                regime: self.get_regime_validation_metrics(regime)
                for regime in self.regime_returns.keys()
            },
            "drift_vs_equal": drift_info,
        }
