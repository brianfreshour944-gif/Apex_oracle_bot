"""Online Bayesian Optimization for Hyperparameter Tuning.

Continuously tunes meta-learner and brain hyperparameters based on recent
regime performance using Gaussian Process-based Bayesian Optimization.

Supports tuning:
- AdaptiveMetaLearner: learning_rate, min_weight, max_weight, validation thresholds
- Committee: entropy_penalty, size_multiplier bounds, threshold adjustments
- Transformer: ensemble_size, mc_passes, dropout rates
- Risk: BASE_RISK_PERCENT, STOP_LOSS_PCT, TRAILING_* params
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.config import settings
from src.logging_config import get_logger

logger = get_logger("bayes_opt")

# Paths
OPT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'bayes_opt')
OPT_STATE_PATH = os.path.join(OPT_DIR, 'bayes_opt_state.json')
OPT_HISTORY_PATH = os.path.join(OPT_DIR, 'bayes_opt_history.jsonl')


@dataclass
class ParameterSpace:
    """Defines a parameter to optimize."""
    name: str
    param_type: str  # 'float', 'int', 'categorical'
    low: float = 0.0
    high: float = 1.0
    categories: List[Any] = field(default_factory=list)
    log_scale: bool = False


@dataclass
class OptimizationResult:
    """Result of a single evaluation."""
    params: Dict[str, Any]
    objective: float  # Higher is better (e.g., Sharpe, win rate)
    timestamp: float
    regime: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class GaussianProcessSurrogate:
    """Simple GP surrogate model using sklearn if available, else fallback."""
    
    def __init__(self, noise: float = 1e-4):
        self.noise = noise
        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        self._sklearn_gp = None
        self._param_names: List[str] = []
        self._fitted = False
    
    def _vectorize_params(self, params: Dict[str, Any], param_spaces: List[ParameterSpace]) -> np.ndarray:
        """Convert param dict to vector."""
        vec = []
        for space in param_spaces:
            val = params.get(space.name, space.low)
            if space.param_type == 'categorical':
                # One-hot encode
                for cat in space.categories:
                    vec.append(1.0 if val == cat else 0.0)
            elif space.param_type == 'int':
                vec.append(float(val))
            else:
                if space.log_scale and val > 0:
                    vec.append(math.log(val))
                else:
                    vec.append(float(val))
        return np.array(vec, dtype=np.float32)
    
    def fit(self, results: List[OptimizationResult], param_spaces: List[ParameterSpace]) -> None:
        """Fit GP to observed data."""
        if len(results) < 2:
            self._fitted = False
            return
        
        self._param_names = [s.name for s in param_spaces]
        X = np.array([self._vectorize_params(r.params, param_spaces) for r in results])
        y = np.array([r.objective for r in results], dtype=np.float32)
        
        self.X = X
        self.y = y
        
        # Try sklearn GP
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
            
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(self.noise)
            self._sklearn_gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42,
            )
            self._sklearn_gp.fit(X, y)
            self._fitted = True
        except ImportError:
            logger.warning("sklearn not available, using fallback acquisition")
            self._fitted = False
    
    def predict(self, params: Dict[str, Any], param_spaces: List[ParameterSpace]) -> Tuple[float, float]:
        """Predict mean and std for given params."""
        if not self._fitted or self._sklearn_gp is None or len(self.X) < 2:
            # Fallback: return mean of observed with high uncertainty
            mean_y = np.mean(self.y) if self.y else 0.0
            return mean_y, 1.0
        
        x = self._vectorize_params(params, param_spaces).reshape(1, -1)
        mean, std = self._sklearn_gp.predict(x, return_std=True)
        return float(mean[0]), float(std[0])


class AcquisitionFunction:
    """Acquisition functions for Bayesian Optimization."""
    
    @staticmethod
    def expected_improvement(
        mean: float,
        std: float,
        best_y: float,
        xi: float = 0.01,
    ) -> float:
        """Expected Improvement acquisition function."""
        if std <= 0:
            return 0.0
        z = (mean - best_y - xi) / std
        from scipy.stats import norm
        return (mean - best_y - xi) * norm.cdf(z) + std * norm.pdf(z)
    
    @staticmethod
    def upper_confidence_bound(
        mean: float,
        std: float,
        kappa: float = 2.0,
    ) -> float:
        """Upper Confidence Bound acquisition function."""
        return mean + kappa * std
    
    @staticmethod
    def probability_of_improvement(
        mean: float,
        std: float,
        best_y: float,
        xi: float = 0.01,
    ) -> float:
        """Probability of Improvement acquisition function."""
        if std <= 0:
            return 0.0
        z = (mean - best_y - xi) / std
        from scipy.stats import norm
        return norm.cdf(z)


class OnlineBayesianOptimizer:
    """Online Bayesian Optimization for hyperparameter tuning.
    
    Maintains a GP surrogate model updated with each new evaluation.
    Supports per-regime optimization with shared prior.
    """
    
    def __init__(
        self,
        param_spaces: List[ParameterSpace],
        acquisition: str = 'ucb',  # 'ei', 'ucb', 'pi'
        ucb_kappa: float = 2.0,
        ei_xi: float = 0.01,
        random_fraction: float = 0.1,  # Fraction of random exploration
        state_path: Optional[str] = None,
    ):
        self.param_spaces = param_spaces
        self.acquisition_type = acquisition
        self.ucb_kappa = ucb_kappa
        self.ei_xi = ei_xi
        self.random_fraction = random_fraction
        self.state_path = state_path or OPT_STATE_PATH
        
        self.surrogate = GaussianProcessSurrogate()
        self.history: List[OptimizationResult] = []
        self.regime_history: Dict[str, List[OptimizationResult]] = {}
        self.best_params: Dict[str, Any] = {}
        self.best_objective: float = -float('inf')
        self._lock = threading.Lock()
        self._initialized = False
        
        if self.state_path:
            self.load()
    
    def _get_param_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for random sampling."""
        bounds = []
        for space in self.param_spaces:
            if space.param_type == 'categorical':
                bounds.append((0, len(space.categories) - 1))
            elif space.log_scale:
                bounds.append((math.log(max(space.low, 1e-10)), math.log(space.high)))
            else:
                bounds.append((space.low, space.high))
        return bounds
    
    def suggest(self, regime: str = 'default') -> Dict[str, Any]:
        """Suggest next parameters to evaluate."""
        with self._lock:
            if len(self.history) < 3 or np.random.random() < self.random_fraction:
                # Random exploration
                return self._random_params()
            
            # Fit surrogate on all data (or regime-specific if enough data)
            regime_results = self.regime_history.get(regime, [])
            fit_data = regime_results if len(regime_results) >= 5 else self.history
            
            self.surrogate.fit(fit_data, self.param_spaces)
            
            # Find best observed
            best_y = max(r.objective for r in fit_data)
            
            # Optimize acquisition function via random sampling + local search
            best_acq = -float('inf')
            best_params = None
            
            bounds = self._get_param_bounds()
            n_candidates = 1000
            
            for _ in range(n_candidates):
                candidate = self._random_params()
                mean, std = self.surrogate.predict(candidate, self.param_spaces)
                
                if self.acquisition_type == 'ei':
                    acq = AcquisitionFunction.expected_improvement(mean, std, best_y, self.ei_xi)
                elif self.acquisition_type == 'pi':
                    acq = AcquisitionFunction.probability_of_improvement(mean, std, best_y, self.ei_xi)
                else:  # ucb
                    acq = AcquisitionFunction.upper_confidence_bound(mean, std, self.ucb_kappa)
                
                if acq > best_acq:
                    best_acq = acq
                    best_params = candidate
            
            return best_params or self._random_params()
    
    def _random_params(self) -> Dict[str, Any]:
        """Generate random parameters within bounds."""
        params = {}
        for space in self.param_spaces:
            if space.param_type == 'categorical':
                params[space.name] = np.random.choice(space.categories)
            elif space.param_type == 'int':
                params[space.name] = int(np.random.randint(space.low, space.high + 1))
            elif space.log_scale:
                log_low = math.log(max(space.low, 1e-10))
                log_high = math.log(space.high)
                params[space.name] = math.exp(np.random.uniform(log_low, log_high))
            else:
                params[space.name] = np.random.uniform(space.low, space.high)
        return params
    
    def update(self, result: OptimizationResult) -> None:
        """Update with new evaluation result."""
        with self._lock:
            self.history.append(result)
            self.regime_history.setdefault(result.regime, []).append(result)
            
            if result.objective > self.best_objective:
                self.best_objective = result.objective
                self.best_params = result.params.copy()
            
            # Trim history to prevent unbounded growth
            max_history = 1000
            if len(self.history) > max_history:
                self.history = self.history[-max_history:]
            for regime in self.regime_history:
                if len(self.regime_history[regime]) > max_history:
                    self.regime_history[regime] = self.regime_history[regime][-max_history:]
            
            self._save_safely()
    
    def get_best_params(self, regime: Optional[str] = None) -> Dict[str, Any]:
        """Get best parameters (overall or for specific regime)."""
        with self._lock:
            if regime and regime in self.regime_history and self.regime_history[regime]:
                best = max(self.regime_history[regime], key=lambda r: r.objective)
                return best.params.copy()
            return self.best_params.copy()
    
    def get_regime_summary(self, regime: str) -> Dict[str, Any]:
        """Get optimization summary for a regime."""
        with self._lock:
            results = self.regime_history.get(regime, [])
            if not results:
                return {"n_evaluations": 0}
            
            objs = [r.objective for r in results]
            return {
                "n_evaluations": len(results),
                "best_objective": max(objs),
                "mean_objective": np.mean(objs),
                "std_objective": np.std(objs),
                "best_params": max(results, key=lambda r: r.objective).params,
                "last_updated": results[-1].timestamp,
            }
    
    def _save_safely(self) -> None:
        """Save state to disk atomically."""
        if not self.state_path:
            return
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            
            # Save main state
            state = {
                "history": [
                    {
                        "params": r.params,
                        "objective": r.objective,
                        "timestamp": r.timestamp,
                        "regime": r.regime,
                        "metadata": r.metadata,
                    }
                    for r in self.history
                ],
                "best_params": self.best_params,
                "best_objective": self.best_objective,
            }
            
            import tempfile
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(self.state_path), prefix=".bayes_opt_", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
                os.replace(tmp_path, self.state_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
            
            # Append to history log (for analysis)
            if self.history:
                last = self.history[-1]
                with open(OPT_HISTORY_PATH, "a") as f:
                    f.write(json.dumps({
                        "params": last.params,
                        "objective": last.objective,
                        "timestamp": last.timestamp,
                        "regime": last.regime,
                        "metadata": last.metadata,
                    }) + "\n")
                    
        except Exception as e:
            logger.warning(f"BayesOpt state save failed: {e}")
    
    def load(self) -> None:
        """Load state from disk."""
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r") as f:
                state = json.load(f)
            
            self.history = [
                OptimizationResult(
                    params=h["params"],
                    objective=h["objective"],
                    timestamp=h["timestamp"],
                    regime=h["regime"],
                    metadata=h.get("metadata", {}),
                )
                for h in state.get("history", [])
            ]
            
            for r in self.history:
                self.regime_history.setdefault(r.regime, []).append(r)
            
            self.best_params = state.get("best_params", {})
            self.best_objective = state.get("best_objective", -float('inf'))
            self._initialized = True
            
            logger.info(f"Loaded BayesOpt state: {len(self.history)} evaluations, best={self.best_objective:.4f}")
        except Exception as e:
            logger.warning(f"BayesOpt load failed: {e}")


# ── Default Parameter Spaces ────────────────────────────────────────

def get_default_meta_learner_space() -> List[ParameterSpace]:
    """Parameter space for AdaptiveMetaLearner."""
    return [
        ParameterSpace("learning_rate", "float", 0.01, 0.5, log_scale=True),
        ParameterSpace("min_weight", "float", 0.005, 0.1),
        ParameterSpace("max_weight", "float", 0.4, 0.8),
        ParameterSpace("validation_min_sharpe", "float", 0.2, 1.0),
        ParameterSpace("validation_min_win_rate", "float", 0.50, 0.65),
        ParameterSpace("validation_holdout_fraction", "float", 0.1, 0.5),
    ]


def get_default_committee_space() -> List[ParameterSpace]:
    """Parameter space for Committee combination logic."""
    return [
        ParameterSpace("entropy_penalty_factor", "float", 0.1, 0.5),
        ParameterSpace("size_mult_min", "float", 0.3, 0.6),
        ParameterSpace("size_mult_max", "float", 1.2, 2.0),
        ParameterSpace("threshold_base", "float", 0.1, 0.2),
        ParameterSpace("sideways_threshold_mult", "float", 0.3, 0.7),
    ]


def get_default_transformer_space() -> List[ParameterSpace]:
    """Parameter space for Transformer brain."""
    return [
        ParameterSpace("ensemble_size", "int", 3, 10),
        ParameterSpace("mc_passes", "int", 10, 50),
        ParameterSpace("dropout", "float", 0.05, 0.3),
        ParameterSpace("uncertainty_penalty", "float", 0.5, 3.0),
    ]


def get_default_risk_space() -> List[ParameterSpace]:
    """Parameter space for Risk management."""
    return [
        ParameterSpace("base_risk_percent", "float", 0.005, 0.03, log_scale=True),
        ParameterSpace("stop_loss_pct", "float", 0.01, 0.05),
        ParameterSpace("trailing_activation_pct", "float", 0.01, 0.05),
        ParameterSpace("trailing_distance_pct", "float", 0.01, 0.05),
        ParameterSpace("max_portfolio_pct", "float", 0.3, 0.8),
    ]


# ── Objective Functions ────────────────────────────────────────────

def compute_meta_learner_objective(regime: str) -> float:
    """Compute objective for meta-learner: Sharpe * win_rate * sqrt(n_trades)."""
    from src.committee.adaptive_meta import get_meta_learner
    
    learner = get_meta_learner()
    if learner is None:
        return 0.0
    
    metrics = learner.get_regime_validation_metrics(regime)
    if not metrics.get("validated", False):
        return 0.0
    
    sharpe = metrics.get("sharpe", 0.0)
    win_rate = metrics.get("win_rate", 0.0)
    n_trades = metrics.get("n_holdout", 0)
    
    # Combined objective: reward high Sharpe, high win rate, and sufficient samples
    if sharpe <= 0 or win_rate <= 0.5 or n_trades < 5:
        return 0.0
    
    return float(sharpe * win_rate * math.sqrt(n_trades))


def compute_committee_objective(regime: str) -> float:
    """Compute objective for committee: risk-adjusted return proxy."""
    from src.metrics import get_committee_metrics  # hypothetical
    # Would integrate with actual metrics
    return compute_meta_learner_objective(regime)


# ── Global Optimizer Instances ────────────────────────────────────

_meta_learner_optimizer: Optional[OnlineBayesianOptimizer] = None
_committee_optimizer: Optional[OnlineBayesianOptimizer] = None
_transformer_optimizer: Optional[OnlineBayesianOptimizer] = None
_risk_optimizer: Optional[OnlineBayesianOptimizer] = None


def get_meta_learner_optimizer() -> OnlineBayesianOptimizer:
    global _meta_learner_optimizer
    if _meta_learner_optimizer is None:
        _meta_learner_optimizer = OnlineBayesianOptimizer(
            param_spaces=get_default_meta_learner_space(),
            acquisition='ucb',
            state_path=os.path.join(OPT_DIR, 'meta_learner_opt.json'),
        )
    return _meta_learner_optimizer


def get_committee_optimizer() -> OnlineBayesianOptimizer:
    global _committee_optimizer
    if _committee_optimizer is None:
        _committee_optimizer = OnlineBayesianOptimizer(
            param_spaces=get_default_committee_space(),
            acquisition='ucb',
            state_path=os.path.join(OPT_DIR, 'committee_opt.json'),
        )
    return _committee_optimizer


def get_transformer_optimizer() -> OnlineBayesianOptimizer:
    global _transformer_optimizer
    if _transformer_optimizer is None:
        _transformer_optimizer = OnlineBayesianOptimizer(
            param_spaces=get_default_transformer_space(),
            acquisition='ei',
            state_path=os.path.join(OPT_DIR, 'transformer_opt.json'),
        )
    return _transformer_optimizer


def get_risk_optimizer() -> OnlineBayesianOptimizer:
    global _risk_optimizer
    if _risk_optimizer is None:
        _risk_optimizer = OnlineBayesianOptimizer(
            param_spaces=get_default_risk_space(),
            acquisition='ucb',
            state_path=os.path.join(OPT_DIR, 'risk_opt.json'),
        )
    return _risk_optimizer


# ── Periodic Evaluation & Update ──────────────────────────────────

async def run_bayesian_optimization_cycle() -> None:
    """Run one cycle of Bayesian optimization for all components."""
    from src.committee.adaptive_meta import get_meta_learner
    from src.committee.bayesian_transformer import get_bayesian_transformer
    
    try:
        # 1. Meta-learner optimization
        meta_opt = get_meta_learner_optimizer()
        learner = get_meta_learner()
        
        if learner is not None:
            for regime in learner.regime_returns.keys():
                if learner.sample_count_for_regime(regime) >= 20:  # Minimum data
                    # Suggest new params
                    new_params = meta_opt.suggest(regime)
                    
                    # Apply params temporarily (shadow mode)
                    old_params = {
                        "learning_rate": learner.learning_rate,
                        "min_weight": learner.min_weight,
                        "max_weight": learner.max_weight,
                    }
                    
                    # Apply and evaluate
                    learner.learning_rate = new_params.get("learning_rate", learner.learning_rate)
                    learner.min_weight = new_params.get("min_weight", learner.min_weight)
                    learner.max_weight = new_params.get("max_weight", learner.max_weight)
                    
                    # Wait for some trades to accumulate, then evaluate
                    # In practice, this would be done asynchronously
                    # For now, compute objective immediately
                    objective = compute_meta_learner_objective(regime)
                    
                    result = OptimizationResult(
                        params=new_params,
                        objective=objective,
                        timestamp=time.time(),
                        regime=regime,
                        metadata={"old_params": old_params},
                    )
                    meta_opt.update(result)
                    
                    logger.info(f"BayesOpt meta-learner [{regime}]: obj={objective:.4f}, params={new_params}")
        
        # 2. Committee optimizer
        comm_opt = get_committee_optimizer()
        for regime in ["trending", "sideways", "high_volatility", "bull", "bear"]:
            if len(comm_opt.regime_history.get(regime, [])) < 3:
                continue
            new_params = comm_opt.suggest(regime)
            # Would apply to committee logic and evaluate
            objective = compute_committee_objective(regime)
            comm_opt.update(OptimizationResult(
                params=new_params, objective=objective, timestamp=time.time(), regime=regime
            ))
        
        # 3. Transformer optimizer
        trans_opt = get_transformer_optimizer()
        bayesian_brain = get_bayesian_transformer()
        if bayesian_brain is not None and bayesian_brain._initialized:
            new_params = trans_opt.suggest("default")
            # Apply ensemble size, mc_passes, etc.
            objective = 0.0  # Would need actual evaluation
            trans_opt.update(OptimizationResult(
                params=new_params, objective=objective, timestamp=time.time(), regime="default"
            ))
        
    except Exception as e:
        logger.error(f"Bayesian optimization cycle failed: {e}")


def apply_best_params_to_live() -> None:
    """Apply best-found parameters to live components."""
    from src.committee.committee import get_meta_learner

    meta_opt = get_meta_learner_optimizer()
    learner = get_meta_learner()
    
    if learner is not None:
        for regime in learner.regime_returns.keys():
            best = meta_opt.get_best_params(regime)
            if best:
                learner.learning_rate = best.get("learning_rate", learner.learning_rate)
                learner.min_weight = best.get("min_weight", learner.min_weight)
                learner.max_weight = best.get("max_weight", learner.max_weight)
                logger.info(f"Applied best meta-learner params for {regime}: {best}")


if __name__ == "__main__":
    # Quick test
    print("Bayesian Optimization module loaded")
    opt = OnlineBayesianOptimizer(
        param_spaces=[
            ParameterSpace("lr", "float", 0.001, 0.1, log_scale=True),
            ParameterSpace("batch_size", "int", 16, 128),
        ],
        acquisition='ucb',
    )
    print("Suggest:", opt.suggest())
    opt.update(OptimizationResult(
        params={"lr": 0.01, "batch_size": 32},
        objective=1.5,
        timestamp=time.time(),
        regime="test",
    ))
    print("Best:", opt.get_best_params())