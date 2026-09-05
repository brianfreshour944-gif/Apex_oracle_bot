"""Population-Based Training (PBT) for Hyperparameter Evolution.

Replaces Online Bayesian Optimization with PBT for non-stationary environments.
Based on: "Population Based Training of Neural Networks" (Jaderberg et al., 2017)
Used by DeepMind for AlphaStar, AlphaZero hyperparameter tuning.

Key advantages over BO:
- Handles non-stationary rewards (market regime changes)
- Parallel exploration of hyperparameter space
- Automatic exploitation of best performers
- Continuous adaptation without retraining from scratch
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import settings
from src.logging_config import get_logger

logger = get_logger("pbt")

# Paths
PBT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'pbt')
PBT_STATE_PATH = os.path.join(PBT_DIR, 'pbt_state.json')
PBT_HISTORY_PATH = os.path.join(PBT_DIR, 'pbt_history.jsonl')


@dataclass
class WorkerConfig:
    """Configuration for a PBT worker."""
    worker_id: int = -1
    # Model hyperparameters
    learning_rate: float = 0.1
    min_weight: float = 0.02
    max_weight: float = 0.60
    entropy_penalty: float = 0.3
    size_mult_min: float = 0.5
    size_mult_max: float = 1.75
    threshold_base: float = 0.15
    sideways_mult: float = 0.5
    # Risk hyperparameters
    base_risk_percent: float = 0.01
    stop_loss_pct: float = 0.02
    trailing_activation: float = 0.015
    trailing_distance: float = 0.015
    max_portfolio_pct: float = 0.5
    # Transformer hyperparameters
    ensemble_size: int = 5
    mc_passes: int = 20
    dropout: float = 0.1
    uncertainty_penalty: float = 1.0
    
    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkerConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def mutate(self, mutation_rate: float = 0.2, mutation_strength: float = 0.2) -> None:
        """Apply random mutations to hyperparameters."""
        for field_name in self.__dataclass_fields__:
            if field_name == 'worker_id':
                continue
            value = getattr(self, field_name)
            if isinstance(value, float):
                # Log-uniform mutation for positive values
                if value > 0:
                    log_val = np.log(value)
                    log_val += np.random.normal(0, mutation_strength)
                    setattr(self, field_name, float(np.exp(log_val)))
                else:
                    # Additive mutation for values that can be negative/zero
                    setattr(self, field_name, value + np.random.normal(0, abs(value) * mutation_strength + 1e-6))
            elif isinstance(value, int):
                # Discrete mutation
                delta = int(np.random.normal(0, max(1, value * mutation_strength)))
                setattr(self, field_name, max(1, value + delta))
    
    def clip(self) -> None:
        """Clip hyperparameters to valid ranges."""
        self.learning_rate = float(np.clip(self.learning_rate, 0.001, 0.5))
        self.min_weight = float(np.clip(self.min_weight, 0.001, 0.2))
        self.max_weight = float(np.clip(self.max_weight, 0.2, 0.9))
        self.entropy_penalty = float(np.clip(self.entropy_penalty, 0.05, 0.5))
        self.size_mult_min = float(np.clip(self.size_mult_min, 0.2, 0.8))
        self.size_mult_max = float(np.clip(self.size_mult_max, 1.0, 3.0))
        self.threshold_base = float(np.clip(self.threshold_base, 0.05, 0.3))
        self.sideways_mult = float(np.clip(self.sideways_mult, 0.2, 0.8))
        self.base_risk_percent = float(np.clip(self.base_risk_percent, 0.001, 0.05))
        self.stop_loss_pct = float(np.clip(self.stop_loss_pct, 0.005, 0.05))
        self.trailing_activation = float(np.clip(self.trailing_activation, 0.005, 0.05))
        self.trailing_distance = float(np.clip(self.trailing_distance, 0.005, 0.05))
        self.max_portfolio_pct = float(np.clip(self.max_portfolio_pct, 0.2, 0.9))
        self.ensemble_size = int(np.clip(self.ensemble_size, 3, 10))
        self.mc_passes = int(np.clip(self.mc_passes, 5, 50))
        self.dropout = float(np.clip(self.dropout, 0.01, 0.5))
        self.uncertainty_penalty = float(np.clip(self.uncertainty_penalty, 0.1, 5.0))


@dataclass
class Worker:
    """A PBT worker with its own model, config, and performance history."""
    worker_id: int
    config: WorkerConfig
    # Performance tracking
    recent_rewards: list[float] = field(default_factory=list)
    recent_returns: list[float] = field(default_factory=list)
    recent_sharpes: list[float] = field(default_factory=list)
    # Metadata
    created_at: float = field(default_factory=time.time)
    last_exploit_at: float = 0.0
    exploit_count: int = 0
    mutation_count: int = 0
    
    @property
    def performance(self) -> float:
        """Compute performance score (Sharpe * sqrt(n) for sample efficiency)."""
        if len(self.recent_returns) < 5:
            return -float('inf')
        returns = np.array(self.recent_returns[-50:])
        if len(returns) < 2 or np.std(returns) == 0:
            return float(np.mean(returns))
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
        # Reward consistency: Sharpe * sqrt(effective_samples)
        effective_n = min(len(returns), 50)
        return float(sharpe * np.sqrt(effective_n))
    
    def is_ready_for_exploit(self, min_samples: int = 20) -> bool:
        return len(self.recent_returns) >= min_samples
    
    def record_reward(self, reward: float, ret: float = 0.0) -> None:
        """Record a reward signal."""
        self.recent_rewards.append(reward)
        self.recent_returns.append(ret)
        if len(self.recent_rewards) > 200:
            self.recent_rewards = self.recent_rewards[-200:]
            self.recent_returns = self.recent_returns[-200:]
            self.recent_sharpes = self.recent_sharpes[-200:]


class PopulationTrainer:
    """Population-Based Training coordinator.
    
    Maintains a population of workers, periodically exploits best performers
    to replace worst performers, then mutates.
    """
    
    def __init__(
        self,
        population_size: int = 16,
        exploit_fraction: float = 0.25,  # Bottom 25% get replaced
        exploit_interval: int = 10,       # Every N decisions
        min_samples_for_exploit: int = 20,
        mutation_rate: float = 0.2,
        mutation_strength: float = 0.2,
        state_path: str | None = None,
    ):
        self.population_size = population_size
        self.exploit_fraction = exploit_fraction
        self.exploit_interval = exploit_interval
        self.min_samples_for_exploit = min_samples_for_exploit
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.state_path = state_path or PBT_STATE_PATH
        
        self.workers: list[Worker] = []
        self.timestep = 0
        self._lock = threading.Lock()
        self._initialized = False
        
        if self.state_path:
            self.load()
    
    def initialize_population(self, base_config: WorkerConfig | None = None) -> None:
        """Create initial population with random hyperparameters."""
        with self._lock:
            if self.workers:
                return
            
            if base_config is None:
                base_config = WorkerConfig()
            
            for i in range(self.population_size):
                config = copy.deepcopy(base_config)
                config.worker_id = i
                config.mutate(mutation_rate=1.0, mutation_strength=0.5)  # Wide initial diversity
                config.clip()
                worker = Worker(worker_id=i, config=config)
                self.workers.append(worker)
            
            self._initialized = True
            logger.info(f"PBT: Initialized population of {self.population_size} workers")
            self._save_safely()
    
    def step(
        self, 
        worker_rewards: dict[int, tuple[float, float]]  # worker_id -> (reward, return)
    ) -> dict[str, Any] | None:
        """Advance PBT by one step.
        
        Args:
            worker_rewards: Dict mapping worker_id to (reward, return) for this step
            
        Returns:
            Best hyperparameters to apply to live bot, or None if no exploit occurred
        """
        with self._lock:
            if not self._initialized:
                self.initialize_population()
            
            # Record rewards
            for worker_id, (reward, ret) in worker_rewards.items():
                if 0 <= worker_id < len(self.workers):
                    self.workers[worker_id].record_reward(reward, ret)
            
            self.timestep += 1
            
            # Check if it's time to exploit
            if self.timestep % self.exploit_interval != 0:
                return None
            
            # Filter workers ready for exploit
            ready_workers = [w for w in self.workers if w.is_ready_for_exploit(self.min_samples_for_exploit)]
            if len(ready_workers) < 4:  # Need at least 4 workers for meaningful exploit
                return None
            
            # Sort by performance
            ready_workers.sort(key=lambda w: w.performance, reverse=True)
            
            n_exploit = max(1, int(len(ready_workers) * self.exploit_fraction))
            best_workers = ready_workers[:n_exploit]
            worst_workers = ready_workers[-n_exploit:]
            
            if not worst_workers:
                return None
            
            best_worker = best_workers[0]
            logger.info(f"PBT: Exploiting at timestep {self.timestep}. Best perf: {best_worker.performance:.4f}, "
                       f"Worst perf: {worst_workers[0].performance:.4f}")
            
            # Exploit: worst workers copy best worker's config
            for worst in worst_workers:
                # Copy model weights would happen here in full implementation
                # For now, copy hyperparameters
                worst.config = copy.deepcopy(best_worker.config)
                worst.config.worker_id = worst.worker_id
                worst.config.mutate(self.mutation_rate, self.mutation_strength)
                worst.config.clip()
                worst.mutation_count += 1
                worst.last_exploit_at = time.time()
                worst.exploit_count += 1
                
                # Reset performance history for fair comparison after exploit
                worst.recent_rewards = []
                worst.recent_returns = []
                worst.recent_sharpes = []
            
            best_worker.exploit_count += 1
            
            self._save_safely()
            
            # Return best hyperparameters for live application
            return {
                'source': 'pbt',
                'timestep': self.timestep,
                'best_worker_id': best_worker.worker_id,
                'best_performance': best_worker.performance,
                'hyperparameters': best_worker.config.to_dict(),
            }
    
    def get_best_config(self) -> WorkerConfig:
        """Get the best worker's configuration."""
        with self._lock:
            if not self.workers:
                return WorkerConfig()
            ready = [w for w in self.workers if w.is_ready_for_exploit(self.min_samples_for_exploit)]
            if ready:
                return max(ready, key=lambda w: w.performance).config
            return max(self.workers, key=lambda w: w.performance).config
    
    def get_population_stats(self) -> dict[str, Any]:
        """Get statistics about the population."""
        with self._lock:
            if not self.workers:
                return {}
            
            perfs = [w.performance for w in self.workers]
            return {
                'population_size': len(self.workers),
                'timestep': self.timestep,
                'mean_performance': float(np.mean(perfs)),
                'max_performance': float(np.max(perfs)),
                'min_performance': float(np.min(perfs)),
                'std_performance': float(np.std(perfs)),
                'best_worker_id': int(np.argmax(perfs)),
                'workers': [
                    {
                        'id': w.worker_id,
                        'performance': w.performance,
                        'n_samples': len(w.recent_rewards),
                        'exploit_count': w.exploit_count,
                        'mutation_count': w.mutation_count,
                        'config': w.config.to_dict(),
                    }
                    for w in self.workers
                ],
            }
    
    def apply_best_to_live(self, live_components: dict[str, Any]) -> None:
        """Apply best hyperparameters to live trading components."""
        best_config = self.get_best_config()
        
        # Apply to meta-learner
        if 'meta_learner' in live_components:
            ml = live_components['meta_learner']
            ml.learning_rate = best_config.learning_rate
            ml.min_weight = best_config.min_weight
            ml.max_weight = best_config.max_weight
        
        # Apply to committee settings (would need access to committee module)
        if 'committee' in live_components:
            # These would be module-level settings
            pass
        
        # Apply to risk manager
        if 'risk_manager' in live_components:
            rm = live_components['risk_manager']
            rm.base_risk_percent = best_config.base_risk_percent
            # Note: stop_loss_pct etc are typically in settings, not runtime
        
        logger.info(f"PBT: Applied best config from worker {best_config.worker_id} to live components")
    
    def _save_safely(self) -> None:
        if not self.state_path:
            return
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            state = {
                'population_size': self.population_size,
                'exploit_fraction': self.exploit_fraction,
                'exploit_interval': self.exploit_interval,
                'min_samples_for_exploit': self.min_samples_for_exploit,
                'mutation_rate': self.mutation_rate,
                'mutation_strength': self.mutation_strength,
                'timestep': self.timestep,
                'workers': [
                    {
                        'worker_id': w.worker_id,
                        'config': w.config.to_dict(),
                        'recent_rewards': w.recent_rewards,
                        'recent_returns': w.recent_returns,
                        'created_at': w.created_at,
                        'last_exploit_at': w.last_exploit_at,
                        'exploit_count': w.exploit_count,
                        'mutation_count': w.mutation_count,
                    }
                    for w in self.workers
                ],
            }
            import tempfile
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(self.state_path), prefix=".pbt_", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
                os.replace(tmp_path, self.state_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
        except Exception as e:
            logger.warning(f"PBT state save failed: {e}")
    
    def load(self) -> None:
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                state = json.load(f)
            
            self.population_size = state.get('population_size', self.population_size)
            self.exploit_fraction = state.get('exploit_fraction', self.exploit_fraction)
            self.exploit_interval = state.get('exploit_interval', self.exploit_interval)
            self.min_samples_for_exploit = state.get('min_samples_for_exploit', self.min_samples_for_exploit)
            self.mutation_rate = state.get('mutation_rate', self.mutation_rate)
            self.mutation_strength = state.get('mutation_strength', self.mutation_strength)
            self.timestep = state.get('timestep', 0)
            
            self.workers = []
            for w_data in state.get('workers', []):
                config = WorkerConfig.from_dict(w_data['config'])
                worker = Worker(
                    worker_id=w_data['worker_id'],
                    config=config,
                    recent_rewards=w_data.get('recent_rewards', []),
                    recent_returns=w_data.get('recent_returns', []),
                    created_at=w_data.get('created_at', time.time()),
                    last_exploit_at=w_data.get('last_exploit_at', 0.0),
                    exploit_count=w_data.get('exploit_count', 0),
                    mutation_count=w_data.get('mutation_count', 0),
                )
                self.workers.append(worker)
            
            self._initialized = True
            logger.info(f"PBT: Loaded population of {len(self.workers)} workers at timestep {self.timestep}")
        except Exception as e:
            logger.warning(f"PBT load failed: {e}")


# Global instance
_pbt_trainer: PopulationTrainer | None = None


def get_pbt_trainer() -> PopulationTrainer:
    """Get or create the global PBT trainer."""
    global _pbt_trainer
    if _pbt_trainer is None:
        _pbt_trainer = PopulationTrainer(
            population_size=getattr(settings, 'PBT_POPULATION_SIZE', 16),
            exploit_fraction=0.25,
            exploit_interval=10,
            min_samples_for_exploit=20,
            mutation_rate=0.2,
            mutation_strength=0.2,
            state_path=os.path.join(PBT_DIR, 'pbt_state.json'),
        )
    return _pbt_trainer


def reset_pbt_trainer() -> None:
    global _pbt_trainer
    _pbt_trainer = None


async def run_pbt_cycle(live_components: dict[str, Any]) -> None:
    """Run one PBT optimization cycle."""
    try:
        trainer = get_pbt_trainer()
        
        # In a real implementation, this would collect rewards from all workers
        # For now, simulate by evaluating current performance
        # This would need integration with the actual trading loop
        
        # Apply best config to live components
        trainer.apply_best_to_live(live_components)
        
    except Exception as e:
        logger.error(f"PBT cycle failed: {e}")


if __name__ == "__main__":
    # Quick test
    print("Testing PBT...")
    trainer = PopulationTrainer(population_size=8)
    trainer.initialize_population()
    
    # Simulate some steps
    for step in range(50):
        rewards = {}
        for i in range(8):
            # Simulate reward: some workers better than others
            base_perf = 1.0 + i * 0.1  # Worker 7 is best
            reward = base_perf + np.random.normal(0, 0.5)
            ret = base_perf * 0.01 + np.random.normal(0, 0.02)
            rewards[i] = (reward, ret)
        
        result = trainer.step(rewards)
        if result:
            print(f"Step {step}: Exploited! Best worker {result['best_worker_id']} perf={result['best_performance']:.4f}")
    
    stats = trainer.get_population_stats()
    print(f"Final stats: best={stats['max_performance']:.4f}, mean={stats['mean_performance']:.4f}")