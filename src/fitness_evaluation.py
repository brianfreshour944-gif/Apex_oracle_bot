"""Fitness evaluation framework for candidate strategies.

Implements user's exact pipeline with complexity/instability penalties.
Never promotes solely on raw historical PnL — requires OOS walk-forward
and shadow live validation.
"""

import math
from typing import Any


def compute_fitness(
    oos_return_pct: float,
    max_drawdown_pct: float,
    annual_turnover_pct: float,
    n_parameters: int,
    return_std: float,
    return_mean: float,
    min_edge_bps: float = 10.0,
    # Penalty weights (tunable)
    drawdown_weight: float = 0.05,
    turnover_weight: float = 0.02,
    complexity_weight: float = 0.01,
    instability_weight: float = 0.03,
    # Thresholds
    max_acceptable_dd: float = -15.0,
    max_acceptable_turnover: float = 300.0,
) -> float:
    """
    Fitness = OOS expectancy - penalties.
    
    Higher is better. Negative = reject.
    
    The complexity penalty prevents overfitting to 3-month windows
    with 4 tuned parameters (RSI=27/lookback=17/z=1.83/ATR=2.14).
    """
    # Base expectancy (annualized, from walk-forward windows)
    expectancy = oos_return_pct

    # Drawdown penalty (non-linear: worse past max_acceptable_dd gets heavier).
    # max_drawdown_pct is NEGATIVE (e.g. -20.0). A drawdown deeper than the
    # acceptable floor (-15 default) must penalize: (acceptable - actual) > 0.
    dd_penalty = max(0.0, (max_acceptable_dd - max_drawdown_pct) / 100.0) * drawdown_weight * abs(oos_return_pct)
    
    # Turnover penalty (high turnover = high cost, high risk of edge decay)
    turnover_penalty = max(0.0, annual_turnover_pct - max_acceptable_turnover) / 100.0 * turnover_weight * abs(oos_return_pct)
    
    # Complexity penalty (each tuned param costs ~1% of fitness)
    complexity_penalty = n_parameters * complexity_weight * abs(oos_return_pct)
    
    # Instability penalty (high variance relative to mean = unreliable)
    instability_penalty = (return_std / max(abs(return_mean), 0.001)) * instability_weight if return_mean != 0 else 0.5
    
    fitness = expectancy - dd_penalty - turnover_penalty - complexity_penalty - instability_penalty
    return fitness


def promotion_gate(
    fitness: float,
    shadow_sharpe: float,
    shadow_win_rate: float,
    n_shadow_trades: int,
    min_shadow_trades: int = 30,
    min_shadow_sharpe: float = 0.5,
    min_shadow_win_rate: float = 0.52,
    min_fitness: float = 0.0,
) -> tuple[bool, str]:
    """
    Hard promotion gate — all must pass.
    
    Prevents promoting a strategy solely because it has good historical PnL.
    Requires: shadow performance + positive fitness + sufficient sample.
    """
    if n_shadow_trades < min_shadow_trades:
        return False, f"insufficient shadow trades ({n_shadow_trades} < {min_shadow_trades})"
    
    if shadow_sharpe < min_shadow_sharpe:
        return False, f"shadow Sharpe too low ({shadow_sharpe:.3f} < {min_shadow_sharpe})"
    
    if shadow_win_rate < min_shadow_win_rate:
        return False, f"shadow win rate too low ({shadow_win_rate:.3f} < {min_shadow_win_rate})"
    
    if fitness < min_fitness:
        return False, f"fitness negative ({fitness:.3f} < {min_fitness}) — overfit or unstable"
    
    return True, "passed"

# Usage: call compute_fitness() on walk-forward result; only if promotion_gate() returns True,
# update AdaptiveMetaLearner / allow live strategy selection.
