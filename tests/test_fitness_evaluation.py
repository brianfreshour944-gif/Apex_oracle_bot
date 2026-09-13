"""Tests for the fitness evaluation framework (compute_fitness + promotion_gate)."""
import pytest

from src.fitness_evaluation import compute_fitness, promotion_gate


# ── compute_fitness ──────────────────────────────────────────────────────────

def test_fitness_positive_for_clean_candidate():
    f = compute_fitness(
        oos_return_pct=10.0, max_drawdown_pct=-8.0, annual_turnover_pct=100.0,
        n_parameters=2, return_std=0.05, return_mean=0.02,
    )
    assert f > 0


def test_more_parameters_lower_fitness():
    """Complexity penalty: same OOS results, more tuned params -> lower fitness."""
    kwargs = dict(
        oos_return_pct=10.0, max_drawdown_pct=-10.0, annual_turnover_pct=100.0,
        return_std=0.05, return_mean=0.02,
    )
    assert compute_fitness(n_parameters=8, **kwargs) < compute_fitness(n_parameters=2, **kwargs)


def test_deeper_drawdown_lower_fitness():
    """Regression: dd penalty must trigger for DEEPER drawdowns (more negative),
    not shallower ones (sign was inverted)."""
    kwargs = dict(
        oos_return_pct=10.0, annual_turnover_pct=100.0,
        n_parameters=2, return_std=0.05, return_mean=0.02,
    )
    shallow = compute_fitness(max_drawdown_pct=-10.0, **kwargs)
    deep = compute_fitness(max_drawdown_pct=-30.0, **kwargs)
    assert deep < shallow


def test_higher_turnover_lower_fitness():
    kwargs = dict(
        oos_return_pct=10.0, max_drawdown_pct=-10.0,
        n_parameters=2, return_std=0.05, return_mean=0.02,
    )
    assert compute_fitness(annual_turnover_pct=800.0, **kwargs) < compute_fitness(annual_turnover_pct=100.0, **kwargs)


def test_high_instability_lower_fitness():
    kwargs = dict(
        oos_return_pct=10.0, max_drawdown_pct=-10.0, annual_turnover_pct=100.0,
        n_parameters=2,
    )
    stable = compute_fitness(return_std=0.01, return_mean=0.05, **kwargs)
    unstable = compute_fitness(return_std=0.50, return_mean=0.01, **kwargs)
    assert unstable < stable


def test_zero_return_rejects_via_penalties():
    """No edge + any complexity -> non-positive fitness."""
    f = compute_fitness(
        oos_return_pct=0.0, max_drawdown_pct=-10.0, annual_turnover_pct=100.0,
        n_parameters=4, return_std=0.05, return_mean=0.0,
    )
    assert f <= 0


# ── promotion_gate ───────────────────────────────────────────────────────────

GOOD = dict(fitness=0.08, shadow_sharpe=1.2, shadow_win_rate=0.58, n_shadow_trades=50)


def test_gate_passes_good_candidate():
    ok, reason = promotion_gate(**GOOD)
    assert ok is True
    assert reason == "passed"


def test_gate_rejects_insufficient_trades():
    ok, reason = promotion_gate(**{**GOOD, "n_shadow_trades": 10})
    assert ok is False
    assert "insufficient shadow trades" in reason


def test_gate_rejects_low_sharpe():
    ok, reason = promotion_gate(**{**GOOD, "shadow_sharpe": 0.2})
    assert ok is False
    assert "Sharpe" in reason


def test_gate_rejects_low_win_rate():
    ok, reason = promotion_gate(**{**GOOD, "shadow_win_rate": 0.45})
    assert ok is False
    assert "win rate" in reason


def test_gate_rejects_negative_fitness():
    ok, reason = promotion_gate(**{**GOOD, "fitness": -0.02})
    assert ok is False
    assert "fitness" in reason
