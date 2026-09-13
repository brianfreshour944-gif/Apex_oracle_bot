"""Tests for apply_uncertainty_scaling — wires the strategy layer's
transition-risk and position-scale metrics into position sizing."""
import math

import pytest

from src.risk import apply_uncertainty_scaling


def test_no_risk_full_scale_is_noop():
    size, mult = apply_uncertainty_scaling(1.0, transition_risk_pct=0.0, position_scale=1.0)
    assert size == pytest.approx(1.0)
    assert mult == pytest.approx(1.0)


def test_transition_risk_scales_down_linearly():
    # 20% transition risk -> 0.8x conviction
    size, mult = apply_uncertainty_scaling(1.0, transition_risk_pct=20.0, position_scale=1.0)
    assert size == pytest.approx(0.8)
    assert mult == pytest.approx(0.8)


def test_transition_risk_floored_at_half():
    # 50%+ risk -> conviction floored at 0.5x (never zero)
    for risk in (50.0, 80.0, 200.0):
        size, _ = apply_uncertainty_scaling(1.0, transition_risk_pct=risk, position_scale=1.0)
        assert size == pytest.approx(0.5)


def test_position_scale_caps_size():
    # Low conviction (confidence*transition) caps at 0.4x of pre-uncertainty size
    size, mult = apply_uncertainty_scaling(1.0, transition_risk_pct=0.0, position_scale=0.4)
    assert size == pytest.approx(0.4)
    assert mult == pytest.approx(0.4)


def test_position_scale_floor_at_quarter():
    size, _ = apply_uncertainty_scaling(1.0, transition_risk_pct=0.0, position_scale=0.0)
    assert size == pytest.approx(0.25)


def test_combined_takes_min_of_both():
    # 25% risk -> 0.75x; scale cap 0.4x -> min(0.75, 0.4) = 0.4
    size, mult = apply_uncertainty_scaling(2.0, transition_risk_pct=25.0, position_scale=0.4)
    assert size == pytest.approx(2.0 * 0.4)
    assert mult == pytest.approx(0.4)


def test_invalid_inputs_degrade_to_noop():
    # NaN / None must not corrupt sizing
    size, mult = apply_uncertainty_scaling(1.0, transition_risk_pct=float("nan"), position_scale=None)
    assert size == pytest.approx(1.0)
    assert mult == pytest.approx(1.0)
    size, mult = apply_uncertainty_scaling(0.0, transition_risk_pct=50.0, position_scale=0.5)
    assert size == 0.0
    size, mult = apply_uncertainty_scaling(-1.0, transition_risk_pct=50.0, position_scale=0.5)
    assert size == -1.0


def test_negative_risk_treated_as_zero():
    size, _ = apply_uncertainty_scaling(1.0, transition_risk_pct=-30.0, position_scale=1.0)
    assert size == pytest.approx(1.0)
