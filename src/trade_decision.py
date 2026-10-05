"""Entry-decision logic shared by the live loop and the backtester.

These rules used to live inline in bot.process_signal_for_symbol(), so the
backtester (src/backtest.py) never applied them and simulated a different bot:
it skipped the adversarial vetoes, used the strategy's raw expected return
instead of the committee-derived one, and ignored the post-sizing multipliers
and the exchange-minimum bump. Keeping one implementation here means a
backtest exercises exactly what live trading does.

Everything here is pure (no I/O, no clocks) so both callers get identical
results for identical inputs.
"""

import math
from typing import Any

from src.config import settings
from src.risk import apply_uncertainty_scaling

# Score floor for the HIGH-disagreement adversarial veto: genuine directional
# disagreement paired with a weak conviction score (< this) is rejected. It
# applies to the DIRECTIONAL disagreement level -- see
# src/committee/models.py:calculate_directional_entropy for why abstentions
# must not count as disagreement here.
ADVERSARIAL_SCORE_FLOOR = 0.55

# Base edge by regime, used to estimate expected return for sizing.
REGIME_EDGE = {
    "trending": 0.015,      # 1.5% edge in trending
    "bull": 0.015,
    "bear": 0.015,
    "mean_reverting": 0.01,  # 1% edge in mean reversion
    "sideways": 0.01,
    "high_volatility": 0.0,  # No edge in high vol
    "low_volatility": 0.005,  # 0.5% edge in low vol
    "neutral": 0.0,
}


def adversarial_veto_reasons(
    signal: dict[str, Any],
    committee_result: Any,
    learner: Any = None,
    validation_min_trades: int | None = None,
) -> list[str]:
    """Return the adversarial veto reasons for an entry (empty = no veto).

    Brain B: execution cost exceeds expected edge (hard constraint).
    Brain C: the regime has enough realized outcomes to be judged and failed
    validation. "No validation data yet" is not evidence of overfitting, so a
    regime below ``validation_min_trades`` is never vetoed here.
    Disagreement: HIGH directional disagreement with a score below
    ADVERSARIAL_SCORE_FLOOR. ``signal["brain_disagreement"]`` must already be
    set by the caller.
    """
    reasons: list[str] = []

    signal_edge = signal.get("expected_edge_bps", 0)
    execution_cost = signal.get("execution_cost_bps", 0) + (signal.get("final_edge_bps", 0) - signal_edge)
    if execution_cost > signal_edge:
        reasons.append(f"Brain B veto: execution_cost ({execution_cost:.1f}bps) > expected_edge ({signal_edge:.1f}bps)")

    if learner is not None and validation_min_trades is not None:
        regime_for_gate = signal.get("regime", "neutral")
        gate_metrics = learner.get_regime_validation_metrics(regime_for_gate) or {}
        n_trades = int(gate_metrics.get("n_trades", 0) or 0)
        if n_trades >= validation_min_trades and not gate_metrics.get("validated", False):
            reasons.append(
                f"Brain C veto: regime '{regime_for_gate}' not validated (OOS Sharpe "
                f"{gate_metrics.get('sharpe', 0.0):.2f} < 0.5 or win rate "
                f"{gate_metrics.get('win_rate', 0.0):.2f} < 52%, n={n_trades})"
            )

    brain_disagreement = signal.get("brain_disagreement", "LOW")
    if brain_disagreement == "HIGH" and committee_result.score < ADVERSARIAL_SCORE_FLOOR:
        reasons.append(
            f"Brain disagreement HIGH (directional) + score "
            f"{committee_result.score:.2f} < {ADVERSARIAL_SCORE_FLOOR:.2f}"
        )
    return reasons


def estimate_expected_return(regime: str, score: float, entropy: float) -> float:
    """Expected return (fraction, e.g. 0.03 = 3%) from regime, committee score and consensus.

    Score represents P(win) mapped to an edge; high entropy (disagreement)
    reduces it. calculate_position_size() takes this as a fraction and
    converts to bps itself -- do NOT multiply by 100 (that once produced a
    100x-inflated edge that defeated the min-edge-after-costs gate).
    """
    regime_edge = REGIME_EDGE.get(regime or "neutral", 0.0)
    score_edge = (score - 0.5) * 2.0  # maps [0,1] -> [-1, 1]
    entropy_penalty = max(0.0, entropy - 0.5) * 0.5
    raw_edge = regime_edge + score_edge * 0.02 - entropy_penalty  # scale score edge to ~2%
    return max(-0.02, min(0.05, raw_edge))  # cap at -2% to +5%


def apply_entry_size_multipliers(
    position_size: float,
    signal: dict[str, Any],
    committee_result: Any,
    current_price: float,
    oracle_multiplier: float = 1.0,
) -> tuple[float, float]:
    """Apply the post-sizing multipliers and re-cap. Returns (size, uncertainty_mult).

    Order matches live: uncertainty scaling (transition risk / position
    scale), oracle regime multiplier (buys only), committee confidence
    multiplier, then MAX_SINGLE_TRADE_USD re-cap for buys -- the multipliers
    can push a capped size back over (committee_mult alone reaches 1.75x).
    """
    position_size, uncertainty_mult = apply_uncertainty_scaling(
        position_size,
        transition_risk_pct=float(signal.get("transition_risk_pct", 0.0) or 0.0),
        position_scale=float(signal.get("position_scale", 1.0) or 1.0),
    )
    if signal["action"] == "buy":
        position_size = position_size * oracle_multiplier
    committee_mult = getattr(committee_result, "size_multiplier", 1.0)
    position_size = round(position_size * committee_mult, 6)
    if signal["action"] == "buy":
        max_qty_at_cap = settings.MAX_SINGLE_TRADE_USD / current_price
        if position_size > max_qty_at_cap:
            position_size = round(max_qty_at_cap, 6)
    return position_size, uncertainty_mult


def min_order_bump(position_size: float, current_price: float) -> float | None:
    """Exchange-minimum handling for buys.

    Returns the size to place: unchanged if at/above MIN_ORDER_USD, bumped up
    to the minimum if it's at least half of it, or None (veto) if it's
    further below -- bumping >2x would be too large a deviation from the
    intended risk. Exposure for any bump is the caller's job.
    """
    min_order_usd = getattr(settings, "MIN_ORDER_USD", 10.0)
    final_notional = current_price * position_size
    if not (0 < final_notional < min_order_usd):
        return position_size
    if final_notional >= min_order_usd * 0.5:
        return math.ceil(min_order_usd / current_price * 1_000_000) / 1_000_000
    return None
