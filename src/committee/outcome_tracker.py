"""Turns closed trades / backtest trades into training examples for the meta-learner.

A training example records which brains voted (and how), the market regime, the
committee's final action & confidence, the realized net PnL after fees, and the
holding period. The label is simply the sign of net PnL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.logging_config import get_logger

logger = get_logger("outcome_tracker")


@dataclass
class TrainingExample:
    """One closed-trade outcome, ready to feed into ``AdaptiveMetaLearner.update``."""

    symbol: str
    regime: str
    final_action: str
    confidence: float
    brain_votes: Dict[str, str]  # brain_name -> action
    realized_pnl: float
    return_pct: float = 0.0
    holding_period_sec: float = 0.0
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    decision_id: Optional[str] = None

    @property
    def label(self) -> int:
        """1 = profitable, 0 = loss, -1 = flat/breakeven (no learning signal)."""
        if self.realized_pnl > 0:
            return 1
        if self.realized_pnl < 0:
            return 0
        return -1

    def to_decision_snapshot(self) -> Dict[str, Any]:
        """Shape expected by ``AdaptiveMetaLearner.update`` as its first argument."""
        return {
            "regime": self.regime,
            "final_action": self.final_action,
            "brain_votes": self.brain_votes,
        }

    def to_realized_outcome(self) -> Dict[str, Any]:
        """Shape expected by ``AdaptiveMetaLearner.update`` as its second argument."""
        return {"net_pnl": self.realized_pnl, "return_pct": self.return_pct}


def _parse_ts(value: Any) -> Optional[float]:
    """Best-effort ISO/epoch timestamp -> epoch seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        from datetime import datetime

        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _holding_period(entry_time: Any, exit_time: Any) -> float:
    a, b = _parse_ts(entry_time), _parse_ts(exit_time)
    if a is None or b is None:
        return 0.0
    return max(0.0, b - a)


def from_decision_snapshot(
    snapshot: Dict[str, Any],
    realized_pnl: float,
    *,
    return_pct: float = 0.0,
    holding_period_sec: float = 0.0,
) -> TrainingExample:
    """Build an example from a persisted committee decision snapshot dict.

    ``snapshot`` is expected to carry: symbol, regime, final_action/action,
    confidence, and either ``brain_votes`` (name->action) or ``votes``.
    """
    votes = snapshot.get("brain_votes")
    if not isinstance(votes, dict):
        votes = {}
        for v in snapshot.get("votes", []) or []:
            if isinstance(v, dict) and v.get("name"):
                votes[str(v["name"])] = str(v.get("action", "hold"))
            elif getattr(v, "name", None):
                votes[str(v.name)] = str(getattr(v, "action", "hold"))
    return TrainingExample(
        symbol=str(snapshot.get("symbol", "")),
        regime=str(snapshot.get("regime", "default")),
        final_action=str(snapshot.get("final_action", snapshot.get("action", "hold"))),
        confidence=float(snapshot.get("confidence", 0.0)),
        brain_votes=votes,
        realized_pnl=float(realized_pnl),
        return_pct=float(return_pct),
        holding_period_sec=float(holding_period_sec),
        entry_time=snapshot.get("entry_time"),
        exit_time=snapshot.get("exit_time"),
        decision_id=snapshot.get("decision_id"),
    )


def from_backtest_trade(
    trade: Any,
    *,
    regime: str = "default",
    brain_votes: Optional[Dict[str, str]] = None,
    confidence: float = 0.0,
) -> TrainingExample:
    """Convert a ``backtest.BacktestTrade`` (or similar) into a TrainingExample.

    BacktestTrade stores side/entry/exit/pnl/pnl_pct/times but not brain votes
    or regime, so those are supplied by the caller when available.
    """
    side = str(getattr(trade, "side", "long"))
    final_action = "buy" if side in ("long", "buy") else "sell"
    entry_time = getattr(trade, "entry_time", None)
    exit_time = getattr(trade, "exit_time", None)
    return TrainingExample(
        symbol=str(getattr(trade, "symbol", "")),
        regime=regime,
        final_action=final_action,
        confidence=confidence,
        brain_votes=brain_votes or {},
        realized_pnl=float(getattr(trade, "pnl", 0.0)),
        return_pct=float(getattr(trade, "pnl_pct", 0.0)),
        holding_period_sec=_holding_period(entry_time, exit_time),
        entry_time=str(entry_time) if entry_time is not None else None,
        exit_time=str(exit_time) if exit_time is not None else None,
    )


def apply_to_learner(learner: Any, examples: List[TrainingExample]) -> int:
    """Feed a batch of examples into a learner. Returns the count applied."""
    applied = 0
    for ex in examples:
        try:
            learner.update(ex.to_decision_snapshot(), ex.to_realized_outcome())
            applied += 1
        except Exception as e:
            logger.warning(f"Skipping training example ({ex.symbol}): {e}")
    return applied
