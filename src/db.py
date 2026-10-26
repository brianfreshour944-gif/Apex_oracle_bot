"""Modern database layer using SQLAlchemy 2.0 with MappedAsDataclass style."""

import datetime
import json
import os
import time
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    String,
    Text,
    create_engine,
    event,
    func,
    select,
    text,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

# Modern SQLAlchemy 2.0 DeclarativeBase
class Base(DeclarativeBase):
    """Modern SQLAlchemy 2.0 declarative base using MappedAsDataclass style."""
    pass


class DecisionSnapshot(Base):
    """A committee decision recorded at trade entry, closed out at exit.

    Correlates the brains' votes + regime + final action taken at entry with the
    realized PnL known only at exit, so the adaptive meta-learner can be updated
    on a completed round-trip. Purely observational: it never gates trading.
    """

    __tablename__ = "decision_snapshots"
    # get_open_snapshot() filters on exactly this (symbol, status) pair,
    # ordered by created_at -- measured via EXPLAIN QUERY PLAN to be a full
    # table scan + temp b-tree sort without this index (SCAN decision_snapshots),
    # and latency grows with table size (2-5ms at <5k rows, 18.8ms at 20k rows).
    __table_args__ = (
        Index("ix_decision_snapshots_symbol_status", "symbol", "status"),
    )

    decision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    regime: Mapped[str] = mapped_column(String(48), default="default")
    final_action: Mapped[str] = mapped_column(String(16), default="hold")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    size_multiplier: Mapped[float] = mapped_column(Float, default=1.0)
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    qty: Mapped[float] = mapped_column(Float, default=0.0)
    votes_json: Mapped[str] = mapped_column(Text, default="{}")  # {brain: action}
    feature_snapshot_json: Mapped[str] = mapped_column(Text, default="{}")
    causal_reasoning_json: Mapped[str] = mapped_column(Text, default="{}") # {feature: contribution}
    tensor_state_json: Mapped[str] = mapped_column(Text, default="{}") # {brain: tensor_state}
    status: Mapped[str] = mapped_column(String(16), default="open")  # open|closed
    exit_reason: Mapped[str | None] = mapped_column(String(48), nullable=True, default=None)
    max_favorable_pct: Mapped[float] = mapped_column(Float, default=0.0)
    max_adverse_pct: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    return_pct: Mapped[float] = mapped_column(Float, default=0.0)
    holding_period_sec: Mapped[float] = mapped_column(Float, default=0.0)
    # Machine-readable trust status of realized_pnl/return_pct. One of
    # "reconciled" or "unreconciled". NULL on rows written before this column
    # existed, which must be treated as "unknown" -- never as reconciled.
    #
    # IMPORTANT SCOPE (2026-10-25 review): "reconciled" means the P&L was
    # REBUILT FROM THE ORDER LEDGER and the arithmetic balanced -- quantities
    # tied out within tolerance, and both legs carried a NONZERO commission.
    # It does NOT mean the fees are VERIFIED. `commission` is an estimated
    # value on this deployment (Alpaca reports none; the adapter derives it
    # from ESTIMATED_TAKER_FEE_BPS) and the estimate flag is not persisted, so
    # a nonzero commission proves a cost was recorded, not that the cost is
    # accurate. Read "reconciled" as "internally consistent reconstruction
    # with recorded fees", never as "audited against exchange billing".
    # Separating accounting reconciliation from fee verification would need a
    # fee-provenance column and is a deliberate follow-up, not part of this
    # change. The LABEL itself never gates trading, risk, or committee
    # behavior; but the realized_pnl/return_pct values it describes DO reach
    # consumers, some of which (e.g. the rolling loss limit) use every closed
    # row regardless of this status. Do not read "unreconciled" as excluded
    # from decisions -- it is a data-quality warning, not an exclusion
    # guarantee.
    outcome_status: Mapped[str | None] = mapped_column(String(16), nullable=True, default=None)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.UTC)
    )
    closed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

class ShadowTrade(Base):
    """Tracks virtual positions taken by candidate models in the Evolution Tournament."""
    __tablename__ = "shadow_trades"
    # check_shadow_stops()/process_shadow_signal() both filter on exactly
    # this (candidate_name, symbol, status) triple, ordered by created_at.
    #
    # The UNIQUE partial index enforces at most ONE open shadow position per
    # (candidate, symbol). Without it, the read-then-insert in
    # _evaluate_shadow_position() is a check-then-act race: two concurrent
    # evaluations (one thread per symbol via asyncio.to_thread) can both see
    # "no open trade" and both insert, opening duplicate positions that
    # corrupt the candidate PnL the monthly cull ranks on. Reproduced
    # 2026-10-12: 6 simultaneous signals -> 3 open rows.
    __table_args__ = (
        Index("ix_shadow_trades_candidate_symbol_status", "candidate_name", "symbol", "status"),
        Index(
            "ux_shadow_trades_open_candidate_symbol",
            "candidate_name",
            "symbol",
            unique=True,
            sqlite_where=text("status = 'open'"),
            postgresql_where=text("status = 'open'"),
        ),
    )

    trade_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    candidate_name: Mapped[str] = mapped_column(String(64))
    symbol: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(16))  # buy or short
    qty: Mapped[float] = mapped_column(Float, default=0.0)
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    status: Mapped[str] = mapped_column(String(16), default="open")  # open|closed
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.UTC)
    )
    closed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

class ExperimentRecord(Base):
    """Registry of all automated research experiments to track historical performance."""
    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    generation_type: Mapped[str] = mapped_column(String(32)) # e.g. "Genetic", "AutoML"
    architecture_details: Mapped[str] = mapped_column(Text) # JSON string
    sharpe: Mapped[float] = mapped_column(Float, default=0.0)
    profit_factor: Mapped[float] = mapped_column(Float, default=0.0)
    max_dd: Mapped[float] = mapped_column(Float, default=0.0)
    total_return: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="Candidate")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.UTC)
    )

class OrderRecord(Base):
    """Record of Alpaca orders with fill data for data integrity verification."""
    __tablename__ = "orders"
    __table_args__ = (
        Index("ix_orders_decision_id", "decision_id"),
        Index("ix_orders_symbol_status", "symbol", "status"),
    )

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    decision_id: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    symbol: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(16))  # buy/sell
    qty: Mapped[float] = mapped_column(Float, default=0.0)
    filled_qty: Mapped[float] = mapped_column(Float, default=0.0)
    filled_avg_price: Mapped[float] = mapped_column(Float, default=0.0)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    # Provenance of `commission`. NULL = unknown -- never assume "verified".
    #   "estimated" = derived from ESTIMATED_TAKER_FEE_BPS by the exchange
    #                 adapter (Alpaca reports no fee on the Order model), NOT
    #                 an exchange-observed charge.
    #   "exchange"  = a nonzero commission the adapter observed directly.
    # A stored 0.0 with NULL provenance means "no fee recorded", which is NOT
    # the same as "no fee charged". Historical rows keep NULL.
    fee_source: Mapped[str | None] = mapped_column(String(16), nullable=True, default=None)
    # Submission-time quote snapshot (2026-10-25, W1 instrumentation). Captured
    # from the NBBO immediately BEFORE the order was submitted, so it is
    # contemporaneous with the decision -- NOT a post-fill price. All three are
    # nullable and mean "not captured" when NULL; a NULL must never be read as a
    # zero spread or a zero cost. `mid_at_submit`/`spread_at_submit` are in price
    # units; `quote_ts` is the exchange timestamp of the captured quote.
    mid_at_submit: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    spread_at_submit: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    quote_ts: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    status: Mapped[str] = mapped_column(String(32), default="new")
    type: Mapped[str] = mapped_column(String(16), default="market")
    time_in_force: Mapped[str] = mapped_column(String(16), default="ioc")
    client_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    submitted_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.UTC)
    )
    filled_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

def save_experiment_record(
    experiment_id: str,
    generation_type: str,
    architecture_details: dict[str, Any],
    sharpe: float,
    max_dd: float,
    total_return: float,
    status: str = "Candidate",
    profit_factor: float = 0.0
) -> bool:
    """Save an experiment to the registry."""
    try:
        _ensure_tables()
        with get_db_session() as session:
            rec = ExperimentRecord(
                experiment_id=experiment_id,
                generation_type=generation_type,
                architecture_details=json.dumps(architecture_details),
                sharpe=float(sharpe),
                profit_factor=float(profit_factor),
                max_dd=float(max_dd),
                total_return=float(total_return),
                status=status
            )
            session.merge(rec)
            session.commit()
        return True
    except Exception as e:
        logger.warning(f"save_experiment_record failed (non-fatal): {e}")
        return False


def get_latest_experiment_stats():
    """Return the most recent experiment's computed performance stats."""
    try:
        _ensure_tables()
        with get_db_session() as session:
            stmt = (
                select(ExperimentRecord)
                .order_by(ExperimentRecord.created_at.desc())
                .limit(1)
            )
            rec = session.execute(stmt).scalar_one_or_none()
            if rec is None:
                return None
            return {
                "experiment_id": rec.experiment_id,
                "sharpe": rec.sharpe,
                "max_drawdown_pct": rec.max_dd,
                "total_return_pct": rec.total_return,
                "profit_factor": rec.profit_factor,
                "status": rec.status,
                "computed_at": rec.created_at.isoformat() if rec.created_at else None,
            }
    except Exception as e:
        logger.warning(f"get_latest_experiment_stats failed (non-fatal): {e}")
        return None


def save_order_record(
    *,
    order_id: str,
    decision_id: str | None = None,
    symbol: str,
    side: str,
    qty: float,
    filled_qty: float = 0.0,
    filled_avg_price: float = 0.0,
    commission: float = 0.0,
    fee_source: str | None = None,
    mid_at_submit: float | None = None,
    spread_at_submit: float | None = None,
    quote_ts: datetime.datetime | None = None,
    status: str = "new",
    type: str = "market",
    time_in_force: str = "ioc",
    client_order_id: str | None = None,
    submitted_at: datetime.datetime | None = None,
    filled_at: datetime.datetime | None = None,
) -> bool:
    """Persist an order record for data integrity verification. Returns True on success."""
    try:
        _ensure_tables()
        with get_db_session() as session:
            order = OrderRecord(
                order_id=order_id,
                decision_id=decision_id,
                symbol=symbol,
                side=side,
                qty=float(qty),
                filled_qty=float(filled_qty),
                filled_avg_price=float(filled_avg_price),
                commission=float(commission),
                fee_source=fee_source,
                mid_at_submit=mid_at_submit,
                spread_at_submit=spread_at_submit,
                quote_ts=quote_ts,
                status=status,
                type=type,
                time_in_force=time_in_force,
                client_order_id=client_order_id,
                submitted_at=submitted_at or datetime.datetime.now(datetime.UTC),
                filled_at=filled_at,
            )
            session.merge(order)
            session.commit()
        return True
    except Exception as e:
        logger.warning(f"save_order_record failed (non-fatal): {e}")
        return False


def get_entry_fee_estimate(
    symbol_clean: str,
    entry_price: float = 0.0,
    decision_id: str | None = None,
    max_age_sec: float = 7200.0,
) -> float:
    """Best-effort estimate of the fee paid on this position's ENTRY leg(s).

    The decision snapshot records only ONE commission (observed on the exit
    fill), but a round trip pays a fee on the entry fill(s) too. That entry fee
    is read back from the orders ledger:

      * When ``decision_id`` is known (the normal path), SUM the commissions of
        every filled buy carrying it. A scale-in folds several buys into the
        SAME snapshot (``update_decision_snapshot_position``), and each add paid
        its own fee, so they must all count. There is deliberately NO age cap
        here: a position may be held for days and its entry fills must still be
        attributed.
      * Only when ``decision_id`` is unavailable does it fall back to the most
        recent buy fill for the symbol whose price is within 5% of
        ``entry_price`` and submitted within ``max_age_sec`` -- the
        pre-decision-id heuristic, used when there is no exact key to match on.

    Returns 0.0 (fail-safe) rather than guessing when nothing matches.
    """
    try:
        _ensure_tables()
        with get_db_session() as session:
            if decision_id:
                stmt = select(OrderRecord).where(OrderRecord.decision_id == decision_id)
            else:
                cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=max_age_sec)
                stmt = (
                    select(OrderRecord)
                    .where(OrderRecord.submitted_at >= cutoff)
                    .order_by(OrderRecord.submitted_at.desc())
                    .limit(50)
                )
            rows = session.execute(stmt).scalars().all()

        def _is_real_buy(r: OrderRecord) -> bool:
            return (
                r.symbol.replace("/", "") == symbol_clean
                and str(r.side).lower() == "buy"
                and float(r.filled_qty or 0.0) > 0.0
                and float(r.filled_avg_price or 0.0) > 0.0
            )

        if decision_id:
            return sum(float(r.commission or 0.0) for r in rows if _is_real_buy(r))

        for r in rows:  # newest-first
            if not _is_real_buy(r):
                continue
            if float(r.commission or 0.0) <= 0.0:
                continue
            if entry_price > 0 and abs(float(r.filled_avg_price) - entry_price) / entry_price > 0.05:
                continue  # a different entry than this position's
            return float(r.commission)
        return 0.0
    except Exception as e:
        logger.warning(f"get_entry_fee_estimate failed (non-fatal): {e}")
        return 0.0


# Relative quantity tolerance, used ONLY as a coarse pre-filter on total
# bought-vs-sold quantity. It is NOT the certifying step: a residual that passes
# here is still rejected by the fee classifier below unless it matches a
# supported fee convention (see reconstruct_position_outcome). Measured on the
# deployed ledger, every reconciled round trip shows residual_qty == buy_qty *
# ESTIMATED_TAKER_FEE_BPS/10000 == 0.0025 (entry fee withheld in the base asset),
# so 0.005 = 2x that admits the real fee-sized residual while the observed 3.93
# (393%) outlier fails closed. It is relative (divided by buy_qty), so it is
# scale-free across assets. A sub-tolerance residual that does NOT match the
# recorded fee rate is still classified "unexplained_residual" -> unreconciled.
_QTY_RECONCILE_TOLERANCE = 0.005


def _count_position_lifecycles(ordered_legs: list[tuple[str, float]]) -> int:
    """Count distinct position lifecycles in time order.

    A lifecycle is: flat -> accumulate buys -> sell back down to flat. A new
    lifecycle begins only when a buy arrives while the position is already flat,
    so a normal scale-in (buy, buy, sell, sell) counts as one and a reused
    decision_id holding two separate round trips counts as two.

    "Flat" cannot be exact: the withheld entry fee means a full exit leaves
    ``sold == bought * (1 - fee_rate)``, not ``bought``. A cycle is therefore
    treated as closed once cumulative sells reach ``(1 - tol)`` of cumulative
    buys, which the 25 bps deployed fee clears comfortably.

    LIMITATION (2026-10-25 review, accepted, unchanged): this threshold is a
    heuristic, not a proof of position identity. It closes a cycle when sells
    reach ``(1 - tol)`` of buys, which is EARLIER than exact flat (1.0) and so
    errs toward SPLITTING (fail-closed: a spurious split yields
    ``multiple_position_lifecycles`` -> unreconciled). It does NOT, however,
    never merge: if an interior exit leaves a residual LARGER than ``tol``, the
    cycle stays open and a later buy is folded into it. If the aggregate later
    returns to flat, the merged row is certified "reconciled" as a single
    lifecycle. The trigger is a residual ABOVE ``tol``, which does NOT depend on
    the fee rate alone: a genuine partial exit (partial fill or scale-out) at the
    deployed 25 bps fee reaches it just as a >0.5% fee rate would. The counter's
    behavior is a function of cumulative quantities and their order, not of the
    configured fee. The threshold is retained deliberately -- hardening it
    changes the classification and is a separate, behavior-affecting change.
    """
    bought = sold = 0.0
    lifecycles = 0
    open_cycle = False
    for side, qty in ordered_legs:
        if side == "buy":
            if not open_cycle:
                lifecycles += 1
                open_cycle = True
            bought += qty
        else:
            sold += qty
            if open_cycle and bought > 0 and sold >= bought * (1.0 - _QTY_RECONCILE_TOLERANCE):
                open_cycle = False
    return lifecycles


def _has_ambiguous_reentry(ordered_legs: list[tuple[str, float]]) -> bool:
    """True if a buy arrives while the open cycle has already had a SELL and the
    residual is still larger than the tolerance.

    That history is consistent with a PARTIAL EXIT followed by a fresh position,
    which cannot be distinguished from a scale-in re-entry of the leftover. The
    aggregate may still return to flat, so the quantity gate passes and the row
    would otherwise be certified -- a false certification (2026-10-25 review).
    Such histories are therefore failed closed.
    """
    bought = sold = 0.0
    sold_since_open = 0.0
    open_cycle = False
    for side, qty in ordered_legs:
        if side == "buy":
            if (open_cycle and sold_since_open > 0 and bought > 0
                    and (bought - sold) / bought > _QTY_RECONCILE_TOLERANCE):
                return True
            if not open_cycle:
                open_cycle = True
                sold_since_open = 0.0
            bought += qty
        else:
            sold += qty
            sold_since_open += qty
            if open_cycle and bought > 0 and sold >= bought * (1.0 - _QTY_RECONCILE_TOLERANCE):
                open_cycle = False
    return False


def reconstruct_position_outcome(
    decision_id: str,
    *,
    exit_price: float | None = None,
    exit_qty: float | None = None,
    exit_commission: float = 0.0,
    entry_price: float | None = None,
    entry_qty: float | None = None,
) -> dict[str, Any]:
    """Rebuild a closed position's realized outcome from the order ledger.

    A position can be entered in several buys (scale-in) and exited in several
    sells (scale-out). The previous implementation closed the snapshot on the
    FIRST exit fill and computed P&L from that one leg's quantity, which
    charged the whole position's entry fee against a dust-sized gross and could
    even flip the sign of the result (2026-10-25 accounting audit). This
    reconstructs from EVERY linked fill.

    Cash-flow definition (this is what an account actually earns):
      cash_out_on_buys  = sum(buy.filled_qty * buy.filled_avg_price)
      cash_in_on_sells  = sum(sell.filled_qty * sell.filled_avg_price) - exit_fees
      realized_pnl      = cash_in_on_sells - cash_out_on_buys - entry_fees_in_quote

    Fee convention, established from the deployed fills themselves rather than
    assumed. The schema records no fee *currency*, so the convention is read off
    the fill quantities, which are the primary evidence:

      * The ENTRY fee is withheld in the BASE asset. On every fully-closed row
        of the deployed ledger ``sell_qty == buy_qty * (1 - 0.0025)`` exactly --
        the position that gets sold is the bought quantity net of the entry fee,
        so the entry fee is already inside ``proceeds`` and must NOT be
        subtracted a second time. (Independently: ``filled_qty == qty`` on every
        leg, so this is not a partial-fill artifact.)
      * The EXIT fee is charged in QUOTE: the whole net position is sold and the
        fee is deducted from the cash received, so it IS subtracted in full.

    The fee convention is therefore resolved by a two-way classifier, not by a
    partial-fee formula. The residual quantity fraction (``residual_qty /
    buy_qty``) is compared against the recorded entry fee rate
    (``entry_fees / position_basis``):

      * residual matches the fee rate -> the entry fee was withheld in BASE, so
        it is already inside the reduced ``proceeds``; ``entry_fee_unaccounted``
        is 0 and the fee is NOT subtracted again.
      * residual is ~0 -> the entry fee was charged in QUOTE as a separate cash
        charge; ``entry_fee_unaccounted`` is the full ``entry_fees``.
      * neither -> the residual is not explained by any supported convention, so
        the true cost is unknown: fail closed as "unexplained_residual".

    Supported conventions (both are recognised; anything else fails closed):
    entry fee withheld in the BASE asset, or charged in the QUOTE asset. The
    schema records no fee *currency* for any fill, so the convention is inferred
    from the recorded quantities -- which are the primary evidence -- and an
    ambiguous or missing fee record can never produce a certified net P&L: a row
    whose residual fits neither branch, or whose entry commission is missing
    while its quantity shows withholding, is returned "unreconciled".

    The caller's own observed exit fill (``exit_price``/``exit_qty``/
    ``exit_commission``) is used ONLY as a substitute exit leg when the ledger
    holds no sell for this decision_id (e.g. the fire-and-forget ledger write
    failed). Likewise ``entry_price``/``entry_qty`` substitute for a missing
    ledger buy. When a leg is substituted the row is labelled "unreconciled" --
    only a round trip assembled entirely from the ledger is "reconciled".

    Fail-closed: if neither the ledger nor the fallbacks can supply an entry and
    an exit, or the bought and sold quantities do not reconcile within
    ``_QTY_RECONCILE_TOLERANCE``, the result is marked "unreconciled" and the
    numeric fields are None -- a partial ledger calculation is NEVER presented
    as the actual P&L.
    """
    try:
        _ensure_tables()
        with get_db_session() as session:
            rows = session.execute(
                select(OrderRecord).where(OrderRecord.decision_id == decision_id)
            ).scalars().all()

        def _filled(r: OrderRecord) -> bool:
            return float(r.filled_qty or 0.0) > 0.0 and float(r.filled_avg_price or 0.0) > 0.0

        buys = [r for r in rows if str(r.side).lower() == "buy" and _filled(r)]
        sells = [r for r in rows if str(r.side).lower() == "sell" and _filled(r)]

        # A reused decision_id holding more than one round trip cannot be summed
        # into a single outcome: the legs belong to different position lifecycles
        # and their P&L must not be merged. Detect that from the time-ordered fill
        # sequence and fail closed rather than silently combining them (the real
        # `c6a6d0423cc2` case is exactly this).
        if buys and sells:
            def _ts(r: OrderRecord) -> float:
                # Execution order, not submission order: a lifecycle is defined
                # by when fills actually landed. Prefer filled_at and fall back
                # to submitted_at only when the fill time is absent. (Measured
                # on the deployed ledger: 24/73 legs lack filled_at, and on all
                # 13 multi-leg decisions the submit- and fill-orderings agree,
                # so the fallback cannot change a lifecycle count there.)
                dt = r.filled_at or r.submitted_at
                if dt is None:
                    return 0.0
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.UTC)
                return dt.timestamp()

            ordered = [
                (str(r.side).lower(), float(r.filled_qty))
                for r in sorted((r for r in rows if _filled(r)), key=_ts)
            ]
            lifecycles = _count_position_lifecycles(ordered)
            if lifecycles > 1 or _has_ambiguous_reentry(ordered):
                return {
                    "realized_pnl": None, "return_pct": None,
                    "outcome_status": "unreconciled",
                    "reason": ("multiple_position_lifecycles" if lifecycles > 1
                               else "ambiguous_lifecycle_history"),
                    "lifecycles": lifecycles,
                }

        legs = [(float(r.filled_qty), float(r.filled_avg_price), float(r.commission or 0.0)) for r in buys]
        exit_legs = [(float(r.filled_qty), float(r.filled_avg_price), float(r.commission or 0.0)) for r in sells]
        from_ledger = bool(buys) and bool(sells)

        # Substitute a missing leg from the caller's observed values. Only the
        # exit leg is substitutable in practice (the live path always persists
        # the entry buys); the entry fallback covers a snapshot with no ledger
        # buy at all, which is then explicitly labelled unreconciled.
        if not exit_legs and exit_price and exit_qty and float(exit_price) > 0 and float(exit_qty) > 0:
            exit_legs = [(float(exit_qty), float(exit_price), abs(float(exit_commission or 0.0)))]
        if not legs and entry_price and entry_qty and float(entry_price) > 0 and float(entry_qty) > 0:
            legs = [(float(entry_qty), float(entry_price), 0.0)]

        if not legs:
            return {
                "realized_pnl": None, "return_pct": None,
                "outcome_status": "unreconciled", "reason": "no_linked_entry_fills",
            }
        if not exit_legs:
            return {
                "realized_pnl": None, "return_pct": None,
                "outcome_status": "unreconciled", "reason": "no_linked_exit_fills",
            }

        buy_qty = sum(leg[0] for leg in legs)
        sell_qty = sum(leg[0] for leg in exit_legs)
        if buy_qty <= 0:
            return {
                "realized_pnl": None, "return_pct": None,
                "outcome_status": "unreconciled", "reason": "zero_entry_qty",
            }

        residual = buy_qty - sell_qty
        if abs(residual) / buy_qty > _QTY_RECONCILE_TOLERANCE:
            return {
                "realized_pnl": None, "return_pct": None,
                "outcome_status": "unreconciled", "reason": "quantity_mismatch",
                "buy_qty": buy_qty, "sell_qty": sell_qty, "residual_qty": residual,
            }

        position_basis = sum(leg[0] * leg[1] for leg in legs)
        proceeds = sum(leg[0] * leg[1] for leg in exit_legs)
        entry_fees = sum(leg[2] for leg in legs)
        exit_fees = sum(leg[2] for leg in exit_legs)
        if position_basis <= 0:
            return {
                "realized_pnl": None, "return_pct": None,
                "outcome_status": "unreconciled", "reason": "zero_position_basis",
            }

        # Cash flow. The schema records no fee *currency*, so the convention is
        # inferred from the recorded quantities, which are the primary evidence:
        #   - entry fee withheld in BASE: residual_qty/buy_qty == entry fee rate,
        #     and the fee is already inside the reduced proceeds;
        #   - entry fee charged in QUOTE: quantities match (residual ~ 0) and the
        #     fee is a separate cash charge.
        # Both tests are relative, so they hold for any position size. A residual
        # fitting neither is not a fee at all: the true cost is unknown, so fail
        # closed rather than guess.
        residual_frac = residual / buy_qty
        fee_rate = (entry_fees / position_basis) if position_basis > 0 else 0.0
        tol_frac = max(1e-6, 0.1 * fee_rate)

        # Fee provenance (2026-10-25 review). ``commission`` is a plain float
        # with no "verified" flag, and the exchange adapter documents that
        # Alpaca reports 0.0 on every fill -- the real fee is *estimated*, never
        # observed (see AlpacaExchange._apply_estimated_commission). A recorded
        # 0 therefore means "no fee recorded", NOT "no fee charged": the schema
        # cannot tell a verified zero from an absent one, so a fee-inclusive
        # P&L cannot be CERTIFIED when a needed fee is 0.
        #
        # This gate applies only to a round trip assembled entirely from the
        # ledger. A caller-substituted leg is already labelled "unreconciled"
        # (its fee is the caller's observed value, a weaker class of evidence),
        # so it keeps its best-effort number; the harm here is a *certified*
        # result, and only ledger rows can be certified. This is what stops an
        # exact-quantity, zero-fee ledger row -- the shape of every BTC order in
        # the deployed snapshot -- from being falsely certified as fee-free.
        if from_ledger and (entry_fees <= 0.0 or exit_fees <= 0.0):
            return {
                "realized_pnl": None, "return_pct": None,
                "outcome_status": "unreconciled", "reason": "unverified_fee_provenance",
                "buy_qty": buy_qty, "sell_qty": sell_qty, "residual_qty": residual,
                "entry_fees": entry_fees, "exit_fees": exit_fees,
            }

        if abs(residual_frac - fee_rate) <= tol_frac:
            entry_fee_unaccounted = 0.0            # withheld in base; already in proceeds
        elif abs(residual_frac) <= tol_frac:
            entry_fee_unaccounted = entry_fees     # charged in quote; subtract in full
        else:
            return {
                "realized_pnl": None, "return_pct": None,
                "outcome_status": "unreconciled", "reason": "unexplained_residual",
                "buy_qty": buy_qty, "sell_qty": sell_qty, "residual_qty": residual,
            }
        # The exit fee is charged in QUOTE (the whole net position is sold and the
        # fee is deducted from the cash received), so it is always subtracted.
        realized_pnl = proceeds - position_basis - exit_fees - entry_fee_unaccounted
        return_pct = realized_pnl / position_basis * 100.0
        return {
            "realized_pnl": realized_pnl,
            "return_pct": return_pct,
            # "reconciled" = ledger-assembled and arithmetic balanced, with a
            # nonzero (but ESTIMATED, not exchange-verified) fee on each leg.
            # It is not a fee-verification claim -- see the DecisionSnapshot
            # .outcome_status column comment for the full scope.
            "outcome_status": "reconciled" if from_ledger else "unreconciled",
            "reason": None if from_ledger else "leg_substituted_from_caller",
            "position_basis": position_basis,
            "proceeds": proceeds,
            "entry_fees": entry_fees,
            "exit_fees": exit_fees,
            "buy_qty": buy_qty,
            "sell_qty": sell_qty,
            "residual_qty": residual,
        }
    except Exception as e:
        logger.warning(f"reconstruct_position_outcome failed for {decision_id} (non-fatal): {e}")
        return {
            "realized_pnl": None, "return_pct": None,
            "outcome_status": "unreconciled", "reason": "reconstruction_error",
        }


def get_stale_protective_stops(max_age_sec: float = 7 * 24 * 3600.0) -> list[dict[str, Any]]:
    """Order-ledger protective-stop rows that may still need a fill refresh.

    A resting stop_limit sell is written to the ledger once, at arm time,
    while it is still working (status "new", filled_avg_price 0). If it later
    triggers, it fills server-side with no bot code in the path, so its row
    stays stale until something refreshes it from the exchange.

    Startup reconciliation cannot rely on ``_state.protective_stops`` to find
    those rows: that registry is in-memory and EMPTY after a restart, and
    reconcile runs at startup -- before any cycle repopulates it. So the
    candidates are read back from the ledger instead: every stop_limit sell
    within ``max_age_sec`` that is either non-final (status still in the live
    set) or has no recorded fill price. Rows already finalized WITH a fill
    price are up to date and skipped.

    The window defaults to 7 days (a resting stop's own lifetime is bounded by
    the position's, and gtc stops can rest a long time), which is why this is
    much wider than get_recent_order_records' 1h.
    """
    from src.exchange import LIVE_ORDER_STATUSES
    try:
        _ensure_tables()
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=max_age_sec)
        with get_db_session() as session:
            stmt = (
                select(OrderRecord)
                .where(
                    OrderRecord.type == "stop_limit",
                    OrderRecord.side == "sell",
                    OrderRecord.submitted_at >= cutoff,
                )
                .order_by(OrderRecord.submitted_at.desc())
                .limit(200)
            )
            rows = session.execute(stmt).scalars().all()
        out: list[dict[str, Any]] = []
        for r in rows:
            status = str(r.status or "").lower()
            if status not in LIVE_ORDER_STATUSES and float(r.filled_avg_price or 0.0) > 0.0:
                continue  # finalized with a real fill -> already up to date
            out.append({
                "order_id": r.order_id,
                "decision_id": r.decision_id,
                "symbol": r.symbol,
                "side": r.side,
                "client_order_id": r.client_order_id,
                "status": r.status,
                "filled_qty": r.filled_qty,
                "filled_avg_price": r.filled_avg_price,
                "submitted_at": r.submitted_at,
                "filled_at": r.filled_at,
            })
        return out
    except Exception as e:
        logger.warning(f"get_stale_protective_stops failed (non-fatal): {e}")
        return []


def get_recent_order_records(symbol_clean: str, max_age_sec: float = 3600.0) -> list[dict[str, Any]]:
    """Most-recent-first order-ledger records for a symbol within max_age_sec.

    `symbol_clean` matches the slash-stripped form exchange positions use
    (e.g. "BTCUSD"), so this also matches records saved with either form by
    stripping the stored symbol the same way. Used by startup reconciliation
    to distinguish a genuine crash-gap fill from a true orphan/ghost (CR-1)
    and to recover the real fill price instead of estimating from the
    current bar (CR-6).
    """
    try:
        _ensure_tables()
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=max_age_sec)
        with get_db_session() as session:
            stmt = (
                select(OrderRecord)
                .where(OrderRecord.submitted_at >= cutoff)
                .order_by(OrderRecord.submitted_at.desc())
                .limit(50)
            )
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "order_id": r.order_id,
                    "decision_id": r.decision_id,
                    "symbol": r.symbol,
                    "side": r.side,
                    "qty": r.qty,
                    "filled_qty": r.filled_qty,
                    "filled_avg_price": r.filled_avg_price,
                    "status": r.status,
                    "type": r.type,
                    "time_in_force": r.time_in_force,
                    "client_order_id": r.client_order_id,
                    "submitted_at": r.submitted_at,
                    "filled_at": r.filled_at,
                }
                for r in rows
                if r.symbol.replace("/", "") == symbol_clean
            ]
    except Exception as e:
        logger.warning(f"get_recent_order_records failed (non-fatal): {e}")
        return []

# Engine will be created lazily when first needed
_engine = None
# Tracks whether Base.metadata.create_all() has already run for the current
# engine. init_db() (called once at bot startup) always ensures this. The
# flag lets the individual save_* functions below self-heal (create tables
# on demand) if init_db() was skipped or failed at startup -- without paying
# create_all()'s table-existence-check cost on every single call, which
# measured ~7ms of pure event-loop blocking time per decision-snapshot write
# even when nothing had changed since the previous call.
_tables_ensured = False
# Set by _recovery_from_corruption(); init_db() folds it into its return value
# so a rebuild that happens inside a tenacity retry is still reported to the
# caller (bot.py alerts on a True return).
_recovered_from_corruption = False
# In-memory cache for get_open_snapshot() results, keyed by symbol.
# Invalidated when close_decision_snapshot() is called for that symbol.
_open_snapshot_cache: dict[str, dict[str, Any] | None] = {}


def _ensure_indexes() -> None:
    """Idempotently create indexes that may be missing on databases created
    before these indexes were added to the models (create_all() only creates
    indexes inline with CREATE TABLE, so it skips already-existing tables).
    """
    with get_engine().connect() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_decision_snapshots_symbol_status "
            "ON decision_snapshots (symbol, status)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_shadow_trades_candidate_symbol_status "
            "ON shadow_trades (candidate_name, symbol, status)"
        ))
        # Pre-existing DBs may already hold duplicate open shadow rows from
        # the check-then-act race this unique index closes. The CREATE UNIQUE
        # INDEX below would fail on them, so first retire every duplicate but
        # the newest into history (status='closed', realized_pnl left as-is)
        # -- nothing is deleted, and the candidate's closed-trade record is
        # preserved for the cull. Runs before the unique index on purpose.
        conn.execute(text(
            "UPDATE shadow_trades SET status = 'closed', closed_at = COALESCE(closed_at, created_at) "
            "WHERE status = 'open' AND trade_id NOT IN ("
            "  SELECT trade_id FROM ("
            "    SELECT trade_id, ROW_NUMBER() OVER ("
            "      PARTITION BY candidate_name, symbol ORDER BY created_at DESC, trade_id DESC"
            "    ) AS rn FROM shadow_trades WHERE status = 'open'"
            "  ) WHERE rn = 1"
            ")"
        ))
        conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_shadow_trades_open_candidate_symbol "
            "ON shadow_trades (candidate_name, symbol) WHERE status = 'open'"
        ))
        conn.commit()


def _ensure_additive_columns() -> None:
    """Idempotently add columns introduced after a table already existed.

    ``create_all()`` only issues CREATE TABLE for tables that do not exist yet,
    so it silently skips new columns on an already-deployed database. This adds
    them with ALTER TABLE, guarded by a dialect-agnostic column inspection so it
    is a no-op once present (SQLite has no ``ADD COLUMN IF NOT EXISTS``, and
    ``PRAGMA table_info`` is SQLite-only). Additive and nullable only -- no
    existing row is rewritten and no data is dropped.
    """
    additions = {
        "decision_snapshots": [("outcome_status", "VARCHAR(16)")],
        "orders": [
            ("fee_source", "VARCHAR(16)"),
            ("mid_at_submit", "FLOAT"),
            ("spread_at_submit", "FLOAT"),
            ("quote_ts", "DATETIME"),
        ],
    }
    try:
        from sqlalchemy import inspect as sa_inspect
        engine = get_engine()
        inspector = sa_inspect(engine)
        existing_tables = set(inspector.get_table_names())
        for table, columns in additions.items():
            if table not in existing_tables:
                continue  # table not created yet; create_all will handle it
            existing = {c["name"] for c in inspector.get_columns(table)}
            for name, ddl_type in columns:
                if name in existing:
                    continue
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl_type}"))
                logger.info(f"Added missing column {table}.{name} ({ddl_type})")
    except Exception as e:
        logger.warning(f"_ensure_additive_columns failed (non-fatal): {e}")


def _ensure_tables() -> None:
    global _tables_ensured
    if not _tables_ensured:
        Base.metadata.create_all(get_engine())
        _ensure_indexes()
        _ensure_additive_columns()
        _tables_ensured = True

def _recovery_from_corruption() -> None:
    """Move a corrupt SQLite database aside so it can be rebuilt cleanly.

    Handles bot.db, bot.db-shm, and bot.db-wal. The corrupt files are
    renamed to .corrupt.bak so the old data is recoverable for forensics
    while a fresh database is created on the next create_all().
    """
    db_url = settings.DATABASE_URL
    if not db_url.startswith("sqlite:///"):
        return
    db_path = db_url[len("sqlite:///"):]
    if not db_path or db_path == ":memory:":
        return
    # Record that a rebuild happened (only after the sqlite/memory guards, so
    # this is never set for a non-recoverable URL). init_db() is wrapped in
    # tenacity's @retry, so a recovery that happens inside a failed attempt is
    # otherwise invisible to the caller: the function raises, tenacity
    # retries, the retry succeeds against the freshly rebuilt file, and it
    # returns False. bot.py relies on a True return to alert that live trade
    # data was moved aside, so a corrupted database was previously rebuilt
    # SILENTLY.
    global _recovered_from_corruption
    _recovered_from_corruption = True

    # CR-7/CR-10: dispose the cached engine BEFORE attempting the rename, not
    # after. On Windows, the connection pool keeps its OS-level file handle
    # open even once the `with engine.connect()` block that triggered this
    # recovery has exited -- renaming (or even deleting) the file while that
    # handle is still open fails with WinError 32 ("used by another
    # process"), silently defeating the whole recovery path. Reproduced
    # 2026-09-21: the rename below failed every time until disposal moved
    # here, ahead of it.
    global _engine, _tables_ensured
    if _engine is not None:
        try:
            _engine.dispose()
        except Exception as e:
            logger.debug(f"Engine dispose during corruption recovery failed (non-fatal): {e}")
        _engine = None
    _tables_ensured = False

    suffixes = [db_path, db_path + "-shm", db_path + "-wal", db_path + "-journal"]
    backup_dir = os.path.join(os.path.dirname(db_path), "corrupt_backups")
    os.makedirs(backup_dir, exist_ok=True)

    for suffix in suffixes:
        if os.path.exists(suffix):
            backup_name = os.path.join(backup_dir, f"bot_corrupt_{int(time.time())}_{os.path.basename(suffix)}")
            try:
                os.rename(suffix, backup_name)
                logger.info(f"Moved corrupt database file {suffix} -> {backup_name}")
            except OSError as e:
                logger.warning(f"Could not move corrupt file {suffix}: {e}")
                try:
                    os.remove(suffix)
                except OSError:
                    pass

def get_engine():
    """Get the database engine, creating it if needed."""
    global _engine
    if _engine is None:
        is_sqlite = settings.DATABASE_URL.startswith("sqlite:///")

        if is_sqlite:
            db_path = settings.DATABASE_URL[len("sqlite:///"):]
            if db_path and db_path != ":memory:":
                db_dir = os.path.dirname(db_path)
                if db_dir:
                    os.makedirs(db_dir, exist_ok=True)

            # SQLite uses QueuePool but does NOT support pool_size / max_overflow.
            # check_same_thread=False is required for usage across asyncio.to_thread calls.
            _engine = create_engine(
                settings.DATABASE_URL,
                pool_recycle=3600,
                echo=False,
                future=True,
                connect_args={"check_same_thread": False},
            )

            # Enable WAL journal mode so concurrent reads and writes from multiple
            # threads (one per symbol evaluated in asyncio.to_thread) don't produce
            # silent "database is locked" OperationalErrors.
            @event.listens_for(_engine, "connect")
            def _set_wal_mode(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()

        else:
            # PostgreSQL / other — pool_size and max_overflow are valid here.
            # Fix malformed DATABASE_URL (missing '/' before db name)
            db_url = settings.DATABASE_URL
            if db_url.startswith("postgres://") or db_url.startswith("postgresql://"):
                from urllib.parse import urlparse
                parsed = urlparse(db_url)
                # Check if path is empty but netloc ends with port+dbname (missing slash)
                if not parsed.path and ':' in parsed.netloc:
                    host_port = parsed.netloc.rsplit(':', 1)
                    if len(host_port) == 2 and host_port[1].isdigit():
                        # Reconstruct with proper path
                        _new_netloc = host_port[0] + ':' + host_port[1]
                        _new_path = '/' + host_port[1]  # This is wrong - need actual db name
                        # Actually, the malformed URL has port and dbname concatenated
                        # e.g. "...:5432database_url" -> need to extract db name
                        # We can't auto-fix this reliably, so log a clear error
                        pass
            
            _engine = create_engine(
                settings.DATABASE_URL,
                pool_recycle=3600,
                echo=False,
                future=True,
                pool_size=10,
                max_overflow=20,
                connect_args={"connect_timeout": 10},
            )

        logger.info(f"Database engine created: {settings.DATABASE_URL}")
    return _engine

# Database session factory (will use lazy engine)
def get_session_factory():
    """Get a session factory using the lazy engine."""
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def init_db() -> bool:
    """Initialize database connection with exponential backoff retries.

    On first connect (SQLite only), also runs PRAGMA integrity_check to
    detect corruption from an unclean shutdown (OOM kill, power loss).
    If the database is corrupt, it is moved aside so create_all() can
    rebuild a clean schema rather than crash-looping.

    Returns True if corruption was detected and the DB was rebuilt (the
    caller should fire a critical alert -- a rebuild silently loses every
    decision snapshot, adaptive-learner sample, and closed-trade record,
    which previously only produced a single logger.warning with nothing
    downstream ever alerted. Confirmed as a real gap 2026-09-21
    cross-checking an external audit against this code: the connection-
    totally-fails case already alerts at the bot.py call site, but this
    silent-rebuild-and-continue case did not). False otherwise.
    """
    global _tables_ensured, _engine, _recovered_from_corruption
    corruption_detected = False
    db_url = settings.DATABASE_URL
    is_sqlite = db_url.startswith("sqlite:///")

    # CR-7: probe with a raw, short-lived stdlib sqlite3 connection BEFORE
    # the SQLAlchemy engine ever touches the file, while it's still None
    # (first call only -- get_engine() below would otherwise create and pool
    # a connection). Fully corrupt files (bad header / not a SQLite file at
    # all -- the realistic shape of hard-kill corruption) fail even a plain
    # schema read, before ever reaching the integrity_check further down, so
    # the graceful rebuild path was only reachable for "soft" corruption
    # (valid header, damaged internal b-tree structure). Tried catching this
    # via the SQLAlchemy connection instead first (dispose the pooled engine,
    # then rename) -- reproduced on Windows that the pooled connection's OS
    # file handle survives engine.dispose() + gc.collect() when the failure
    # happens inside the WAL-mode "connect" pool event, permanently failing
    # the rename with WinError 32 on every retry. A raw stdlib connection
    # opened and closed outside any pool has no such entanglement. Found via
    # an external crash-recovery audit, root-caused and fixed 2026-09-21/22.
    #
    # NOTE: the probe query is `PRAGMA schema_version`, not `SELECT 1`.
    # `SELECT 1` never reads the schema, so it succeeds on a corrupt-header
    # file and this whole pre-probe silently no-ops (see _probe_sqlite_file;
    # reproduced 2026-10-06).
    if is_sqlite and _engine is None:
        raw_path = db_url[len("sqlite:///"):]
        if raw_path and raw_path != ":memory:" and os.path.exists(raw_path):
            corruption_msg = _probe_sqlite_file(raw_path)
            if corruption_msg:
                logger.warning(
                    f"SQLite file failed a raw pre-connect probe (header-level "
                    f"corruption): {corruption_msg}. Moving aside and rebuilding."
                )
                _recovery_from_corruption()
                corruption_detected = True

    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))

            # SQLite integrity check — detect corruption from unclean shutdown.
            # Also the fallback for corruption introduced after the engine
            # already exists (rarer in practice than the fresh-startup case
            # the pre-probe above covers, and shares the same pooled-
            # connection Windows-lock risk if it ever has to recover here).
            if is_sqlite:
                try:
                    result = conn.execute(text("PRAGMA integrity_check")).fetchone()
                    if result and result[0] != "ok":
                        logger.warning(f"SQLite integrity check failed: {result[0]}. "
                                       f"Database may be corrupt — moving aside and rebuilding.")
                        _recovery_from_corruption()
                        corruption_detected = True
                except SQLAlchemyError as pragma_err:
                    logger.warning(f"Could not run integrity_check: {pragma_err}")

        # Create ORM tables if they do not exist (safe/idempotent). Reopens
        # against a fresh engine if recovery just ran above.
        Base.metadata.create_all(get_engine())
        _tables_ensured = True

        # create_all() only issues CREATE TABLE (with inline indexes) for
        # tables that don't exist yet -- it silently skips index creation for
        # tables that were created before these indexes were added to the
        # models. Explicitly (and idempotently) ensure they exist on already-
        # deployed databases too.
        _ensure_indexes()
        # Same reason, for columns: create_all() skips new columns on tables
        # that already exist, so add any missing ones here.
        _ensure_additive_columns()

        logger.info(f"Database connected: {settings.DATABASE_URL}")
        # Consume the flag: report recoveries seen up to this call, then clear
        # it so a later healthy init_db() returns False again.
        recovered = corruption_detected or _recovered_from_corruption
        _recovered_from_corruption = False
        return recovered
    except SQLAlchemyError as e:
        # F1: a corrupt -wal/-shm sidecar (valid main file) fails HERE with
        # "disk I/O error", not via the raw pre-probe above -- reproduced
        # that a plain, unpooled `sqlite3.connect(path); SELECT 1` does NOT
        # hit this error the way SQLAlchemy's own connection (which runs
        # PRAGMA journal_mode=WAL/synchronous=NORMAL on every new physical
        # connection via the "connect" pool event) does, so the pre-probe
        # can't catch this shape and it previously retried 5x then gave up
        # permanently with nothing ever clearing the poisoned WAL (reproduced
        # 3/3 runs). Only remove the WAL/SHM siblings, never the main file --
        # a corrupt WAL is by definition never-checkpointed data, so this
        # can't lose anything that was actually committed. If it doesn't
        # help, fall through to the normal retry (and eventually the raw
        # pre-probe's own corruption path on the next init_db() call, since
        # _engine gets reset to None below on this path too). Found via an
        # external crash-recovery audit, root-caused and fixed 2026-09-22.
        if is_sqlite and "disk i/o error" in str(e).lower():
            raw_path = db_url[len("sqlite:///"):]
            if raw_path and raw_path != ":memory:":
                if _engine is not None:
                    try:
                        _engine.dispose()
                    except Exception:
                        pass
                    _engine = None
                if _try_clear_wal_sidecars(raw_path):
                    logger.warning(
                        "Cleared a corrupt WAL/SHM sidecar after a disk I/O "
                        "error; main file untouched. Retrying."
                    )
                else:
                    logger.warning(f"Database connection attempt failed: {e}. Retrying...")
                raise
        # Fallback for header-level corruption that the raw pre-probe above
        # could not catch because the engine already existed (e.g. the file
        # was replaced/corrupted after the first successful connect). Move the
        # file aside and rebuild instead of retrying 5x and giving up.
        if is_sqlite and any(sig in str(e).lower() for sig in _SQLITE_CORRUPTION_SIGNATURES):
            logger.warning(
                f"SQLite connection failed with a corruption signature: {e}. "
                f"Moving aside and rebuilding."
            )
            _recovery_from_corruption()
            corruption_detected = True
            raise
        logger.warning(f"Database connection attempt failed: {e}. Retrying...")
        raise


_SQLITE_CORRUPTION_SIGNATURES = (
    "file is not a database",
    "database disk image is malformed",
    "database is corrupt",
    "not a database",
)


def _try_clear_wal_sidecars(db_path: str) -> bool:
    """Remove ONLY the -wal/-shm sidecar files, never the main .db file, and
    confirm a fresh connection now succeeds.

    A corrupt WAL is by definition data that was never checkpointed into
    the main file -- deleting it can only lose whatever was in-flight at
    the crash, never anything that was actually committed. This is the
    surgical alternative to _recovery_from_corruption() (which moves the
    main file aside too) for the specific case where the main file is
    perfectly healthy and only its WAL/SHM siblings are the problem. See
    _probe_sqlite_file's disk-I/O-error branch.

    Bounded retry on the removal itself: on Windows, the connection that
    just failed with the disk I/O error can leave its OS file handle open
    on the -wal file for a short window even after engine.dispose() --
    reproduced directly (WinError 32 "used by another process" on every
    immediate attempt, but the file becomes removable moments later once
    the OS finishes releasing the handle). POSIX unlink() doesn't have this
    problem at all (a file can be removed while still open elsewhere), so
    this only matters on Windows -- the retry is cheap insurance either way.
    """
    import sqlite3
    for suffix in ("-wal", "-shm"):
        target = db_path + suffix
        for attempt in range(5):
            try:
                os.remove(target)
                logger.info(f"Removed SQLite sidecar file: {target}")
                break
            except FileNotFoundError:
                break
            except OSError as e:
                if attempt == 4:
                    logger.warning(f"Could not remove {target} after 5 attempts: {e}")
                else:
                    time.sleep(0.3)
    try:
        conn = sqlite3.connect(db_path, timeout=1.0)
        try:
            # PRAGMA schema_version (not SELECT 1) so a still-corrupt file
            # isn't reported as recovered -- see _probe_sqlite_file.
            conn.execute("PRAGMA schema_version")
        finally:
            conn.close()
        return True
    except Exception:
        return False


def _probe_sqlite_file(db_path: str) -> str | None:
    """Raw, unpooled connectivity/corruption probe using stdlib sqlite3
    directly -- deliberately NOT going through the SQLAlchemy engine/pool
    (see init_db()'s CR-7 comment for why: a pooled connection that fails
    during the WAL-mode "connect" event can leak its OS file handle on
    Windows even after engine.dispose(), permanently blocking the rename
    this function's caller needs to do). Opens and closes in a plain
    try/finally so the handle is released deterministically either way.

    Returns a description of the corruption if detected (main file needs
    the full move-aside-and-rebuild), else None -- which also covers the
    case where a corrupt WAL/SHM sidecar was found and already cleared
    in-place below, since at that point the file genuinely is usable and
    calling the destructive full recovery on top would discard an intact
    main file for no reason. Conservative by design otherwise: any error
    whose text doesn't match a known corruption signature is treated as
    "not clearly corruption" (e.g. a locked file, a permissions issue) so a
    transient problem can't trigger a destructive rebuild.

    The probe query is `PRAGMA schema_version`, deliberately NOT
    `SELECT 1`. `SELECT 1` is answered without ever reading the schema, so
    on this Python 3.13 / SQLite 3.46 stack a file with a corrupt header
    (the realistic shape of hard-kill corruption) let `SELECT 1` succeed
    while every schema read failed with "file is not a database". The
    corruption then only surfaced later from SQLAlchemy's WAL-mode "connect"
    pool event (which runs `PRAGMA journal_mode=WAL`), past the point this
    pre-probe could move the file aside -- so init_db() retried 5x and gave
    up permanently instead of rebuilding. `PRAGMA schema_version` forces
    SQLite to read the header + schema on the raw stdlib connection, so it
    detects the same corruption this probe exists to catch. Reproduced
    2026-10-06 (test_corrupt_db_file_is_rebuilt_not_crash_looped).
    """
    import sqlite3
    try:
        conn = sqlite3.connect(db_path, timeout=1.0)
        try:
            conn.execute("PRAGMA schema_version")
        finally:
            conn.close()
        return None
    except sqlite3.DatabaseError as e:
        msg = str(e).lower()
        if "disk i/o error" in msg:
            # SQLite needs to read/replay the WAL on ANY connection to a
            # WAL-mode database, even a plain SELECT 1 -- a corrupt -wal/-shm
            # sidecar with an otherwise-healthy main file fails with exactly
            # this error, not the header-corruption signatures below, and
            # previously fell through to an unrecoverable retry-then-crash-
            # loop (reproduced 3/3 runs: init_db() retried 5x then gave up
            # permanently, with nothing ever clearing the poisoned WAL).
            # Found via an external crash-recovery audit, root-caused and
            # fixed 2026-09-22.
            logger.warning(
                f"SQLite connection failed with a disk I/O error (likely a "
                f"corrupt WAL/SHM sidecar, not main-file corruption): {e}. "
                f"Removing WAL/SHM siblings and retrying."
            )
            if _try_clear_wal_sidecars(db_path):
                logger.info("WAL/SHM sidecar removal recovered the database; main file untouched.")
                return None
            logger.warning("WAL/SHM sidecar removal did not fix it -- falling back to full corruption recovery.")
            return str(e)
        if any(sig in msg for sig in _SQLITE_CORRUPTION_SIGNATURES):
            return str(e)
        return None
    except Exception:
        return None


def get_db_session() -> Session:
    """Get a database session."""
    return get_session_factory()()


# ---------------------------------------------------------------------------
# Adaptive meta-learner decision-snapshot persistence (all fail-safe: any DB
# error is logged and swallowed so trading is never blocked by logging).
# ---------------------------------------------------------------------------

def save_decision_snapshot(
    *,
    decision_id: str,
    symbol: str,
    regime: str,
    final_action: str,
    confidence: float,
    size_multiplier: float,
    entry_price: float,
    qty: float,
    brain_votes: dict[str, str],
    feature_snapshot_json: str = "{}",
    causal_reasoning_json: str = "{}",
    tensor_state_json: str = "{}"
) -> bool:
    """Persist a committee decision at entry. Returns True on success."""
    try:
        _ensure_tables()
        with get_db_session() as session:
            # Data-integrity upsert: if an open snapshot already exists for this
            # symbol (e.g. a stale-positions "new entry" fired before the previous
            # cycle's fills settled), fold the new entry into it instead of
            # inserting a duplicate open snapshot.
            existing = (
                session.query(DecisionSnapshot)
                .filter(
                    DecisionSnapshot.symbol == symbol,
                    DecisionSnapshot.status == "open",
                )
                .order_by(DecisionSnapshot.created_at.desc())
                .first()
            )
            if existing is not None:
                new_qty = float(existing.qty) + float(qty)
                if new_qty > 0 and float(existing.entry_price) > 0 and float(entry_price) > 0:
                    existing.entry_price = (
                        float(existing.entry_price) * float(existing.qty)
                        + float(entry_price) * float(qty)
                    ) / new_qty
                existing.qty = new_qty
                session.commit()
                logger.warning(
                    f"[DATA_INTEGRITY] save_decision_snapshot: open snapshot "
                    f"{existing.decision_id} already exists for {symbol}; folded "
                    f"entry {decision_id} into it (no duplicate created)."
                )
                return True
            snap = DecisionSnapshot(
                decision_id=decision_id,
                symbol=symbol,
                regime=regime,
                final_action=final_action,
                confidence=float(confidence),
                size_multiplier=float(size_multiplier),
                entry_price=float(entry_price),
                qty=float(qty),
                votes_json=json.dumps(brain_votes or {}),
                feature_snapshot_json=feature_snapshot_json,
                causal_reasoning_json=causal_reasoning_json,
                tensor_state_json=tensor_state_json,
                status="open",
            )
            session.merge(snap)
            session.commit()
        return True
    except Exception as e:
        logger.warning(f"save_decision_snapshot failed (non-fatal): {e}")
        return False


def save_decision_snapshots_batch(
    records: list[dict[str, Any]]
) -> int:
    """Persist multiple decision snapshots in a single DB session.

    Each record has the same keys as save_decision_snapshot().
    Returns the number of snapshots saved."""
    if not records:
        return 0
    try:
        _ensure_tables()
        with get_db_session() as session:
            for rec in records:
                snap = DecisionSnapshot(
                    decision_id=rec["decision_id"],
                    symbol=rec["symbol"],
                    regime=rec.get("regime", "default"),
                    final_action=rec.get("final_action", "hold"),
                    confidence=float(rec.get("confidence", 0.0)),
                    size_multiplier=float(rec.get("size_multiplier", 1.0)),
                    entry_price=float(rec.get("entry_price", 0.0)),
                    qty=float(rec.get("qty", 0.0)),
                    votes_json=json.dumps(rec.get("brain_votes", {}) or {}),
                    feature_snapshot_json=rec.get("feature_snapshot_json", "{}"),
                    causal_reasoning_json=rec.get("causal_reasoning_json", "{}"),
                    tensor_state_json=rec.get("tensor_state_json", "{}"),
                    status="open",
                )
                session.merge(snap)
            session.commit()
        return len(records)
    except Exception as e:
        logger.warning(f"save_decision_snapshots_batch failed (non-fatal): {e}")
        return 0


def get_closed_decision_snapshots(limit: int = 2000, symbol: str | None = None) -> list[dict[str, Any]]:
    """Return closed decision snapshots (real live/paper trade outcomes), most
    recent first, in the exact shape MetaDecisionEnv expects (see
    scripts/evolutionary_ppo_trainer.py's synthetic snapshot construction).

    This is what lets the PPO Meta-Learner train on genuine forward-test
    results alongside its synthetic backtest data, rather than exclusively
    on backtests. Read-only and purely additive: never touches trading.
    """
    try:
        with get_db_session() as session:
            stmt = (
                select(DecisionSnapshot)
                .where(DecisionSnapshot.status == "closed")
            )
            if symbol:
                stmt = stmt.where(DecisionSnapshot.symbol == symbol)
            stmt = stmt.order_by(DecisionSnapshot.closed_at.desc()).limit(limit)

            rows = session.execute(stmt).scalars().all()
            results = []
            for row in rows:
                try:
                    results.append({
                        "symbol": row.symbol,
                        "regime": row.regime,
                        "final_action": row.final_action,
                        "confidence": row.confidence,
                        "brain_votes": json.loads(row.votes_json or "{}"),
                        "features": json.loads(row.feature_snapshot_json or "{}"),
                        "entry_time": row.created_at.isoformat() if row.created_at else None,
                        "exit_time": row.closed_at.isoformat() if row.closed_at else None,
                        "realized_pnl": row.realized_pnl,
                        # Read-only additions for the adaptive-learner startup
                        # bootstrap (bot._bootstrap_adaptive_learner_from_history):
                        # return_pct feeds the validation gate's return history
                        # directly; entry_price/qty let it reconstruct the
                        # live-path return for legacy rows closed before
                        # return_pct was recorded.
                        "return_pct": row.return_pct,
                        "entry_price": row.entry_price,
                        "qty": row.qty,
                        # Accounting trust label: "reconciled", "unreconciled",
                        # or None (historical/unknown). "reconciled" means the
                        # P&L was rebuilt from linked ledger fills; it does NOT
                        # mean the fees are exchange-verified. Consumers that
                        # build training data or statistics SHOULD exclude
                        # anything not "reconciled", but this is not enforced
                        # here -- verify each consumer, since some (e.g. the
                        # rolling loss limit) intentionally use all closed rows.
                        "outcome_status": row.outcome_status,
                    })
                except Exception as e:
                    logger.warning(f"Skipping malformed decision snapshot {row.decision_id}: {e}")
            return results
    except Exception as e:
        logger.warning(f"get_closed_decision_snapshots failed (non-fatal): {e}")
        return []


def get_open_snapshot(symbol: str) -> dict[str, Any] | None:
    """Return the most recent open decision snapshot for a symbol, as a dict.

    Results are cached in-memory per symbol for the duration of the
    trading cycle. The cache is invalidated when close_decision_snapshot()
    is called for the same symbol."""
    try:
        # Check the in-memory cache first
        cached = _open_snapshot_cache.get(symbol)
        if cached is not None:
            return cached

        with get_db_session() as session:
            stmt = (
                select(DecisionSnapshot)
                .where(DecisionSnapshot.symbol == symbol, DecisionSnapshot.status == "open")
                .order_by(DecisionSnapshot.created_at.desc())
            )
            rows = session.execute(stmt).scalars().all()
            if not rows:
                return None
            row = rows[0]
            if len(rows) > 1:
                # Invariant violation: the intended pattern folds every
                # scale-in into the SAME snapshot (update_decision_snapshot_position
                # above), so there should never be more than one open row per
                # symbol. If this fires, a race created two open snapshots for
                # the same symbol, and outcome attribution below is picking
                # "most recent" as a best-effort guess rather than a verified
                # match -- exactly the mismatch risk flagged in
                # ADVERSARIAL_AUDIT_2026-09-20.md §12. Surfaced loudly instead
                # of silently trusting the guess, since this poisons the
                # adaptive learner's training signal if wrong.
                # logger.error (not an async alert) deliberately: this function
                # is always called via asyncio.to_thread() from bot.py, i.e. it
                # runs in a worker thread with no running event loop, so
                # asyncio.create_task() here would silently fail. The
                # structured [DATA_INTEGRITY] prefix makes this greppable /
                # alertable from log-shipping infra without needing an
                # event-loop-safe call from inside a thread.
                logger.error(
                    f"[DATA_INTEGRITY] {len(rows)} simultaneous open decision "
                    f"snapshots found for {symbol} (ids={[r.decision_id for r in rows]}); "
                    f"using most recent ({row.decision_id}). This should never happen -- "
                    f"a scale-in race likely created a duplicate snapshot instead of "
                    f"updating the existing one."
                )
            # sqlite DateTime columns round-trip offset-naive; a naive
            # timestamp in this column is always UTC. Emit an aware ISO
            # string so consumers subtracting datetime.now(timezone.utc)
            # don't hit a naive/aware TypeError (silently swallowed).
            created_iso = None
            if row.created_at is not None:
                created_dt = row.created_at
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=datetime.UTC)
                created_iso = created_dt.isoformat()
            result = {
                "decision_id": row.decision_id,
                "symbol": row.symbol,
                "regime": row.regime,
                "final_action": row.final_action,
                "confidence": row.confidence,
                "entry_price": row.entry_price,
                "qty": row.qty,
                "brain_votes": json.loads(row.votes_json or "{}"),
                "feature_snapshot": json.loads(row.feature_snapshot_json or "{}"),
                "tensor_state": json.loads(row.tensor_state_json or "{}").get("transformer"),
                "created_at": created_iso,
            }
            _open_snapshot_cache[symbol] = result
            return result
    except Exception as e:
        logger.warning(f"get_open_snapshot failed (non-fatal): {e}")
        return None


def update_decision_snapshot_position(
    decision_id: str,
    *,
    entry_price: float,
    qty: float,
) -> bool:
    """Update an open snapshot's entry_price/qty after a scale-in fill.

    Audit finding F-A: the snapshot originally records the FIRST entry only;
    on a scale-in add the bot folds the add into the same snapshot (weighted
    average entry, summed qty) so the exit math in _record_committee_outcome
    matches the exchange's actual position. Only updates status='open' rows --
    a closed snapshot is never mutated.
    """
    try:
        with get_db_session() as session:
            row = session.get(DecisionSnapshot, decision_id)
            if row is None or row.status != "open":
                return False
            row.entry_price = float(entry_price)
            row.qty = float(qty)
            session.commit()
        # Keep the in-memory cache consistent with the updated row.
        try:
            with get_db_session() as session:
                row = session.get(DecisionSnapshot, decision_id)
                if row is not None:
                    _open_snapshot_cache.pop(row.symbol, None)
        except Exception:
            pass
        return True
    except Exception as e:
        logger.warning(f"update_decision_snapshot_position failed (non-fatal): {e}")
        return False


def get_all_open_snapshots() -> list[dict[str, Any]]:
    """Return minimal info for ALL open decision snapshots.

    Used by the startup reconciliation pass (bot.reconcile_open_snapshots) to
    find ghost snapshots left open by a crash or by a position that was closed
    outside the bot while it was down. Read-only.
    """
    try:
        with get_db_session() as session:
            stmt = select(DecisionSnapshot).where(DecisionSnapshot.status == "open")
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "decision_id": row.decision_id,
                    "symbol": row.symbol,
                    "final_action": row.final_action,
                    "entry_price": row.entry_price,
                    "qty": row.qty,
                }
                for row in rows
            ]
    except Exception as e:
        # Deliberately re-raised, not swallowed to []: this is the ONLY
        # caller (bot.reconcile_open_snapshots), which needs to tell "no
        # open snapshots" apart from "couldn't read the DB to find out".
        # Silently returning [] here made a DB error indistinguishable from
        # a genuinely empty result -- reconciliation would proceed as if
        # nothing were open, skipping the ghost-close and orphan-check
        # passes without ever surfacing that anything went wrong. Found via
        # an external crash-recovery audit, 2026-09-22.
        logger.warning(f"get_all_open_snapshots failed: {e}")
        raise


def close_decision_snapshot(
    decision_id: str,
    *,
    realized_pnl: float,
    return_pct: float = 0.0,
    holding_period_sec: float = 0.0,
    exit_reason: str | None = None,
    max_favorable_pct: float = 0.0,
    max_adverse_pct: float = 0.0,
    outcome_status: str | None = None,
) -> bool:
    """Mark a snapshot closed and record its realized outcome.

    ``outcome_status`` is the accounting trust label ("reconciled" /
    "unreconciled"). Callers that cannot certify the P&L should pass
    "unreconciled" so consumers CAN filter the row; leaving it None keeps the
    row "unknown" (the historical default). Note this is a data-quality label,
    not an exclusion guarantee -- it does not by itself remove the row from any
    calculation (the rolling loss limit, for one, sums every closed row).

    Also invalidates the in-memory open-snapshot cache for the
    affected symbol so that a subsequent get_open_snapshot() call
    returns the updated (closed) state."""
    try:
        # Invalidate the in-memory cache for this symbol.
        # We need to find the symbol first.
        try:
            with get_db_session() as session:
                row = session.get(DecisionSnapshot, decision_id)
                if row is not None:
                    _open_snapshot_cache.pop(row.symbol, None)
        except Exception:
            pass

        # Mark the snapshot closed with realized outcome.
        with get_db_session() as session:
            row = session.get(DecisionSnapshot, decision_id)
            if row is None:
                return False
            row.status = "closed"
            row.realized_pnl = float(realized_pnl)
            row.return_pct = float(return_pct)
            row.holding_period_sec = float(holding_period_sec)
            if exit_reason is not None:
                row.exit_reason = exit_reason
            row.max_favorable_pct = float(max_favorable_pct)
            row.max_adverse_pct = float(max_adverse_pct)
            row.outcome_status = outcome_status
            row.closed_at = datetime.datetime.now(datetime.UTC)
            session.commit()
        return True
    except Exception as e:
        logger.warning(f"close_decision_snapshot failed (non-fatal): {e}")
        return False


def get_recent_realized_pnl(hours: float) -> float:
    """Sum realized P&L of decision snapshots closed within the last ``hours``.

    Used by the rolling soft loss-limit entry block (bot.py). Fail-safe:
    returns 0.0 on any error so trading is never blocked by a bookkeeping bug.

    LIMITATION (2026-10-25 review, deliberate and unchanged): this sums EVERY
    closed row regardless of ``outcome_status``, so a mix of reconciled and
    unreconciled rows -- and rows whose fees are estimated, not exchange-verified
    -- feeds the loss limit. That is intentional here: the loss limit must fail
    safe, and under-counting losses to avoid a block is the more dangerous error.
    The consequence is that reconstructed P&L directly informs a trading
    decision (whether to block a new entry), and it is NOT audited performance.
    Changing this to filter on ``outcome_status`` would change trading behavior
    and is therefore OUT OF SCOPE for the accounting patch -- it needs its own
    review and authorization.
    """
    try:
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=float(hours))
        with get_db_session() as session:
            total = (
                session.query(func.sum(DecisionSnapshot.realized_pnl))
                .filter(
                    DecisionSnapshot.status == "closed",
                    DecisionSnapshot.closed_at.isnot(None),
                    DecisionSnapshot.closed_at >= cutoff,
                )
                .scalar()
            )
            return float(total) if total is not None else 0.0
    except Exception as e:
        logger.warning(f"get_recent_realized_pnl failed (non-fatal, fail-open): {e}")
        return 0.0


def get_db_health() -> bool:
    """Quick health check: returns True if the database engine can connect."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return False
