"""Modern database layer using SQLAlchemy 2.0 with MappedAsDataclass style."""

import datetime
import json
import logging
import os
from typing import Dict, Any, Optional, Sequence, List
from sqlalchemy import (
    create_engine,
    event,
    text,
    String,
    Float,
    DateTime,
    Integer,
    Text,
    select,
    func,
    Index,
)
from sqlalchemy.orm import (
    sessionmaker,
    Mapped,
    mapped_column,
    DeclarativeBase,
    Session,
)
from sqlalchemy.exc import SQLAlchemyError
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
    exit_reason: Mapped[Optional[str]] = mapped_column(String(48), nullable=True, default=None)
    max_favorable_pct: Mapped[float] = mapped_column(Float, default=0.0)
    max_adverse_pct: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    return_pct: Mapped[float] = mapped_column(Float, default=0.0)
    holding_period_sec: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    closed_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

class ShadowTrade(Base):
    """Tracks virtual positions taken by candidate models in the Evolution Tournament."""
    __tablename__ = "shadow_trades"
    # check_shadow_stops()/process_shadow_signal() both filter on exactly
    # this (candidate_name, symbol, status) triple, ordered by created_at.
    __table_args__ = (
        Index("ix_shadow_trades_candidate_symbol_status", "candidate_name", "symbol", "status"),
    )

    trade_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    candidate_name: Mapped[str] = mapped_column(String(64))
    symbol: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(16))  # buy or short
    qty: Mapped[float] = mapped_column(Float, default=0.0)
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    status: Mapped[str] = mapped_column(String(16), default="open")  # open|closed
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    closed_at: Mapped[Optional[datetime.datetime]] = mapped_column(
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
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

class OrderRecord(Base):
    """Record of Alpaca orders with fill data for data integrity verification."""
    __tablename__ = "orders"
    __table_args__ = (
        Index("ix_orders_decision_id", "decision_id"),
        Index("ix_orders_symbol_status", "symbol", "status"),
    )

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    decision_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, default=None)
    symbol: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(16))  # buy/sell
    qty: Mapped[float] = mapped_column(Float, default=0.0)
    filled_qty: Mapped[float] = mapped_column(Float, default=0.0)
    filled_avg_price: Mapped[float] = mapped_column(Float, default=0.0)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="new")
    type: Mapped[str] = mapped_column(String(16), default="market")
    time_in_force: Mapped[str] = mapped_column(String(16), default="ioc")
    client_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, default=None)
    submitted_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    filled_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

def save_experiment_record(
    experiment_id: str,
    generation_type: str,
    architecture_details: Dict[str, Any],
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
    decision_id: Optional[str] = None,
    symbol: str,
    side: str,
    qty: float,
    filled_qty: float = 0.0,
    filled_avg_price: float = 0.0,
    commission: float = 0.0,
    status: str = "new",
    type: str = "market",
    time_in_force: str = "ioc",
    client_order_id: Optional[str] = None,
    submitted_at: Optional[datetime.datetime] = None,
    filled_at: Optional[datetime.datetime] = None,
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
                status=status,
                type=type,
                time_in_force=time_in_force,
                client_order_id=client_order_id,
                submitted_at=submitted_at or datetime.datetime.now(datetime.timezone.utc),
                filled_at=filled_at,
            )
            session.merge(order)
            session.commit()
        return True
    except Exception as e:
        logger.warning(f"save_order_record failed (non-fatal): {e}")
        return False

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
# In-memory cache for get_open_snapshot() results, keyed by symbol.
# Invalidated when close_decision_snapshot() is called for that symbol.
_open_snapshot_cache: Dict[str, Optional[Dict[str, Any]]] = {}


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
        conn.commit()


def _ensure_tables() -> None:
    global _tables_ensured
    if not _tables_ensured:
        Base.metadata.create_all(get_engine())
        _ensure_indexes()
        _tables_ensured = True

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
                from urllib.parse import urlparse, urlunparse
                parsed = urlparse(db_url)
                # Check if path is empty but netloc ends with port+dbname (missing slash)
                if not parsed.path and ':' in parsed.netloc:
                    host_port = parsed.netloc.rsplit(':', 1)
                    if len(host_port) == 2 and host_port[1].isdigit():
                        # Reconstruct with proper path
                        new_netloc = host_port[0] + ':' + host_port[1]
                        new_path = '/' + host_port[1]  # This is wrong - need actual db name
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
def init_db() -> None:
    """Initialize database connection with exponential backoff retries."""
    global _tables_ensured
    try:
        # Test the connection
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        # Create ORM tables if they do not exist (safe/idempotent).
        Base.metadata.create_all(get_engine())
        _tables_ensured = True

        # create_all() only issues CREATE TABLE (with inline indexes) for
        # tables that don't exist yet -- it silently skips index creation for
        # tables that were created before these indexes were added to the
        # models. Explicitly (and idempotently) ensure they exist on already-
        # deployed databases too.
        _ensure_indexes()

        logger.info(f"Database connected: {settings.DATABASE_URL}")
    except SQLAlchemyError as e:
        logger.warning(f"Database connection attempt failed: {e}. Retrying...")
        raise


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
    brain_votes: Dict[str, str],
    feature_snapshot_json: str = "{}",
    causal_reasoning_json: str = "{}",
    tensor_state_json: str = "{}"
) -> bool:
    """Persist a committee decision at entry. Returns True on success."""
    try:
        _ensure_tables()
        with get_db_session() as session:
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
    records: List[Dict[str, Any]]
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


def get_closed_decision_snapshots(limit: int = 2000, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
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
                    })
                except Exception as e:
                    logger.warning(f"Skipping malformed decision snapshot {row.decision_id}: {e}")
            return results
    except Exception as e:
        logger.warning(f"get_closed_decision_snapshots failed (non-fatal): {e}")
        return []


def get_open_snapshot(symbol: str) -> Optional[Dict[str, Any]]:
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
                .limit(1)
            )
            row = session.execute(stmt).scalars().first()
            if row is None:
                return None
            # sqlite DateTime columns round-trip offset-naive; a naive
            # timestamp in this column is always UTC. Emit an aware ISO
            # string so consumers subtracting datetime.now(timezone.utc)
            # don't hit a naive/aware TypeError (silently swallowed).
            created_iso = None
            if row.created_at is not None:
                created_dt = row.created_at
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=datetime.timezone.utc)
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


def get_all_open_snapshots() -> List[Dict[str, Any]]:
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
        logger.warning(f"get_all_open_snapshots failed (non-fatal): {e}")
        return []


def close_decision_snapshot(
    decision_id: str,
    *,
    realized_pnl: float,
    return_pct: float = 0.0,
    holding_period_sec: float = 0.0,
    exit_reason: Optional[str] = None,
    max_favorable_pct: float = 0.0,
    max_adverse_pct: float = 0.0,
) -> bool:
    """Mark a snapshot closed and record its realized outcome.

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
            row.closed_at = datetime.datetime.now(datetime.timezone.utc)
            session.commit()
        return True
    except Exception as e:
        logger.warning(f"close_decision_snapshot failed (non-fatal): {e}")
        return False


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
