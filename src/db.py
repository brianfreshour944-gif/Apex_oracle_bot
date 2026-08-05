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
            _engine = create_engine(
                settings.DATABASE_URL,
                pool_recycle=3600,
                echo=False,
                future=True,
                pool_size=10,
                max_overflow=20,
                connect_args={"connect_timeout": 5},
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
    causal_reasoning_json: str = "{}"
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
                status="open",
            )
            session.merge(snap)
            session.commit()
        return True
    except Exception as e:
        logger.warning(f"save_decision_snapshot failed (non-fatal): {e}")
        return False


def get_open_snapshot(symbol: str) -> Optional[Dict[str, Any]]:
    """Return the most recent open decision snapshot for a symbol, as a dict."""
    try:
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
            return {
                "decision_id": row.decision_id,
                "symbol": row.symbol,
                "regime": row.regime,
                "final_action": row.final_action,
                "confidence": row.confidence,
                "entry_price": row.entry_price,
                "qty": row.qty,
                "brain_votes": json.loads(row.votes_json or "{}"),
                "feature_snapshot": json.loads(row.feature_snapshot_json or "{}"),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
    except Exception as e:
        logger.warning(f"get_open_snapshot failed (non-fatal): {e}")
        return None


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
    """Mark a snapshot closed and record its realized outcome."""
    try:
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
