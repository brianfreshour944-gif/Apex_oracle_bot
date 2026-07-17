"""Modern database layer using SQLAlchemy 2.0 with MappedAsDataclass style."""

import datetime
import logging
import time
from typing import List, Dict, Any, Optional, Sequence
from sqlalchemy import (
    create_engine,
    text,
    String,
    Float,
    DateTime,
    Integer,
    select,
    func,
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

logger = logging.getLogger(__name__)

# Modern SQLAlchemy 2.0 DeclarativeBase
class Base(DeclarativeBase):
    """Modern SQLAlchemy 2.0 declarative base using MappedAsDataclass style."""
    pass

# Modern ORM models using MappedAsDataclass style
class TradeLog(Base):
    """Trade log table using modern SQLAlchemy 2.0 mapped_column style."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bot_name: Mapped[str] = mapped_column(String(50), index=True)
    exchange: Mapped[str] = mapped_column(String(50))
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(10))
    price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    value: Mapped[float] = mapped_column(Float)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, index=True
    )

class BotStatus(Base):
    """Bot status table using modern SQLAlchemy 2.0 mapped_column style."""

    __tablename__ = "bot_status"

    bot_name: Mapped[str] = mapped_column(String(50), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), default="running")
    starting_equity: Mapped[float] = mapped_column(Float)
    daily_starting_equity: Mapped[float] = mapped_column(Float)
    live_equity: Mapped[float] = mapped_column(Float)
    buying_power: Mapped[float] = mapped_column(Float, default=0.0)
    daily_pnl_pct: Mapped[float] = mapped_column(Float, default=0.0)
    open_positions_count: Mapped[int] = mapped_column(Integer, default=0)
    trades_today: Mapped[int] = mapped_column(Integer, default=0)
    live_equity_updated_at: Mapped[datetime.datetime] = mapped_column(DateTime)
    last_update: Mapped[datetime.datetime] = mapped_column(DateTime)

# Database engine and session factory
engine = create_engine(
    settings.DATABASE_URL,
    pool_recycle=3600,
    echo=False,
    future=True,
    pool_size=10,
    max_overflow=20,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

def init_db() -> None:
    """Initialize database schema using modern SQLAlchemy 2.0 approach."""
    try:
        Base.metadata.create_all(engine)
        ensure_columns()
        logger.info("Database schema checked/initialized successfully.")
    except SQLAlchemyError as e:
        logger.error(f"Database initialization failed: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _record_bot_error(bot_name: str, message: str) -> None:
    """Record bot errors with retry logic using tenacity."""
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO bot_errors (bot_name, error_message, timestamp)
                    VALUES (:bot_name, :msg, :ts)
                """),
                {"bot_name": bot_name, "msg": message, "ts": datetime.datetime.utcnow()},
            )
    except Exception as e:
        logger.error(f"Failed to record error to bot_errors table: {e}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def log_trade(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    pnl: Optional[float] = None,
    exchange: str = "Alpaca",
    fee: float = 0.0,
    order_id: Optional[str] = None,
    bot_name: str = settings.BOT_NAME,
    max_retries: int = 3,
) -> bool:
    """Log an executed trade to the trades table with modern error handling.

    Args:
        symbol: Trading symbol
        side: 'BUY' or 'SELL'
        quantity: Trade quantity
        price: Execution price
        pnl: Realized PnL (for SELL orders)
        exchange: Exchange name
        fee: Transaction fee
        order_id: Exchange order ID
        bot_name: Bot identifier
        max_retries: Maximum retry attempts

    Returns:
        True on success, False if failed after retries
    """
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        session: Session = SessionLocal()
        try:
            rec = TradeLog(
                bot_name=bot_name,
                exchange=exchange,
                symbol=symbol,
                side=side.upper(),
                price=float(price),
                quantity=float(quantity),
                value=float(quantity) * float(price),
                fee=float(fee),
                realized_pnl=float(pnl) if pnl is not None else None,
                order_id=str(order_id) if order_id else None,
                timestamp=datetime.datetime.utcnow(),
            )
            session.add(rec)
            session.commit()
            logger.info(f"DB Log: {side.upper()} {float(quantity):.6f} {symbol} @ {price}")
            return True
        except Exception as e:
            session.rollback()
            last_err = e
            logger.warning(
                f"log_trade attempt {attempt}/{max_retries} failed "
                f"({side.upper()} {quantity} {symbol} @ {price}): {e}"
            )
            if attempt < max_retries:
                time.sleep(0.5 * attempt)
        finally:
            session.close()

    # All retries exhausted
    msg = (
        f"FAILED TO LOG FILLED TRADE after {max_retries} attempts: "
        f"{side.upper()} {quantity} {symbol} @ {price} "
        f"(order_id={order_id}): {last_err}"
    )
    logger.critical(msg)
    _record_bot_error(bot_name, msg)
    return False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def update_bot_status(
    starting_equity: float,
    live_equity: float,
    buying_power: float = 0.0,
    daily_pnl_pct: float = 0.0,
    open_positions_count: int = 0,
    trades_today: int = 0,
    status: str = "running",
    bot_name: str = settings.BOT_NAME,
) -> None:
    """Upsert the bot status/equity record using modern SQLAlchemy 2.0."""
    session: Session = SessionLocal()
    try:
        now = datetime.datetime.utcnow()
        rec = session.get(BotStatus, bot_name)
        if rec is None:
            rec = BotStatus(
                bot_name=bot_name,
                starting_equity=float(starting_equity),
                daily_starting_equity=float(starting_equity),
            )
            session.add(rec)

        # starting_equity is the lifetime baseline: set once and never overwritten
        if rec.starting_equity is None:
            rec.starting_equity = float(starting_equity)

        # Keep daily_starting_equity populated on first creation
        if rec.daily_starting_equity is None:
            rec.daily_starting_equity = float(starting_equity)

        rec.status = status
        rec.live_equity = float(live_equity)
        rec.buying_power = float(buying_power)
        rec.daily_pnl_pct = float(daily_pnl_pct)
        rec.open_positions_count = int(open_positions_count)
        rec.trades_today = int(trades_today)
        rec.live_equity_updated_at = now
        rec.last_update = now
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to update bot status in DB: {e}")
        raise
    finally:
        session.close()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def reset_daily_starting_equity(equity: float, bot_name: str = settings.BOT_NAME) -> None:
    """Reset the persisted daily_starting_equity baseline."""
    session: Session = SessionLocal()
    try:
        rec = session.get(BotStatus, bot_name)
        if rec is None:
            rec = BotStatus(bot_name=bot_name, daily_starting_equity=float(equity))
            session.add(rec)
        rec.daily_starting_equity = float(equity)
        rec.last_update = datetime.datetime.utcnow()
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to reset daily starting equity: {e}")
        raise
    finally:
        session.close()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_recent_trades(bot_name: str = settings.BOT_NAME, limit: int = 30) -> List[Dict[str, Any]]:
    """Return recent trades as a list of dicts (newest first)."""
    session: Session = SessionLocal()
    try:
        stmt = (
            select(TradeLog)
            .where(TradeLog.bot_name == bot_name)
            .order_by(TradeLog.timestamp.desc())
            .limit(limit)
        )
        rows = session.execute(stmt).scalars().all()

        return [
            {
                "symbol": r.symbol,
                "side": r.side,
                "price": float(r.price),
                "quantity": float(r.quantity),
                "value": float(r.value),
                "timestamp": r.timestamp,
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error querying trades: {e}")
        return []
    finally:
        session.close()

def ensure_columns() -> None:
    """Ensure all required database columns exist."""
    with engine.connect() as conn:
        # List of (column_name, data_type, default)
        columns = [
            ('daily_starting_equity', 'REAL', 'NULL'),
            ('buying_power', 'REAL', '0.0'),
            ('daily_pnl_pct', 'REAL', '0.0'),
            ('open_positions_count', 'INTEGER', '0'),
            ('trades_today', 'INTEGER', '0'),
            ('live_equity_updated_at', 'TIMESTAMP', 'NULL'),
        ]

        for col, dtype, default in columns:
            # Check if column exists
            result = conn.execute(
                text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name='bot_status' AND column_name=:col
                """),
                {'col': col}
            )

            if not result.fetchone():
                conn.execute(
                    text(f"""
                        ALTER TABLE bot_status ADD COLUMN {col} {dtype} DEFAULT {default}
                    """)
                )
                conn.commit()
                logger.info(f"Added missing column: {col}")

def get_last_buy(symbol: str) -> Optional[TradeLog]:
    """Get the most recent BUY TradeLog row for a symbol."""
    session: Session = SessionLocal()
    try:
        stmt = (
            select(TradeLog)
            .where(TradeLog.symbol == symbol, TradeLog.side == "BUY")
            .order_by(TradeLog.timestamp.desc())
        )
        return session.execute(stmt).scalars().first()
    except Exception as e:
        logger.error(f"get_last_buy failed for {symbol}: {e}")
        return None
    finally:
        session.close()

def get_entry_price(symbol: str, fallback: float) -> float:
    """Get entry price for a symbol with fallback."""
    row = get_last_buy(symbol)
    return float(row.price) if row else float(fallback)