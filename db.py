# db.py
import os
import time
import logging
import datetime

from sqlalchemy import (
    create_engine, text, Column, Integer, String, Float, DateTime,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from config import DATABASE_URL, BOT_NAME
import psycopg2

logger = logging.getLogger(__name__)

# Ensure the directory for a SQLite file DB exists before the engine connects.

# Setup Database connection pool.
engine = create_engine(DATABASE_URL, pool_recycle=3600, echo=False, future=True, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()

class TradeLog(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bot_name = Column(String(50), index=True)
    exchange = Column(String(50))
    symbol = Column(String(20), index=True)
    side = Column(String(10))
    price = Column(Float)
    quantity = Column(Float)
    value = Column(Float)
    fee = Column(Float, default=0.0)
    realized_pnl = Column(Float, nullable=True)
    order_id = Column(String(100), nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)

class BotStatus(Base):
    __tablename__ = "bot_status"

    bot_name = Column(String(50), primary_key=True)
    status = Column(String(20), default="running")
    # Lifetime baseline: set once at first launch, never overwritten.
    # Used for the lifetime drawdown killswitch.
    starting_equity = Column(Float)
    # Daily baseline: reset to the account equity at each UTC midnight.
    # Used for the daily loss-limit killswitch. Distinct from starting_equity
    # so the two drawdown mechanisms do not clobber each other.
    daily_starting_equity = Column(Float)
    live_equity = Column(Float)
    buying_power = Column(Float, default=0.0)
    daily_pnl_pct = Column(Float, default=0.0)
    open_positions_count = Column(Integer, default=0)
    trades_today = Column(Integer, default=0)
    live_equity_updated_at = Column(DateTime)
    last_update = Column(DateTime)

def init_db():
    """Creates tables if they do not exist."""
    Base.metadata.create_all(engine)
    ensure_columns()
    logger.info("Database schema checked/initialized successfully.")

def _record_bot_error(bot_name: str, message: str):
    """Best-effort write to bot_errors so a failure is visible in the
    dashboard's Error Log tab instead of only sitting in a local log file.
    Wrapped in its own try/except -- if the DB is unreachable this must not
    raise, or it'd mask the original failure it's trying to report."""
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO bot_errors (bot_name, error_message, timestamp) "
                     "VALUES (:bot_name, :msg, :ts)"),
                {"bot_name": bot_name, "msg": message,
                 "ts": datetime.datetime.utcnow()},
            )
    except Exception as e:
        logger.error(f"Also failed to record error to bot_errors table: {e}")


def log_trade(symbol: str, side: str, quantity: float, price: float,
              pnl: float = None, exchange: str = "Alpaca", fee: float = 0.0,
              order_id: str = None, bot_name: str = BOT_NAME,
              max_retries: int = 3) -> bool:
    """Logs an executed trade to the trades table.

    IMPORTANT: by the time this is called the order has *already filled* on
    the exchange -- there is no "undo" here. A trade that fails to persist
    becomes invisible inventory: the dashboard's FIFO engine never learns
    the position was opened/closed, so calculated equity silently drifts
    away from the real account balance with no error shown anywhere. That
    class of bug is why this now retries and, on final failure, records the
    failure so a human can reconcile the position manually. Returns True on
    success, False if it could not be persisted after retries -- callers
    should escalate loudly (page/alert) on False rather than continue
    silently, since this is the one failure mode that has no audit trail
    anywhere else.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        session = SessionLocal()
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

    # All retries exhausted: this trade is now a real, filled position with
    # zero record of it. Log at CRITICAL, not ERROR -- silent failure here
    # is exactly what causes live-vs-calculated equity drift.
    msg = (f"FAILED TO LOG FILLED TRADE after {max_retries} attempts: "
           f"{side.upper()} {quantity} {symbol} @ {price} "
           f"(order_id={order_id}): {last_err}")
    logger.critical(msg)
    _record_bot_error(bot_name, msg)
    return False

def update_bot_status(starting_equity: float, live_equity: float,
                      buying_power: float = 0.0, daily_pnl_pct: float = 0.0,
                      open_positions_count: int = 0, trades_today: int = 0,
                      status: str = "running", bot_name: str = BOT_NAME):
    """Upserts the bot status / equity record.

    `starting_equity` is the lifetime baseline and is set once, never
    overwritten. The daily baseline lives in `daily_starting_equity` and is
    managed separately by `reset_daily_starting_equity`.
    """
    session = SessionLocal()
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
        # starting_equity is the lifetime baseline: set once and never
        # overwritten afterwards.
        if rec.starting_equity is None:
            rec.starting_equity = float(starting_equity)
        # Keep daily_starting_equity populated on first creation so the daily
        # loss check has a sane baseline even before the first midnight reset.
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
    finally:
        session.close()

def reset_daily_starting_equity(equity: float, bot_name: str = BOT_NAME):
    """Resets the persisted daily_starting_equity baseline (used for the daily
    loss limit). This writes ONLY the daily column and never touches the
    lifetime `starting_equity` baseline."""
    session = SessionLocal()
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
    finally:
        session.close()

def query_recent_trades(bot_name: str = BOT_NAME, limit: int = 30) -> list:
    """Returns recent trades as a list of dicts (newest first)."""
    session = SessionLocal()
    try:
        rows = (
            session.query(TradeLog)
            .filter(TradeLog.bot_name == bot_name)
            .order_by(TradeLog.timestamp.desc())
            .limit(limit)
            .all()
        )
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

def ensure_columns():
    engine = create_engine(DATABASE_URL)
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
            result = conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name='bot_status' AND column_name=:col
            """), {'col': col})
            if not result.fetchone():
                conn.execute(text(f"""
                    ALTER TABLE bot_status ADD COLUMN {col} {dtype} DEFAULT {default}
                """))
                conn.commit()
                logger.info(f"Added missing column: {col}")
