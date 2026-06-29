# db.py
import os
import logging
import datetime

from sqlalchemy import (
    create_engine, text, Column, Integer, String, Float, DateTime,
)
from sqlalchemy.orm import sessionmaker, declarative_base

from config import DATABASE_URL, BOT_NAME

logger = logging.getLogger(__name__)

# Ensure the directory for a SQLite file DB exists before the engine connects.
if DATABASE_URL.startswith("sqlite:///"):
    db_path = DATABASE_URL.replace("sqlite:///", "", 1)
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

# Setup Database connection pool.
engine = create_engine(DATABASE_URL, pool_recycle=3600, echo=False, future=True)
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
    qty = Column(Float)
    value = Column(Float)
    fee = Column(Float, default=0.0)
    pnl = Column(Float, nullable=True)
    order_id = Column(String(100), nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)


class BotStatus(Base):
    __tablename__ = "bot_status"

    bot_name = Column(String(50), primary_key=True)
    status = Column(String(20), default="running")
    starting_equity = Column(Float)
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
    logger.info("Database schema checked/initialized successfully.")


def log_trade(symbol: str, side: str, qty: float, price: float,
              pnl: float = None, exchange: str = "OKX", fee: float = 0.0,
              order_id: str = None, bot_name: str = BOT_NAME):
    """Logs an executed trade to the trades table."""
    session = SessionLocal()
    try:
        rec = TradeLog(
            bot_name=bot_name,
            exchange=exchange,
            symbol=symbol,
            side=side.upper(),
            price=float(price),
            qty=float(qty),
            value=float(qty) * float(price),
            fee=float(fee),
            pnl=float(pnl) if pnl is not None else None,
            order_id=str(order_id) if order_id else None,
            timestamp=datetime.datetime.utcnow(),
        )
        session.add(rec)
        session.commit()
        logger.info(f"DB Log: {side.upper()} {float(qty):.6f} {symbol} @ {price}")
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to log trade to DB: {e}")
    finally:
        session.close()


def update_bot_status(starting_equity: float, live_equity: float,
                      buying_power: float = 0.0, daily_pnl_pct: float = 0.0,
                      open_positions_count: int = 0, trades_today: int = 0,
                      status: str = "running", bot_name: str = BOT_NAME):
    """Upserts the bot status / equity record."""
    session = SessionLocal()
    try:
        now = datetime.datetime.utcnow()
        rec = session.get(BotStatus, bot_name)
        if rec is None:
            rec = BotStatus(bot_name=bot_name, starting_equity=float(starting_equity))
            session.add(rec)
        # starting_equity is set once and never overwritten afterwards.
        if rec.starting_equity is None:
            rec.starting_equity = float(starting_equity)
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
    """Resets the persisted starting_equity baseline (used for daily loss limit)."""
    session = SessionLocal()
    try:
        rec = session.get(BotStatus, bot_name)
        if rec is None:
            rec = BotStatus(bot_name=bot_name)
            session.add(rec)
        rec.starting_equity = float(equity)
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
                "qty": float(r.qty),
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
