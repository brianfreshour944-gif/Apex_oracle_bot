"""Modern database layer using SQLAlchemy 2.0 with MappedAsDataclass style."""

import datetime
import logging
import os
from typing import Dict, Any, Optional, Sequence
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
from src.logging_config import get_logger

logger = get_logger(__name__)

# Modern SQLAlchemy 2.0 DeclarativeBase
class Base(DeclarativeBase):
    """Modern SQLAlchemy 2.0 declarative base using MappedAsDataclass style."""
    pass

# Engine will be created lazily when first needed
_engine = None

def get_engine():
    """Get the database engine, creating it if needed."""
    global _engine
    if _engine is None:
        if settings.DATABASE_URL.startswith("sqlite:///"):
            db_path = settings.DATABASE_URL[len("sqlite:///"):]
            if db_path and db_path != ":memory:":
                db_dir = os.path.dirname(db_path)
                if db_dir:
                    os.makedirs(db_dir, exist_ok=True)

        _engine = create_engine(
            settings.DATABASE_URL,
            pool_recycle=3600,
            echo=False,
            future=True,
            pool_size=10,
            max_overflow=20,
            connect_args={"connect_timeout": 5} if "postgresql" in settings.DATABASE_URL else {},
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
    try:
        # Test the connection
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Database connected: {settings.DATABASE_URL}")
    except SQLAlchemyError as e:
        logger.warning(f"Database connection attempt failed: {e}. Retrying...")
        raise


def get_db_session() -> Session:
    """Get a database session."""
    return get_session_factory()()
