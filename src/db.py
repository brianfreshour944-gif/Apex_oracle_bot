"""Modern database layer using SQLAlchemy 2.0 with MappedAsDataclass style."""

import datetime
import logging
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

# Modern ORM models would go here
# (Tradelog, BotStatus, etc.)

def init_db() -> None:
    """Initialize database connection."""
    try:
        engine = create_engine(
            settings.DATABASE_URL,
            pool_recycle=3600,
            echo=False,
            future=True,
            pool_size=10,
            max_overflow=20,
        )
        logger.info(f"Database connected: {settings.DATABASE_URL}")
    except SQLAlchemyError as e:
        logger.error(f"Database connection failed: {e}")
        raise

# Database session factory
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

def get_db_session() -> Session:
    """Get a database session."""
    return SessionLocal()