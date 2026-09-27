"""Verify the configured DATABASE_URL is a well-formed, supported URL."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_database_configuration():
    """Assert the configured DATABASE_URL parses and uses a supported scheme.

    The bot ships with a SQLite default (sqlite:///data/bot.db) and optionally
    runs on PostgreSQL, so either scheme is valid -- what must not happen is an
    unparseable or empty URL silently falling through to a broken connection.
    """
    from src.config import settings

    url = settings.DATABASE_URL
    assert url, "DATABASE_URL must not be empty"
    assert url.startswith(("sqlite:///", "postgresql://", "postgresql+psycopg2://")), (
        f"unsupported DATABASE_URL scheme: {url}"
    )

    if url.startswith("postgresql"):
        # A PostgreSQL URL must name a host and a database.
        assert "/" in url.rsplit("@", 1)[-1], f"PostgreSQL URL missing database name: {url}"
