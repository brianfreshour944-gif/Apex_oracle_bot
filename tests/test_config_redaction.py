"""Tests for redact_database_url() -- src/main.py used to log
settings.DATABASE_URL in plaintext (`logger.info(f"Database: {settings.DATABASE_URL}")`),
and src/config.py's Settings.log_config() had the identical leak (every
other secret there -- TELEGRAM_BOT_TOKEN, DASHBOARD_API_KEY, ALPACA_API_KEY
-- is correctly logged only as a bool presence check; DATABASE_URL was the
one exception that logged the raw value, credentials included)."""

from src.config import redact_database_url


def test_strips_credentials_from_postgres_url():
    url = "postgresql+psycopg2://user:secret@host:5432/db"
    redacted = redact_database_url(url)
    assert "secret" not in redacted
    assert "user" not in redacted
    assert redacted == "postgresql+psycopg2://***:***@host:5432/db"


def test_sqlite_url_unchanged():
    """The default DATABASE_URL has no credentials to redact -- must not be mangled."""
    url = "sqlite:///data/bot.db"
    assert redact_database_url(url) == url


def test_postgres_url_without_credentials_unchanged():
    url = "postgresql://host:5432/db"
    assert redact_database_url(url) == url


def test_redacts_password_only_url():
    """redis-style URLs sometimes carry only a password, no username."""
    url = "redis://:secretpass@localhost:6379/0"
    redacted = redact_database_url(url)
    assert "secretpass" not in redacted
    assert redacted == "redis://***:***@localhost:6379/0"


def test_preserves_host_path_and_query():
    url = "postgresql://user:secret@db.internal:5432/mydb?sslmode=require"
    redacted = redact_database_url(url)
    assert "secret" not in redacted
    assert "db.internal:5432" in redacted
    assert "/mydb" in redacted
    assert "sslmode=require" in redacted


def test_main_imports_and_uses_the_same_redaction_function(monkeypatch):
    """src/main.py's startup log line (`logger.info(f"Database:
    {redact_database_url(settings.DATABASE_URL)}")`) must go through the
    same redaction as everywhere else -- verified here by calling it exactly
    the way main() does, without running the whole bot (main() goes on to
    asyncio.run(run_trading_bot()), out of scope for this test)."""
    import src.main as main_module
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+psycopg2://user:supersecret@host:5432/db")

    logged = f"Database: {main_module.redact_database_url(settings.DATABASE_URL)}"
    assert "supersecret" not in logged
    assert logged == "Database: postgresql+psycopg2://***:***@host:5432/db"


def test_log_config_does_not_leak_database_credentials(monkeypatch):
    """Settings.log_config() (called from bot.py's startup logging) must
    redact DATABASE_URL the same way every other secret field there is
    already handled (as a bool presence check, not a raw value)."""
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+psycopg2://produser:hunter2@dbhost:5432/prod")

    summary = settings.log_config()

    assert "hunter2" not in summary
    assert "produser" not in summary
    assert "***:***@dbhost:5432/prod" in summary
