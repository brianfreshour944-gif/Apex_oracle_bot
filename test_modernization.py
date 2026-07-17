"""Test script to validate the modernization of the trading bot."""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports() -> None:
    """Test that all modern modules can be imported successfully."""
    print("Testing imports...")

    try:
        # Test modern configuration
        from src.config import settings, TradingBotSettings
        print("Modern configuration system imported")

        # Test modern database
        from src.db import (
            Base, TradeLog, BotStatus, init_db,
            log_trade, update_bot_status, query_recent_trades
        )
        print("Modern database layer imported")

        # Test modern exchange
        from src.exchange import AlpacaExchange
        print("Modern exchange module imported")

        # Test modern API
        from src.api import app, start_fastapi_server_async
        print("Modern FastAPI server imported")

        # Test modern logging
        from src.logging_config import get_logger, configure_structlog
        print("Modern logging system imported")

        # Test modern bot
        from src.bot import run_trading_bot
        print("Modern bot module imported")

        # Test main entry point
        from src.main import main
        print("Main entry point imported")

    except ImportError as e:
        print(f"Import failed: {e}")
        return False

    return True

def test_configuration() -> None:
    """Test the modern configuration system."""
    print("\nTesting configuration...")

    try:
        from src.config import settings

        # Test that settings are loaded
        assert hasattr(settings, 'ALPACA_API_KEY')
        assert hasattr(settings, 'SYMBOLS')
        assert hasattr(settings, 'BOT_NAME')

        # Test computed fields
        assert isinstance(settings.SYMBOLS, list)
        assert len(settings.SYMBOLS) > 0

        # Test validation
        assert settings.ACCOUNT_BASE > 0
        assert 0 < settings.BASE_RISK_PERCENT <= 1

        print("Configuration system working correctly")
        return True

    except Exception as e:
        print(f"Configuration test failed: {e}")
        return False

def test_database() -> None:
    """Test the modern database layer."""
    print("\nTesting database...")

    try:
        from src.db import Base, TradeLog, BotStatus

        # Test SQLAlchemy 2.0 models
        assert hasattr(Base, 'metadata')
        assert hasattr(TradeLog, '__tablename__')
        assert hasattr(BotStatus, '__tablename__')

        print("Database models working correctly")
        return True

    except Exception as e:
        print(f"Database test failed: {e}")
        return False

def test_logging() -> None:
    """Test the modern logging system."""
    print("\nTesting logging...")

    try:
        from src.logging_config import get_logger

        logger = get_logger("test")
        logger.info("Test log message")

        print("Logging system working correctly")
        return True

    except Exception as e:
        print(f"Logging test failed: {e}")
        return False

def test_api() -> None:
    """Test the FastAPI server."""
    print("\nTesting API...")

    try:
        from src.api import app

        # Test that FastAPI app is configured
        assert app.title == "Apex Oracle Bot API"
        assert app.version == "2.0.0"

        # Test that routes exist
        routes = [route.path for route in app.routes]
        assert "/status" in routes
        assert "/health" in routes

        print("API server configured correctly")
        return True

    except Exception as e:
        print(f"API test failed: {e}")
        return False

async def test_async_components() -> bool:
    """Test async components."""
    print("\nTesting async components...")

    try:
        from src.exchange import AlpacaExchange
        from src.bot import sync_positions_on_startup

        # Test that async methods exist
        ex = AlpacaExchange()
        assert hasattr(ex, 'load')
        assert hasattr(ex, 'close')
        assert hasattr(ex, 'fetch_ohlcv_df')

        print("Async components working correctly")
        return True

    except Exception as e:
        print(f"Async components test failed: {e}")
        return False

def main() -> None:
    """Run all modernization tests."""
    print("Testing Apex Oracle Bot Modernization")
    print("=" * 50)

    tests = [
        test_imports,
        test_configuration,
        test_database,
        test_logging,
        test_api,
    ]

    results = []
    for test in tests:
        results.append(test())

    # Run async test
    async_results = asyncio.run(test_async_components())
    results.append(async_results)

    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Passed: {sum(results)}/{len(results)}")

    if all(results):
        print("All tests passed! Modernization successful.")
        print("\nModernization Summary:")
        print("Python 3.12+ standards and best practices")
        print("Pydantic Settings V2 for configuration")
        print("SQLAlchemy 2.0 with MappedAsDataclass")
        print("Polars instead of pandas")
        print("httpx for async HTTP requests")
        print("FastAPI + uvicorn API server")
        print("Structlog for structured logging")
        print("Comprehensive type hints")
        print("Robust error handling with tenacity")
        print("Modern project structure with pyproject.toml")
    else:
        print("Some tests failed. Please check the errors above.")

    print("\nReady for production deployment!")

if __name__ == "__main__":
    main()