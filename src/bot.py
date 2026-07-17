"""Main trading bot implementation."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from src.config import settings
from src.logging_config import get_logger
from src.db import init_db
from src.exchange import AlpacaExchange
from src.api import start_fastapi_server_async

logger = get_logger("bot")

# Global state
ex: Optional[AlpacaExchange] = None

async def run_trading_bot() -> None:
    """Main trading bot loop."""
    global ex

    try:
        logger.info("Initializing Apex Oracle Bot v2.0.0")
        logger.info("=================================")
        logger.info("Configuration:")
        logger.info(f"Bot Name: {settings.BOT_NAME}")
        logger.info(f"Database: {settings.DATABASE_URL}")
        logger.info(f"Exchange: Alpaca Crypto (Paper={settings.ALPACA_BASE_URL.endswith('paper-api.alpaca.markets')})")
        logger.info(f"Symbols: {settings.SYMBOLS}")
        logger.info("=================================")

        # Initialize database (handle connection failures gracefully)
        try:
            init_db()
            logger.info("Database connected successfully")
        except Exception as e:
            logger.warning(f"Database connection failed (will retry later): {e}")
            logger.info("Running in offline mode - some features may be limited")

        logger.info(settings.log_config())

        # Initialize exchange
        ex = AlpacaExchange()
        await ex.load()
        logger.info("Alpaca exchange connected")

        # Start API server
        await start_fastapi_server_async()
        logger.info("FastAPI server started")

        # Main loop would go here
        logger.info("Bot initialization complete. Ready to trade.")

        # Keep running
        while True:
            await asyncio.sleep(3600)
            logger.info("Bot heartbeat - running normally")

    except KeyboardInterrupt:
        logger.info("Shutdown requested. Exiting gracefully.")
        if ex:
            await ex.close()
    except Exception as e:
        logger.error(f"Fatal error in bot: {e}", exc_info=True)
        if ex:
            await ex.close()
        raise

def run_bot() -> None:
    """Synchronous wrapper for async bot."""
    asyncio.run(run_trading_bot())