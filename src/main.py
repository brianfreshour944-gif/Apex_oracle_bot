"""Main entry point for the Apex Oracle Bot with robust error handling."""

import asyncio
import sys
import traceback

from src.bot import run_trading_bot
from src.config import redact_database_url, settings
from src.logging_config import get_logger

logger = get_logger("main")

def main() -> None:
    """Main entry point for the trading bot."""
    try:
        logger.info("Starting Apex Oracle Bot v2.0.0")
        logger.info("=================================")
        logger.info("Configuration:")
        logger.info(f"Database: {redact_database_url(settings.DATABASE_URL)}")
        logger.info("Exchange: Alpaca Crypto")
        logger.info("=================================")

        # Run the trading bot
        asyncio.run(run_trading_bot())

    except KeyboardInterrupt:
        logger.info("Shutdown requested. Exiting gracefully.")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        print("\n=== FULL TRACEBACK ===", file=sys.stderr)
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n💥 UNCAUGHT STARTUP CRASH: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)