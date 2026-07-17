"""Main entry point for the modern trading bot."""

import asyncio
from src.bot import run_trading_bot
from src.logging_config import get_logger

logger = get_logger("main")

def main() -> None:
    """Main entry point."""
    try:
        logger.info("Starting Apex Oracle Bot v2.0.0")
        asyncio.run(run_trading_bot())
    except KeyboardInterrupt:
        logger.info("Shutdown requested. Exiting gracefully.")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()