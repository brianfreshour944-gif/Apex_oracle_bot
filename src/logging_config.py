"""Modern logging configuration using Structlog with robust error handling."""

import logging
import sys
import structlog

def configure_structlog() -> None:
    """Configure Structlog with robust fallback for early crashes."""
    try:
        # Shared processors for all loggers
        shared_processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]

        # Always use console renderer to catch early errors
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Configure standard logging
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            stream=sys.stdout,
            force=True,
        )
        print("✅ Structlog configured successfully (debug mode)", file=sys.stderr)
    except Exception as e:
        print(f"⚠️  Logging setup failed: {e}. Using basic fallback.", file=sys.stderr)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s | %(levelname)s | %(message)s",
            force=True,
        )

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger with the given name."""
    return structlog.get_logger(name)

# Configure at import time
configure_structlog()

# Module-level logger
logger = get_logger(__name__)