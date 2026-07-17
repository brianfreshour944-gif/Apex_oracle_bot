"""Modern logging configuration using Structlog."""

import logging
import sys
import structlog
from structlog.types import Processor

def configure_structlog() -> None:
    """Configure Structlog for structured JSON logging."""
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

    # Development processors (console output)
    dev_processors = shared_processors + [
        structlog.dev.ConsoleRenderer(colors=True)
    ]

    # Production processors (JSON output)
    prod_processors = shared_processors + [
        structlog.processors.JSONRenderer()
    ]

    # Configure structlog
    structlog.configure(
        processors=dev_processors if sys.stdout.isatty() else prod_processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory() if sys.stdout.isatty() else structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    # Configure standard logging to use structlog
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger with the given name."""
    return structlog.get_logger(name)

# Configure at import time
configure_structlog()

# Module-level logger
logger = get_logger(__name__)