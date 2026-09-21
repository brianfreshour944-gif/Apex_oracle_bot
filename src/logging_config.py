"""Modern logging configuration using Structlog with robust error handling."""

import json
import logging
import sys
import uuid
from datetime import datetime, UTC

import structlog


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            for key, value in record.extra_fields.items():
                log_data[key] = value
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


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

        # JSON formatter for production, console for development
        import os
        if os.getenv("ENVIRONMENT", "development") == "production":
            processors = shared_processors + [
                structlog.processors.JSONRenderer()
            ]
        else:
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
            level=logging.INFO,
            format="%(message)s",
            stream=sys.stdout,
            force=True,
        )
        
        # Silence noisy third-party libraries
        for noisy_logger in ["httpx", "httpcore", "websockets", "urllib3", "apscheduler", "alpaca", "torch", "matplotlib"]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)
        
        print("✅ Structlog configured successfully (info mode)", file=sys.stderr)
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


def set_correlation_id(correlation_id: str | None = None) -> str:
    """Bind a correlation ID into structlog's contextvars so every log line
    emitted for the rest of this run/request carries it automatically (via
    the `merge_contextvars` processor already in configure_structlog's
    pipeline above) without needing to pass it to every logger.info() call.

    Generates a random UUID4 if none is given. Returns the ID that was bound.

    This was previously imported by scripts/retrain_transformer.py but never
    defined anywhere in this module -- that import has been raising
    ImportError at module load since it was added, meaning the script could
    never actually run. Fixed 2026-09-20 while wiring that script's
    multi-symbol walk-forward validation (ADVERSARIAL_AUDIT_2026-09-20.md §8/§9).
    """
    cid = correlation_id or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    return cid