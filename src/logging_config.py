"""Modern logging configuration using Structlog with robust error handling."""

import json
import logging
import sys
import structlog
from typing import Any, Dict
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
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