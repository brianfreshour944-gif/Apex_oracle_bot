"""Modern logging configuration using Structlog with robust error handling."""

import json
import logging
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler

import structlog

# ── Persistent rotating file logging ────────────────────────────────────────
# The bot otherwise logs ONLY to stdout, which Docker's json-file driver keeps
# but Coolify's log UI truncates and no host-side grep can reach without
# `docker logs`. A rotating file under the persistent data volume survives
# restarts, is greppable (TRAINING_JOB / LIVE_BUFFER / RECONCILE markers), and
# is bounded so it can never fill the volume. Path is derived from this file
# (src/logging_config.py -> <repo>/data/logs) so it lands inside the
# bot_data:/app/data volume in the container; APEX_LOG_DIR overrides it (the
# test suite points it at a temp dir).
LOG_DIR = os.environ.get("APEX_LOG_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "logs"
)
LOG_FILE = os.path.join(LOG_DIR, "apex_bot.log")
LOG_MAX_BYTES = 20 * 1024 * 1024  # 20 MB per file
LOG_BACKUP_COUNT = 5              # 20 MB x 5 = 100 MB cap

_file_logging_configured = False
_structlog_configured = False
_training_job_started: dict[str, float] = {}


def configure_file_logging(
    log_dir: str | None = None,
    *,
    max_bytes: int = LOG_MAX_BYTES,
    backup_count: int = LOG_BACKUP_COUNT,
) -> str | None:
    """Attach a bounded RotatingFileHandler to the root logger.

    Idempotent -- safe to call from every entrypoint; only the first call
    installs a handler. Returns the log file path, or None if the directory
    could not be created (fail-open: file logging is a convenience, it must
    never stop the bot from starting).
    """
    global _file_logging_configured
    if _file_logging_configured:
        return LOG_FILE
    path = os.path.join(log_dir or LOG_DIR, os.path.basename(LOG_FILE))
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        handler.setFormatter(JSONFormatter())
        handler.setLevel(logging.INFO)
        root = logging.getLogger()
        root.addHandler(handler)
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
        _file_logging_configured = True
        return path
    except Exception as e:  # pragma: no cover - defensive
        print(f"⚠️  File logging setup failed ({e}); continuing with stdout only.", file=sys.stderr)
        return None


def log_training_job_start(job: str, script: str | None = None) -> None:
    """Mark the start of a scheduled training/retrain job.

    One grep-able line per run so an operator can see from the log file whether
    every scheduled job (analyzer, automl, cull, research, transformer replay,
    PPO, decision-transformer, PBT, OOD, post-mortem) actually fired and when.
    Records the start time so the matching log_training_job_result() can report
    a duration.
    """
    _training_job_started[job] = time.monotonic()
    suffix = f" script={os.path.basename(script)}" if script else ""
    logging.getLogger("training_job").info(
        f"TRAINING_JOB START job={job}{suffix} at={datetime.now(UTC).isoformat()}"
    )


def log_training_job_result(
    job: str,
    returncode: int | None = None,
    duration_sec: float | None = None,
    detail: str = "",
    script: str | None = None,
) -> None:
    """Mark the end of a scheduled training/retrain job with its outcome."""
    if duration_sec is None:
        t0 = _training_job_started.pop(job, None)
        if t0 is not None:
            duration_sec = time.monotonic() - t0
    suffix = f" script={os.path.basename(script)}" if script else ""
    dur = f" duration={duration_sec:.1f}s" if duration_sec is not None else ""
    rc = f" returncode={returncode}" if returncode is not None else ""
    extra = f" {detail}" if detail else ""
    logging.getLogger("training_job").info(
        f"TRAINING_JOB RESULT job={job}{suffix}{rc}{dur}{extra}"
    )


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
    """Configure Structlog with robust fallback for early crashes.

    Idempotent: safe to call from every entrypoint. It also installs the
    persistent rotating file handler, so any process that calls this gets
    both structured stdout and the on-disk log.
    """
    global _structlog_configured
    if _structlog_configured:
        return
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
            processors = [
                *shared_processors,
                structlog.processors.JSONRenderer()
            ]
        else:
            processors = [
                *shared_processors,
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

        _structlog_configured = True
        # Attach the persistent rotating file handler (after basicConfig, so
        # basicConfig(force=True) cannot wipe it) and announce where it is.
        log_path = configure_file_logging()
        if log_path:
            logging.getLogger("logging_config").info(f"File logging active: {log_path} (20 MB x 5)")
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