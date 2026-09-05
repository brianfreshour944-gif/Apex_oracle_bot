"""Circuit breaker for exchange/network calls."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any, TypeVar

from src.config import settings
from src.logging_config import get_logger

logger = get_logger("circuit_breaker")

T = TypeVar("T")


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int | None = None, open_seconds: float | None = None):
        self.name = name
        self.failure_threshold = failure_threshold or getattr(settings, "CIRCUIT_FAILURE_THRESHOLD", 5)
        self.open_seconds = open_seconds or getattr(settings, "CIRCUIT_OPEN_SECONDS", 60.0)

        self._failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at = 0.0
        self._lock = asyncio.Lock()
        self._half_open_successes = 0

    @property
    def state(self) -> CircuitState:
        return self._state

    def _maybe_transition(self) -> None:
        if self._state == CircuitState.OPEN and (time.monotonic() - self._opened_at) >= (self.open_seconds or 0.0):
            self._state = CircuitState.HALF_OPEN
            self._half_open_successes = 0
            logger.warning(f"Circuit {self.name!r} -> HALF_OPEN")

    async def call(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        async with self._lock:
            self._maybe_transition()
            if self._state == CircuitState.OPEN:
                raise RuntimeError(f"Circuit {self.name!r} is OPEN - calls blocked")
            probe = self._state == CircuitState.HALF_OPEN
        try:
            result = await func(*args, **kwargs)
        except Exception:
            async with self._lock:
                if probe:
                    # Immediate reopening on HALF_OPEN probe failure
                    self._state = CircuitState.OPEN
                    self._opened_at = time.monotonic()
                    self._half_open_successes = 0
                    logger.critical(f"Circuit {self.name!r} failed during HALF_OPEN probe -> OPEN")
                else:
                    # Existing failure_threshold-based logic for CLOSED state
                    self._failures += 1
                    if self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN) and self._failures >= (self.failure_threshold or 0):
                        self._state = CircuitState.OPEN
                        self._opened_at = time.monotonic()
                        logger.critical(f"Circuit {self.name!r} TRIPPED OPEN")
            raise
        async with self._lock:
            if probe:
                if self._state == CircuitState.HALF_OPEN:
                    # Success in HALF_OPEN - need consecutive successes to close
                    self._half_open_successes += 1
                    if self._half_open_successes >= 2:
                        self._state = CircuitState.CLOSED
                        self._failures = 0
                        self._half_open_successes = 0
                        logger.info(f"Circuit {self.name!r} recovered -> CLOSED")
            # In CLOSED state, reset failures on success
            elif self._state == CircuitState.CLOSED:
                self._failures = 0
            return result

    def reset(self) -> None:
        self._failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at = 0.0
        self._half_open_successes = 0