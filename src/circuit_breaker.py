  
"""Circuit breaker for exchange/network calls."""  
  
from __future__ import annotations  
  
import asyncio  
import time  
from enum import Enum  
from typing import Callable, TypeVar  
  
from src.config import settings  
from src.logging_config import get_logger  
  
logger = get_logger("circuit_breaker")  
  
T = TypeVar("T")  
  
class CircuitState(str, Enum):  
    CLOSED = "closed"  
    OPEN = "open"  
    HALF_OPEN = "half_open"  
  
class CircuitBreaker:  
    def __init__(self, name, failure_threshold=None, open_seconds=None):  
        self.name = name  
        self.failure_threshold = failure_threshold or settings.CIRCUIT_FAILURE_THRESHOLD  
        self.open_seconds = open_seconds or settings.CIRCUIT_OPEN_SECONDS  
        self._failures = 0  
        self._state = CircuitState.CLOSED  
        self._opened_at = 0.0  
        self._lock = asyncio.Lock()  
  
    @property  
    def state(self):  
        return self._state  
  
    def _maybe_transition(self):  
        if self._state == CircuitState.OPEN and (time.monotonic() - self._opened_at) >= self.open_seconds:  
            self._state = CircuitState.HALF_OPEN  
            logger.warning(f"Circuit {self.name!r} -^> HALF_OPEN")  
  
    async def call(self, func, *args, **kwargs):  
        async with self._lock:  
            self._maybe_transition()  
            if self._state == CircuitState.OPEN:  
                raise RuntimeError(f"Circuit {self.name!r} is OPEN - calls blocked")  
            probe = self._state == CircuitState.HALF_OPEN  
        try:  
            result = await func(*args, **kwargs)  
        except Exception:  
            async with self._lock:  
                self._failures += 1  
                if self._state != CircuitState.OPEN and self._failures >= self.failure_threshold:  
                    self._state = CircuitState.OPEN  
                    self._opened_at = time.monotonic()  
                    logger.critical(f"Circuit {self.name!r} TRIPPED OPEN")  
            raise  
        async with self._lock:  
            if probe:  
                self._state = CircuitState.CLOSED  
                self._failures = 0  
                logger.info(f"Circuit {self.name!r} recovered -^> CLOSED")  
            elif self._state == CircuitState.CLOSED:  
                self._failures = 0  
        return result  
  
    def reset(self):  
        self._failures = 0  
        self._state = CircuitState.CLOSED  
        self._opened_at = 0.0  
