"""Tests for circuit breaker functionality."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    """Tests for CircuitBreaker functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        return CircuitBreaker("test_circuit", failure_threshold=3, open_seconds=1)

    @pytest.mark.asyncio
    async def test_circuit_closed_initially(self, circuit_breaker):
        """Circuit breaker starts in CLOSED state."""
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self, circuit_breaker):
        """Circuit opens after threshold failures."""
        async def failing_func():
            raise Exception("Simulated failure")

        # Fail up to threshold
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_blocks_calls_when_open(self, circuit_breaker):
        """Open circuit blocks subsequent calls."""
        circuit_breaker._state = CircuitState.OPEN
        circuit_breaker._opened_at = time.monotonic() + 1000  # Far future to prevent HALF_OPEN transition

        async def test_func():
            return "success"

        with pytest.raises(RuntimeError, match="Circuit.*is OPEN"):
            await circuit_breaker.call(test_func)

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self, circuit_breaker):
        """Circuit transitions to HALF_OPEN after timeout."""
        circuit_breaker._state = CircuitState.OPEN
        circuit_breaker._opened_at = 0  # Past timeout

        async def success_func():
            return "success"

        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_closes_after_success_in_half_open(self, circuit_breaker):
        """Circuit closes after 2 consecutive successes in HALF_OPEN."""
        circuit_breaker._state = CircuitState.HALF_OPEN

        async def success_func():
            return "success"

        # First success - should stay in HALF_OPEN
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.HALF_OPEN

        # Second success - should close
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED

@pytest.mark.asyncio
    async def test_reopens_on_failure_in_half_open(self, circuit_breaker):
        """Circuit reopens on failure in HALF_OPEN state."""
        circuit_breaker._state = CircuitState.HALF_OPEN

        async def failing_func():
            raise Exception("fail")

        with pytest.raises(Exception):
            await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_success_resets_failures_in_closed(self, circuit_breaker):
        """Successful calls reset failure count in CLOSED state."""
        async def success_func():
            return "success"

        result = await circuit_breaker.call(success_func)
        assert result == "success"

        # Fail twice (below threshold)
        for _ in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("fail")))

        # Success should reset failure count
        result = await circuit_breaker.call(lambda: "success")
        assert result == "success"


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with exchange."""

    @pytest.mark.asyncio
    async def test_exchange_circuit_breaker_opens_on_failures(self):
        """Test that exchange circuit breaker opens after failures."""
        from src.exchange import AlpacaExchange
        from src.circuit_breaker import CircuitBreaker

        with patch.object(AlpacaExchange, '__init__', lambda self: None):
            exchange = AlpacaExchange()
            exchange.circuit_breaker = CircuitBreaker("test_exchange", failure_threshold=2, open_seconds=1)

            async def failing_call():
                raise Exception("API error")

            # Fail twice to trip circuit
            for _ in range(2):
                with pytest.raises(Exception):
                    await exchange.circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("fail")))

            assert exchange.circuit_breaker.state == CircuitState.OPEN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])