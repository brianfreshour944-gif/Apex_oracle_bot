"""Unit tests for AlpacaExchange rate limiting and order handling."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.exchange import RateLimiter, AlpacaExchange


@pytest.mark.asyncio
async def test_rate_limiter_acquires_tokens():
    """Verify rate limiter allows rapid acquiring under limit."""
    limiter = RateLimiter(max_rate=100.0, time_period=60.0)
    # Acquire 5 tokens quickly
    for _ in range(5):
        await limiter.acquire()
    assert limiter.tokens >= 0.0


@pytest.mark.asyncio
async def test_create_order_confirmation_mock():
    """Verify create_order polling logic with mock client."""
    ex = AlpacaExchange()
    ex.client = AsyncMock()
    ex.rate_limiter = AsyncMock()

    # Mock POST /v2/orders response
    post_resp = MagicMock()
    post_resp.json.return_value = {"id": "ord_123", "status": "pending_new"}
    post_resp.raise_for_status = MagicMock()

    # Mock GET /v2/orders/ord_123 response
    get_resp = MagicMock()
    get_resp.json.return_value = {"id": "ord_123", "status": "filled"}
    get_resp.raise_for_status = MagicMock()

    ex.client.post.return_value = post_resp
    ex.client.get.return_value = get_resp

    res = await ex.create_order("BTC/USD", 1.0, "buy", confirm=True, confirm_timeout=2.0)
    assert res["status"] == "filled"
    assert res["id"] == "ord_123"
