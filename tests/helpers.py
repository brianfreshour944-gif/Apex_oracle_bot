"""Test helpers shared across the suite."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.paper_guard import PAPER_BASE_URL


def paper_trading_client() -> MagicMock:
    """A MagicMock trading client that DECLARES the paper endpoint.

    The fail-closed paper guard (``src/paper_guard.py``) inspects the client's
    ACTUAL resolved endpoint (``client._base_url``) before every order
    submission. A bare ``MagicMock`` has ``_base_url`` set to another mock, so
    the guard correctly classifies it as non-paper and refuses the order. Tests
    that exercise an order-submission path therefore must declare their venue
    explicitly; this helper does that so the test stays about the behaviour
    under test rather than about the guard.
    """
    client = MagicMock()
    client._base_url = PAPER_BASE_URL
    return client
