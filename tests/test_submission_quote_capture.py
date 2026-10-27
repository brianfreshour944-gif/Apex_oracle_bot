"""Submission-quote instrumentation (2026-10-25, W1).

Measurement-only change: snapshot the NBBO (bid/ask/mid/spread + quote
timestamp) immediately BEFORE each order submission and persist it on the
order row, so execution cost (spread, and slippage vs a *contemporaneous*
reference) becomes measurable. Nothing here changes order behaviour.

Invariants pinned:
  - ``AlpacaExchange.get_latest_quote`` returns bid/ask/mid/spread/quote_ts.
  - It NEVER raises: a quote outage returns None (order path unaffected).
  - A missing/partial quote persists as NULL, never a zero spread/cost.
  - ``bot._persist_order_record`` carries the quote from the adapter result.
  - The additive migration adds the three columns to a legacy table, is
    idempotent, and leaves legacy rows NULL.

No live endpoint is contacted; a fake data client is injected.
"""

import datetime
import threading
import time
import types

import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import select, text

import src.bot as bot
import src.db as db
from src import exchange as _exchange_mod
from src.config import settings
from src.exchange import AlpacaExchange

# ── fakes ──────────────────────────────────────────────────────────────────

class _FakeQuote:
    def __init__(self, bid, ask, ts):
        self.bid_price = bid
        self.ask_price = ask
        self.timestamp = ts


class _FakeDataClient:
    """Stands in for CryptoHistoricalDataClient; no network."""

    def __init__(self, quotes=None, raise_exc=None):
        self._quotes = quotes if quotes is not None else {}
        self._raise = raise_exc

    def get_crypto_latest_quote(self, request):
        if self._raise is not None:
            raise self._raise
        return self._quotes


def _exchange_with(data_client):
    ex = AlpacaExchange.__new__(AlpacaExchange)  # bypass __init__ (no creds)
    ex.data_client = data_client
    return ex


# ── 1-2: get_latest_quote returns the derived fields ───────────────────────

@pytest.mark.asyncio
async def test_get_latest_quote_returns_bid_ask_mid_spread():
    ts = datetime.datetime(2026, 10, 24, 12, 0, tzinfo=datetime.UTC)
    ex = _exchange_with(_FakeDataClient({"BTC/USD": _FakeQuote(100.0, 102.0, ts)}))
    q = await ex.get_latest_quote("BTC/USD")
    assert q["bid"] == 100.0
    assert q["ask"] == 102.0
    assert q["mid"] == 101.0
    assert q["spread"] == 2.0
    assert q["quote_ts"] == ts.isoformat()


@pytest.mark.asyncio
async def test_get_latest_quote_never_raises_on_error():
    ex = _exchange_with(_FakeDataClient(raise_exc=RuntimeError("boom")))
    assert await ex.get_latest_quote("BTC/USD") is None


@pytest.mark.asyncio
async def test_get_latest_quote_missing_symbol_is_none():
    ex = _exchange_with(_FakeDataClient({}))
    assert await ex.get_latest_quote("BTC/USD") is None


@pytest.mark.asyncio
async def test_get_latest_quote_rejects_nonpositive_prices():
    ts = datetime.datetime(2026, 10, 24, 12, 0, tzinfo=datetime.UTC)
    ex = _exchange_with(_FakeDataClient({"BTC/USD": _FakeQuote(0.0, 0.0, ts)}))
    assert await ex.get_latest_quote("BTC/USD") is None


# ── 3-4: persistence funnel carries the quote ──────────────────────────────

@pytest.fixture
def fresh_db(tmp_path):
    original = settings.DATABASE_URL
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    db._open_snapshot_cache.clear()
    settings.DATABASE_URL = "sqlite:///" + str(tmp_path / "bot.db").replace("\\", "/")
    db.init_db()
    yield
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    db._open_snapshot_cache.clear()
    settings.DATABASE_URL = original


def _read_order(order_id: str):
    from src.db import OrderRecord, get_db_session
    with get_db_session() as session:
        return session.execute(
            select(OrderRecord).where(OrderRecord.order_id == order_id)
        ).scalar_one()


def test_persist_carries_submission_quote(fresh_db):
    ts = "2026-10-24T12:00:00+00:00"
    bot._persist_order_record(
        {"id": "q-1", "qty": 1.0, "filled_qty": 1.0, "filled_avg_price": 100.0,
         "commission": 0.0, "status": "filled",
         "submission_quote": {"bid": 99.0, "ask": 101.0, "mid": 100.0,
                              "spread": 2.0, "quote_ts": ts}},
        "BTC/USD", "buy", "cid-q1",
    )
    row = _read_order("q-1")
    assert row.mid_at_submit == pytest.approx(100.0)
    assert row.spread_at_submit == pytest.approx(2.0)
    assert row.quote_ts is not None


def test_missing_quote_persists_null_not_zero(fresh_db):
    bot._persist_order_record(
        {"id": "q-2", "qty": 1.0, "filled_qty": 1.0, "filled_avg_price": 100.0,
         "commission": 0.0, "status": "filled"},  # no submission_quote key
        "BTC/USD", "buy", "cid-q2",
    )
    row = _read_order("q-2")
    assert row.mid_at_submit is None
    assert row.spread_at_submit is None
    assert row.quote_ts is None


def test_none_quote_persists_null(fresh_db):
    bot._persist_order_record(
        {"id": "q-3", "qty": 1.0, "filled_qty": 1.0, "filled_avg_price": 100.0,
         "commission": 0.0, "status": "filled", "submission_quote": None},
        "BTC/USD", "buy", "cid-q3",
    )
    row = _read_order("q-3")
    assert row.mid_at_submit is None and row.spread_at_submit is None


def test_partial_quote_persists_only_present_fields(fresh_db):
    bot._persist_order_record(
        {"id": "q-4", "qty": 1.0, "filled_qty": 1.0, "filled_avg_price": 100.0,
         "commission": 0.0, "status": "filled",
         "submission_quote": {"mid": 100.0}},  # no spread, no ts
        "BTC/USD", "buy", "cid-q4",
    )
    row = _read_order("q-4")
    assert row.mid_at_submit == pytest.approx(100.0)
    assert row.spread_at_submit is None
    assert row.quote_ts is None


def test_save_order_record_accepts_quote_kwargs(fresh_db):
    db.save_order_record(
        order_id="q-5", symbol="ETH/USD", side="sell", qty=1.0,
        filled_qty=1.0, filled_avg_price=2500.0, commission=0.0,
        mid_at_submit=2500.0, spread_at_submit=1.25,
        quote_ts=datetime.datetime(2026, 10, 24, 12, 0, tzinfo=datetime.UTC),
        status="filled",
    )
    row = _read_order("q-5")
    assert row.mid_at_submit == pytest.approx(2500.0)
    assert row.spread_at_submit == pytest.approx(1.25)
    assert row.quote_ts is not None


# ── 5: additive migration on a legacy table, idempotent ────────────────────

def test_migration_adds_quote_columns_to_legacy_orders(tmp_path):
    original = settings.DATABASE_URL
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    settings.DATABASE_URL = "sqlite:///" + str(tmp_path / "legacy.db").replace("\\", "/")
    try:
        engine = db.get_engine()
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE orders (order_id VARCHAR(64) PRIMARY KEY, "
                "decision_id VARCHAR(64), symbol VARCHAR(32), side VARCHAR(16), "
                "qty FLOAT, filled_qty FLOAT, filled_avg_price FLOAT, "
                "commission FLOAT, status VARCHAR(32), type VARCHAR(16), "
                "time_in_force VARCHAR(16), client_order_id VARCHAR(64), "
                "submitted_at DATETIME, filled_at DATETIME)"
            ))
            conn.execute(text(
                "INSERT INTO orders (order_id, symbol, side, commission, status) "
                "VALUES ('legacy-q', 'BTC/USD', 'buy', 0.0, 'filled')"
            ))
        before = {c["name"] for c in sa_inspect(engine).get_columns("orders")}
        assert not {"mid_at_submit", "spread_at_submit", "quote_ts"} & before

        db._ensure_additive_columns()
        cols = {c["name"] for c in sa_inspect(engine).get_columns("orders")}
        assert {"mid_at_submit", "spread_at_submit", "quote_ts"} <= cols
        # Legacy row stays NULL -- never a fabricated zero quote.
        with engine.begin() as conn:
            r = conn.execute(text(
                "SELECT mid_at_submit, spread_at_submit, quote_ts "
                "FROM orders WHERE order_id='legacy-q'"
            )).one()
            assert r == (None, None, None)

        # Idempotent: a second run must not raise or duplicate.
        db._ensure_additive_columns()
        names = [c["name"] for c in sa_inspect(engine).get_columns("orders")]
        for col in ("mid_at_submit", "spread_at_submit", "quote_ts"):
            assert names.count(col) == 1
    finally:
        if db._engine is not None:
            db._engine.dispose()
        db._engine = None
        db._tables_ensured = False
        settings.DATABASE_URL = original


# ── 6: capture is gated by the flag and never blocks the order ─────────────

def test_capture_flag_defaults_off():
    assert settings.CAPTURE_SUBMISSION_QUOTE is False


@pytest.mark.asyncio
async def test_capture_disabled_skips_quote(monkeypatch):
    """With the flag off, create_order must not call get_latest_quote."""
    calls = []
    ex = AlpacaExchange.__new__(AlpacaExchange)
    ex.trading_client = types.SimpleNamespace()
    # circuit_breaker.call(fn, *args) must be a passthrough in the fake.
    ex.circuit_breaker = types.SimpleNamespace(
        call=lambda fn, *a, **kw: fn(*a, **kw)
    )
    ex._order_cache = {}
    ex._recent_order_submissions = {}
    ex._duplicate_order_window_sec = 10.0

    async def _fake_quote(symbol):
        calls.append(symbol)
        return {"bid": 1.0, "ask": 2.0, "mid": 1.5, "spread": 1.0, "quote_ts": None}

    monkeypatch.setattr(ex, "get_latest_quote", _fake_quote)
    monkeypatch.setattr(settings, "CAPTURE_SUBMISSION_QUOTE", False)

    class _FakeOrder:
        id = "x1"
        symbol = "BTC/USD"
        qty = 1.0
        status = types.SimpleNamespace(value="new")

    ex.trading_client.submit_order = lambda req: _FakeOrder()
    res = await ex.create_order("BTC/USD", qty=1.0, side="buy", confirm=False)
    assert calls == []
    assert res.get("submission_quote") is None


@pytest.mark.asyncio
async def test_capture_enabled_attaches_quote(monkeypatch):
    """With the flag on, create_order attaches the captured quote to its result."""
    ex = AlpacaExchange.__new__(AlpacaExchange)
    ex.trading_client = types.SimpleNamespace()
    ex.circuit_breaker = types.SimpleNamespace(call=lambda fn, *a, **kw: fn(*a, **kw))
    ex._order_cache = {}
    ex._recent_order_submissions = {}
    ex._duplicate_order_window_sec = 10.0

    async def _fake_quote(symbol):
        return {"bid": 99.0, "ask": 101.0, "mid": 100.0, "spread": 2.0,
                "quote_ts": "2026-10-24T12:00:00+00:00"}

    monkeypatch.setattr(ex, "get_latest_quote", _fake_quote)
    monkeypatch.setattr(settings, "CAPTURE_SUBMISSION_QUOTE", True)

    class _FakeOrder:
        id = "x2"
        symbol = "BTC/USD"
        qty = 1.0
        status = types.SimpleNamespace(value="new")

    ex.trading_client.submit_order = lambda req: _FakeOrder()
    res = await ex.create_order("BTC/USD", qty=1.0, side="buy", confirm=False)
    assert res["submission_quote"]["mid"] == 100.0
    assert res["submission_quote"]["spread"] == 2.0


@pytest.mark.asyncio
async def test_quote_failure_does_not_block_order(monkeypatch):
    """A raising quote helper must not fail the order; quote is simply absent."""
    ex = AlpacaExchange.__new__(AlpacaExchange)
    ex.trading_client = types.SimpleNamespace()
    ex.circuit_breaker = types.SimpleNamespace(call=lambda fn, *a, **kw: fn(*a, **kw))
    ex._order_cache = {}
    ex._recent_order_submissions = {}
    ex._duplicate_order_window_sec = 10.0

    async def _boom(symbol):
        raise RuntimeError("quote backend down")

    monkeypatch.setattr(ex, "get_latest_quote", _boom)
    monkeypatch.setattr(settings, "CAPTURE_SUBMISSION_QUOTE", True)

    class _FakeOrder:
        id = "x3"
        symbol = "BTC/USD"
        qty = 1.0
        status = types.SimpleNamespace(value="new")

    ex.trading_client.submit_order = lambda req: _FakeOrder()
    res = await ex.create_order("BTC/USD", qty=1.0, side="buy", confirm=False)
    assert res["id"] == "x3"                 # order still succeeded
    assert res["submission_quote"] is None    # quote absent, not fabricated


# ── 7: executor isolation (2026-10-25) ─────────────────────────────────────
#
# Quote capture runs on its own small pool, NOT the loop's default executor,
# which submit_order also uses. These tests pin that the order path cannot be
# starved by a hung quote, that the caller is released by the wait_for bound,
# that the pool is released on close(), and that the capture genuinely runs off
# the default executor.


def _bare_exchange():
    """An exchange with the attributes create_order touches, no creds/network."""
    ex = AlpacaExchange.__new__(AlpacaExchange)
    ex.trading_client = types.SimpleNamespace()
    ex.circuit_breaker = types.SimpleNamespace(call=lambda fn, *a, **kw: fn(*a, **kw))
    ex._order_cache = {}
    ex._recent_order_submissions = {}
    ex._duplicate_order_window_sec = 10.0
    ex._quote_executor = None
    return ex


@pytest.mark.asyncio
async def test_hung_quote_does_not_starve_submit_order(monkeypatch):
    """T1: with the quote backend hung, submit_order must still run promptly.

    This is the assertion the isolation design exists for: on the shared default
    pool a hung capture delayed submit_order by ~1.9s; on the dedicated pool it
    must not wait for the capture at all.
    """
    ex = _bare_exchange()
    release = threading.Event()

    class _HungDataClient:
        def get_crypto_latest_quote(self, request):
            release.wait(5.0)          # simulate a hung socket read
            return {}

    ex.data_client = _HungDataClient()

    class _FakeOrder:
        id = "iso1"
        symbol = "BTC/USD"
        qty = 1.0
        status = types.SimpleNamespace(value="new")

    ex.trading_client.submit_order = lambda req: _FakeOrder()
    monkeypatch.setattr(settings, "CAPTURE_SUBMISSION_QUOTE", True)
    # Bound the capture wait tightly so the test is fast; the point is that the
    # order path does not wait on the hung capture thread.
    monkeypatch.setattr(_exchange_mod, "_QUOTE_CAPTURE_TIMEOUT_SEC", 0.3)

    t0 = time.monotonic()
    res = await ex.create_order("BTC/USD", qty=1.0, side="buy", confirm=False)
    elapsed = time.monotonic() - t0

    release.set()                      # let the hung thread finish
    assert res["id"] == "iso1"         # the order went through
    assert res["submission_quote"] is None  # hung capture -> no quote, not a guess
    assert elapsed < 1.5, f"submit waited {elapsed:.2f}s behind a hung quote"
    if ex._quote_executor is not None:
        ex._quote_executor.shutdown(wait=False)


@pytest.mark.asyncio
async def test_wait_for_releases_caller_on_slow_quote(monkeypatch):
    """T2: a quote slower than the bound is abandoned and the order proceeds."""
    ex = _bare_exchange()

    class _SlowDataClient:
        def get_crypto_latest_quote(self, request):
            time.sleep(2.0)
            return {}

    ex.data_client = _SlowDataClient()

    class _FakeOrder:
        id = "iso2"
        symbol = "BTC/USD"
        qty = 1.0
        status = types.SimpleNamespace(value="new")

    ex.trading_client.submit_order = lambda req: _FakeOrder()
    monkeypatch.setattr(settings, "CAPTURE_SUBMISSION_QUOTE", True)
    monkeypatch.setattr(_exchange_mod, "_QUOTE_CAPTURE_TIMEOUT_SEC", 0.2)

    q = await ex.get_latest_quote("BTC/USD")
    assert q is None                   # timed out -> None, never raises

    res = await ex.create_order("BTC/USD", qty=1.0, side="buy", confirm=False)
    assert res["id"] == "iso2"
    assert res["submission_quote"] is None
    if ex._quote_executor is not None:
        ex._quote_executor.shutdown(wait=False)


@pytest.mark.asyncio
async def test_close_shuts_down_quote_executor():
    """T3: close() must release the dedicated pool (no leaked threads)."""
    ex = _bare_exchange()
    ex.trading_client = types.SimpleNamespace()
    ex.data_client = types.SimpleNamespace()

    pool = ex._get_quote_executor()
    assert pool is not None

    await ex.close()
    assert ex._quote_executor is None
    # A shut-down pool rejects new work rather than silently queueing it.
    with pytest.raises(RuntimeError):
        pool.submit(lambda: None)


@pytest.mark.asyncio
async def test_capture_runs_off_default_executor():
    """T4: the capture must run on the dedicated pool, not the default executor."""
    ex = _bare_exchange()
    seen = {}

    class _NameDataClient:
        def get_crypto_latest_quote(self, request):
            seen["thread"] = threading.current_thread().name
            return {}

    ex.data_client = _NameDataClient()
    await ex.get_latest_quote("BTC/USD")

    assert "quote-capture" in seen.get("thread", "")
    ex._quote_executor.shutdown(wait=False)
