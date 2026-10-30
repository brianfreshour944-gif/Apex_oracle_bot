"""Regression tests for the shadow-arena duplicate-open race (Fix 1).

_evaluate_shadow_position() is a read-then-insert ("is there an open trade? no
-> insert one") with no atomicity. Candidates are evaluated from one thread per
symbol via asyncio.to_thread, so two evaluations of the same candidate/symbol
can both observe "no open trade" and both insert, opening duplicate shadow
positions. Those duplicates corrupt the candidate PnL the monthly cull ranks
on. Reproduced 2026-10-12 against a real SQLite DB: 6 simultaneous signals
produced up to 3 open rows.

Guards the two layers of the fix:
  - a unique partial index on shadow_trades(candidate_name, symbol) WHERE
    status='open' (the hard backstop), and
  - a per-(candidate, symbol) lock serializing the check-then-act.
"""
import threading
import uuid

import pytest
from sqlalchemy import text

import src.db as db
from src.config import settings


@pytest.fixture
def fresh_db(tmp_path):
    original = settings.DATABASE_URL
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    settings.DATABASE_URL = "sqlite:///" + str(tmp_path / "bot.db").replace("\\", "/")
    db.init_db()
    yield
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    settings.DATABASE_URL = original


def _open_count(candidate="Bot_A_Light", symbol="BTC/USD"):
    with db.get_db_session() as session:
        return len(
            session.execute(
                db.select(db.ShadowTrade).where(
                    db.ShadowTrade.candidate_name == candidate,
                    db.ShadowTrade.symbol == symbol,
                    db.ShadowTrade.status == "open",
                )
            ).scalars().all()
        )


def test_unique_open_index_exists(fresh_db):
    with db.get_engine().connect() as conn:
        rows = conn.execute(text("PRAGMA index_list('shadow_trades')")).fetchall()
    unique_names = [r[1] for r in rows if r[2] == 1]
    assert "ux_shadow_trades_open_candidate_symbol" in unique_names, \
        f"expected a UNIQUE index on shadow_trades open rows, got: {[r[1] for r in rows]}"


def test_repeated_buy_signals_open_exactly_one(fresh_db):
    from src.shadow_arena import _evaluate_shadow_position
    for _ in range(5):
        _evaluate_shadow_position("Bot_A_Light", "BTC/USD", 83682.98, 0.02, 0.04, new_action="buy", is_live=True)
    assert _open_count() == 1


def test_concurrent_buy_signals_open_exactly_one(fresh_db):
    from src.shadow_arena import _evaluate_shadow_position
    barrier = threading.Barrier(6)
    errors = []

    def worker():
        try:
            barrier.wait()
            _evaluate_shadow_position("Bot_A_Light", "BTC/USD", 83682.98, 0.02, 0.04, new_action="buy", is_live=True)
        except Exception as e:  # pragma: no cover - surfaced via assertion below
            errors.append(repr(e))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent shadow evaluation raised: {errors}"
    assert _open_count() == 1


def test_unique_index_rejects_second_open_row(fresh_db):
    """The DB-level backstop: a direct second open insert must be rejected."""
    from sqlalchemy.exc import IntegrityError
    with db.get_db_session() as session:
        session.add(db.ShadowTrade(
            trade_id=uuid.uuid4().hex, candidate_name="Bot_A_Light", symbol="BTC/USD",
            side="buy", qty=1.0, entry_price=100.0, status="open",
        ))
        session.commit()
    with pytest.raises(IntegrityError):
        with db.get_db_session() as session:
            session.add(db.ShadowTrade(
                trade_id=uuid.uuid4().hex, candidate_name="Bot_A_Light", symbol="BTC/USD",
                side="buy", qty=1.0, entry_price=101.0, status="open",
            ))
            session.commit()


def test_ensure_indexes_reconciles_pre_existing_duplicates(fresh_db):
    """A DB carrying duplicates from before the fix must still migrate: all
    but the newest open row are retired to history (nothing deleted)."""
    # Drop the unique index, plant two open rows (as the old code could), then
    # re-run _ensure_indexes() -- the upgrade path a deployed DB will take.
    with db.get_engine().connect() as conn:
        conn.execute(text("DROP INDEX IF EXISTS ux_shadow_trades_open_candidate_symbol"))
        conn.commit()
    with db.get_db_session() as session:
        session.add(db.ShadowTrade(
            trade_id="older", candidate_name="Bot_A_Light", symbol="BTC/USD",
            side="buy", qty=1.0, entry_price=100.0, status="open",
        ))
        session.add(db.ShadowTrade(
            trade_id="newer", candidate_name="Bot_A_Light", symbol="BTC/USD",
            side="buy", qty=1.0, entry_price=101.0, status="open",
        ))
        session.commit()

    db._ensure_indexes()

    with db.get_db_session() as session:
        open_ids = {
            r.trade_id for r in session.execute(
                db.select(db.ShadowTrade).where(db.ShadowTrade.status == "open")
            ).scalars().all()
        }
        total = len(session.execute(db.select(db.ShadowTrade)).scalars().all())
    assert open_ids == {"newer"}, f"newest open row should survive, got {open_ids}"
    assert total == 2, "no rows should be deleted during reconciliation"
