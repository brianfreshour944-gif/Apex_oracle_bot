"""Tests for the adaptive-learner startup bootstrap (rebuild from DB history).

The per-regime sample gates (e.g. 'Insufficient regime samples: 1 < 30') only
grow one sample per closed trade after process start. The bootstrap replays
the DB's closed decision snapshots through the learner at startup so lifetime
closed-trade history counts toward the gates. These tests pin the contract:
correct per-regime counts, idempotency across restarts, tolerance of legacy
rows (no return_pct recorded, no brain votes), and the empty-history fail-safe
that must NOT reset an already-learned state file.
"""

import os

import pytest

from src import db
from src.bot import _bootstrap_adaptive_learner_from_history
from src.committee.committee import get_meta_learner
from src.config import settings


@pytest.fixture
def fresh_db(tmp_path):
    """Isolate DB + adaptive learner state, mirroring test_reconciliation's
    fixture, plus a reset of the learner singleton and its state file."""
    import src.committee.committee as committee_mod

    original_url = settings.DATABASE_URL
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    db._open_snapshot_cache.clear()
    settings.DATABASE_URL = "sqlite:///" + str(tmp_path / "bot.db").replace("\\", "/")
    committee_mod._META_LEARNER = None
    if os.path.exists(settings.ADAPTIVE_STATE_PATH):
        os.remove(settings.ADAPTIVE_STATE_PATH)
    db.init_db()
    yield
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    db._open_snapshot_cache.clear()
    settings.DATABASE_URL = original_url
    committee_mod._META_LEARNER = None


def _make_closed_trade(
    decision_id: str,
    symbol: str,
    regime: str,
    action: str,
    votes: dict,
    entry_price: float,
    qty: float,
    pnl: float,
    return_pct: float = 0.0,
) -> None:
    """Persist one closed decision snapshot, as a live round trip would."""
    assert db.save_decision_snapshot(
        decision_id=decision_id,
        symbol=symbol,
        regime=regime,
        final_action=action,
        confidence=0.6,
        size_multiplier=1.0,
        entry_price=entry_price,
        qty=qty,
        brain_votes=votes,
    )
    assert db.close_decision_snapshot(
        decision_id, realized_pnl=pnl, return_pct=return_pct, exit_reason="test"
    )


def test_bootstrap_counts_lifetime_closed_trades_per_regime(fresh_db):
    _make_closed_trade("d1", "BTC/USD", "trending", "buy",
                       {"momentum": "buy", "quant": "buy"}, 100.0, 1.0, 10.0, return_pct=10.0)
    _make_closed_trade("d2", "ETH/USD", "trending", "buy",
                       {"momentum": "buy", "quant": "buy"}, 50.0, 1.0, -5.0, return_pct=-10.0)
    _make_closed_trade("d3", "SOL/USD", "sideways", "buy",
                       {"momentum": "buy", "quant": "hold"}, 20.0, 1.0, 4.0, return_pct=20.0)

    replayed = _bootstrap_adaptive_learner_from_history()
    assert replayed == 3

    learner = get_meta_learner()
    assert learner is not None
    assert learner.sample_count == 3
    assert learner.sample_count_for_regime("trending") == 2
    assert learner.sample_count_for_regime("sideways") == 1
    # Validation-gate returns must be oldest-first, exactly as the trades closed.
    assert learner.regime_returns["trending"] == [10.0, -10.0]
    assert learner.regime_returns["sideways"] == [20.0]


def test_bootstrap_is_idempotent_across_restarts(fresh_db):
    _make_closed_trade("d1", "BTC/USD", "trending", "buy",
                       {"momentum": "buy", "quant": "buy"}, 100.0, 1.0, 10.0, return_pct=10.0)

    first = _bootstrap_adaptive_learner_from_history()
    second = _bootstrap_adaptive_learner_from_history()

    assert first == 1
    assert second == 1
    learner = get_meta_learner()
    # A second (restart) bootstrap must NOT double-count samples or returns.
    assert learner.sample_count == 1
    assert learner.regime_returns["trending"] == [10.0]


def test_legacy_rows_without_return_pct_are_reconstructed(fresh_db):
    # pnl 10 on notional 100*1 -> the live exit path would have recorded
    # return_pct = 10.0; this row was closed before that field existed.
    _make_closed_trade("d1", "BTC/USD", "trending", "buy",
                       {"momentum": "buy"}, 100.0, 1.0, 10.0, return_pct=0.0)

    _bootstrap_adaptive_learner_from_history()

    learner = get_meta_learner()
    assert learner.regime_returns["trending"] == [pytest.approx(10.0)]


def test_rows_without_brain_votes_still_count(fresh_db):
    _make_closed_trade("d1", "BTC/USD", "bear", "buy", {}, 100.0, 1.0, -3.0, return_pct=-3.0)

    replayed = _bootstrap_adaptive_learner_from_history()
    assert replayed == 1

    learner = get_meta_learner()
    assert learner.sample_count == 1
    assert learner.sample_count_for_regime("bear") == 1
    # No directional votes -> weights stay equal (5 brains -> 0.2 each), no crash.
    assert all(abs(w - 0.2) < 1e-9 for w in learner.weights["bear"].values())


def test_empty_history_leaves_loaded_state_untouched(fresh_db):
    """A DB outage / empty history must NOT reset an already-learned state."""
    learner = get_meta_learner()
    learner.update(
        {"regime": "sideways", "final_action": "buy", "brain_votes": {"momentum": "buy"}},
        {"net_pnl": 5.0, "return_pct": 5.0},
    )
    assert learner.sample_count == 1

    # No closed snapshots in the DB -> bootstrap must no-op.
    assert _bootstrap_adaptive_learner_from_history() == 0
    assert get_meta_learner().sample_count == 1