"""The monthly cull must promote a winning shadow candidate to where the live
transformer actually loads from, with its architecture.

It used to copy only the weights and scaler into data/, a directory no brain
reads, so a winning candidate never went live; and a candidate of a different
size (Bot_C_Heavy is 256-wide) would not have loaded against the old
transformer_config.json even if it had.
"""
import json
import os
import sys
import uuid
from datetime import UTC, datetime

import pytest

import src.db as db
from src import model_store
from src.config import settings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import evolution_cull


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


def test_winning_candidate_is_promoted_into_the_model_store_with_its_architecture(fresh_db, tmp_path, monkeypatch):
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / "Bot_C_Heavy.pth").write_bytes(b"heavy-candidate-weights")
    (candidates / "feature_scaler.pkl").write_bytes(b"candidate-scaler")
    (candidates / "Bot_C_Heavy_config.json").write_text(json.dumps({"layers": 4, "embed": 256, "dropout": 0.3}))
    monkeypatch.setattr(evolution_cull, "CANDIDATES_DIR", str(candidates))
    store = tmp_path / "store"
    monkeypatch.setenv("APEX_MODEL_STORE_DIR", str(store))

    async def no_alert(msg):
        return None
    monkeypatch.setattr(evolution_cull, "send_telegram_alert", no_alert)

    with db.get_db_session() as session:
        session.add(db.ShadowTrade(
            trade_id=str(uuid.uuid4()), candidate_name="Bot_C_Heavy", symbol="BTC/USD", side="buy",
            qty=1.0, entry_price=100.0, exit_price=150.0, status="closed", realized_pnl=50.0,
            closed_at=datetime.now(UTC),
        ))
        session.commit()

    assert evolution_cull.evaluate_and_cull() == 0

    assert model_store.bundle_dir("transformer") == str(store)
    assert (store / "grok_gqa_v9_best.pth").read_bytes() == b"heavy-candidate-weights"
    assert (store / "feature_scaler.pkl").read_bytes() == b"candidate-scaler"
    arch = json.loads((store / "transformer_config.json").read_text())
    assert arch == {"num_layers": 4, "embed_dim": 256, "num_q_heads": 8, "num_kv_heads": 2}


def test_production_defending_its_title_promotes_nothing(fresh_db, tmp_path, monkeypatch):
    monkeypatch.setattr(evolution_cull, "CANDIDATES_DIR", str(tmp_path / "candidates"))
    store = tmp_path / "store"
    monkeypatch.setenv("APEX_MODEL_STORE_DIR", str(store))

    async def no_alert(msg):
        return None
    monkeypatch.setattr(evolution_cull, "send_telegram_alert", no_alert)

    assert evolution_cull.evaluate_and_cull() == 0
    assert not (store / "grok_gqa_v9_best.pth").exists()

