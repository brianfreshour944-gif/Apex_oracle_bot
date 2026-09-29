"""The per-trade learning path in bot._record_committee_outcome.

After every closed trade the bot (1) appends the trade to the transformer's
live replay buffer and (2) takes one online gradient step on the live model.
The online step used to call scaler.transform() on a tensor that was already
scaled (transformer_brain.py stores data_scaled), training the model on
inputs roughly 100x off the scale it predicts on. And a crashed lock type
(asyncio.Lock used with a sync `with`) previously made the step a no-op.
"""
import asyncio
import json

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

import src.bot as bot
import src.db as db
from src.committee.transformer_brain import GrokGQA_Transformer
from src.config import settings


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


class SpyScaler(StandardScaler):
    calls = 0

    def transform(self, X, copy=None):
        SpyScaler.calls += 1
        return super().transform(X, copy=copy)


@pytest.fixture
def tiny_predictor(monkeypatch):
    torch.manual_seed(0)
    model = GrokGQA_Transformer(input_dim=11, seq_len=32, num_layers=1, embed_dim=16,
                                num_q_heads=4, num_kv_heads=2)
    model.eval()
    scaler = SpyScaler().fit(np.random.RandomState(0).randn(200, 11) * 5 + 20)
    SpyScaler.calls = 0
    predictor = {"model": model, "scaler": scaler, "device": torch.device("cpu")}
    monkeypatch.setattr("src.committee.transformer_brain.get_ml_predictor", lambda: predictor)
    # Past warm-up with a visible learning rate, so one step measurably moves
    # the weights (the real schedule starts at 1e-7, below float32 resolution).
    monkeypatch.setattr(bot._state, "_transformer_online_updates", bot._state._transformer_online_warmup_steps)
    monkeypatch.setattr(bot._state, "_transformer_online_lr_base", 1e-2)
    return predictor


async def test_closed_trade_updates_model_and_live_buffer(fresh_db, tiny_predictor, tmp_path, monkeypatch):
    buffer = tmp_path / "live.jsonl"
    monkeypatch.setattr(bot, "LIVE_BUFFER_PATH", str(buffer))
    tensor = (np.random.RandomState(1).randn(32, 11) * 0.9).astype(np.float32)  # already on model scale
    db.save_decision_snapshot(
        decision_id="dec-online", symbol="BTC/USD", regime="trending", final_action="buy",
        confidence=0.7, size_multiplier=1.0, entry_price=100.0, qty=1.0, brain_votes={"transformer": "buy"},
        tensor_state_json=json.dumps({"transformer": tensor.tolist()}),
    )
    model = tiny_predictor["model"]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    tasks_before = set(bot._state._background_tasks)

    await bot._record_committee_outcome("BTC/USD", 101.0, exit_reason="signal_close",
                                        entry_price=100.0, qty=1.0, commission=0.0)
    new_tasks = [t for t in bot._state._background_tasks - tasks_before if t.get_name() == "transformer_online"]
    await asyncio.gather(*new_tasks)

    changed = any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert new_tasks, "no online training step was scheduled"
    assert changed, "the online step did not change any model weight"
    assert SpyScaler.calls == 0, "the already-scaled tensor was scaled a second time"

    lines = [json.loads(line) for line in buffer.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["label"] == 1.0
    assert np.array(lines[0]["tensor"]).shape == (32, 11)
    assert lines[0].get("entry_time")
