"""Tests for src/model_store.py, the persistent home for runtime-trained models.

On the VM, models/ comes from the Docker image and is replaced on every
redeploy; only data/ is a persistent volume. These tests pin down that a
promoted model is what loaders get back, that a model newly shipped from git
still wins, and that a half-present bundle is never used.
"""
import os

import pytest

from src import model_store

WEIGHTS, SCALER, CONFIG = model_store.BUNDLES["transformer"]


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    baked = tmp_path / "models"
    store = tmp_path / "store"
    baked.mkdir()
    for name, body in ((WEIGHTS, b"baked-weights"), (SCALER, b"baked-scaler"), (CONFIG, b'{"num_layers": 4}')):
        (baked / name).write_bytes(body)
    monkeypatch.setattr(model_store, "BAKED_DIR", str(baked))
    monkeypatch.setenv("APEX_MODEL_STORE_DIR", str(store))
    return baked, store


def _staged(tmp_path, name, body):
    path = tmp_path / f"new_{name}"
    path.write_bytes(body)
    return str(path)


def test_empty_store_falls_back_to_baked_models(dirs):
    baked, _store = dirs
    assert model_store.bundle_dir("transformer") == str(baked)


def test_promoted_model_is_what_loaders_get_and_missing_files_carry_over(dirs, tmp_path):
    _baked, store = dirs

    model_store.promote("transformer", {WEIGHTS: _staged(tmp_path, WEIGHTS, b"trained")})

    assert model_store.bundle_dir("transformer") == str(store)
    assert (store / WEIGHTS).read_bytes() == b"trained"
    # scaler and config were not retrained, so the active ones travel with it
    assert (store / SCALER).read_bytes() == b"baked-scaler"
    assert (store / CONFIG).read_bytes() == b'{"num_layers": 4}'
    assert model_store.promoted_at("transformer") is not None


def test_new_model_shipped_from_git_supersedes_stored_one(dirs, tmp_path):
    baked, _store = dirs
    model_store.promote("transformer", {WEIGHTS: _staged(tmp_path, WEIGHTS, b"trained")})

    (baked / WEIGHTS).write_bytes(b"new-model-from-git")

    assert model_store.bundle_dir("transformer") == str(baked)


def test_incomplete_store_bundle_is_never_used(dirs):
    baked, store = dirs
    store.mkdir()
    (store / WEIGHTS).write_bytes(b"orphan weights without config or scaler")

    assert model_store.bundle_dir("transformer") == str(baked)


def test_promote_rejects_files_outside_the_bundle(dirs, tmp_path):
    with pytest.raises(ValueError):
        model_store.promote("transformer", {"other.pth": _staged(tmp_path, "other.pth", b"x")})


def test_promote_without_any_source_for_a_file_fails_and_writes_nothing(dirs, tmp_path):
    _baked, store = dirs
    # Nothing baked for the decision transformer weights, and none given.
    with pytest.raises(FileNotFoundError):
        model_store.promote("decision_transformer", {})
    assert not (store / "store_manifest.json").exists()


def test_staging_dir_is_removed_after_promotion(dirs):
    _baked, store = dirs
    with model_store.staging("transformer") as staged:
        path = os.path.join(staged, WEIGHTS)
        with open(path, "wb") as f:
            f.write(b"trained")
        model_store.promote("transformer", {WEIGHTS: path})
    assert not os.path.exists(staged)
    assert (store / WEIGHTS).read_bytes() == b"trained"


def test_transformer_paths_resolve_through_store(dirs, tmp_path):
    _baked, store = dirs
    model_store.promote("transformer", {WEIGHTS: _staged(tmp_path, WEIGHTS, b"trained")})

    assert model_store.transformer_paths() == (
        str(store / WEIGHTS), str(store / SCALER), str(store / CONFIG),
    )


def test_explicit_transformer_path_override_is_respected(dirs, tmp_path, monkeypatch):
    from src.config import settings

    custom = tmp_path / "custom" / "my_model.pth"
    monkeypatch.setattr(settings, "TRANSFORMER_MODEL_PATH", str(custom))

    weights, _scaler, config = model_store.transformer_paths()

    assert weights == str(custom)
    assert config == str(tmp_path / "custom" / CONFIG)


def test_bot_drops_cached_transformer_only_when_a_new_one_was_promoted(dirs, tmp_path):
    import src.bot as bot
    import src.committee.transformer_brain as tb

    tb.set_ml_predictor_override("model", "scaler", "cpu", 11)
    marks = bot._promotion_marks()
    bot._reload_promoted_models(marks)
    assert tb._predictor_initialized is True  # nothing promoted: keep cache (and online updates)

    model_store.promote("transformer", {WEIGHTS: _staged(tmp_path, WEIGHTS, b"trained")})
    bot._reload_promoted_models(marks)

    assert tb._predictor_initialized is False
    assert tb._predictor_instance is None
