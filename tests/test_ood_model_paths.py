"""Regression tests for the OOD discriminator's model paths.

Two separate bugs lived here:

1. OOD_DIR was joined with one '..' too many, resolving to
   <repo>/../models/ood (i.e. /models/ood inside the container) instead of
   <repo>/models/ood. load() therefore never found a model, _is_trained stayed
   False, and the whole check in src/committee/committee.py silently no-op'd.

2. Even after that fix, the weights sat in the git-tracked models/ directory.
   The discriminator is retrained at runtime (hourly, in-process), and models/
   is replaced by the image on every redeploy -- so every retrain was wiped and
   the discriminator came back untrained. Runtime-trained weights must live on
   the persistent data volume: model_store.store_dir()/ood (data/models/ood by
   default, APEX_MODEL_STORE_DIR in tests), the same base the PPO/transformer
   trainers use.
"""
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="src.ood_discriminator imports torch at module level")


def _repo_root() -> Path:
    from src import ood_discriminator

    return Path(ood_discriminator.__file__).resolve().parents[1]


def test_ood_model_dir_is_under_the_persistent_model_store():
    """OOD weights are retrained at runtime, so they must resolve under the
    persistent store (data/models/ood), not the baked models/ directory."""
    from src import model_store, ood_discriminator

    expected = (Path(model_store.store_dir()) / "ood").resolve()
    assert Path(ood_discriminator.OOD_DIR).resolve() == expected


def test_ood_model_dir_is_not_the_baked_models_dir():
    """Guard the specific regression: weights must never resolve back into the
    git-tracked models/ dir that a redeploy replaces."""
    from src import model_store, ood_discriminator

    baked_models = (_repo_root() / "models").resolve()
    ood_dir = Path(ood_discriminator.OOD_DIR).resolve()
    assert ood_dir != baked_models
    assert baked_models not in ood_dir.parents
    # ...and it IS inside the store, which the bot_data volume persists.
    assert Path(model_store.store_dir()).resolve() in ood_dir.parents


def test_ood_model_and_config_live_in_that_dir():
    from src import ood_discriminator

    assert Path(ood_discriminator.OOD_MODEL_PATH).parent.resolve() == Path(ood_discriminator.OOD_DIR).resolve()
    assert Path(ood_discriminator.OOD_CONFIG_PATH).parent.resolve() == Path(ood_discriminator.OOD_DIR).resolve()
