"""Regression tests for the OOD discriminator's model paths.

The discriminator exists to veto trades taken in states structurally unlike its
training data, but OOD_DIR was joined with one '..' too many, resolving to
<repo>/../models/ood (i.e. /models/ood inside the container) instead of
<repo>/models/ood. load() therefore never found a model, _is_trained stayed
False, and the whole check in src/committee/committee.py silently no-op'd --
while save() wrote any trained model outside the image/repo.
"""
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="src.ood_discriminator imports torch at module level")


def _repo_root() -> Path:
    from src import ood_discriminator

    return Path(ood_discriminator.__file__).resolve().parents[1]


def test_ood_model_dir_resolves_inside_the_repo():
    from src import ood_discriminator

    expected = (_repo_root() / "models" / "ood").resolve()
    assert Path(ood_discriminator.OOD_DIR).resolve() == expected


def test_ood_model_and_config_live_in_that_dir():
    from src import ood_discriminator

    assert Path(ood_discriminator.OOD_MODEL_PATH).parent.resolve() == Path(ood_discriminator.OOD_DIR).resolve()
    assert Path(ood_discriminator.OOD_CONFIG_PATH).parent.resolve() == Path(ood_discriminator.OOD_DIR).resolve()


def test_ood_paths_agree_with_the_other_model_paths():
    """Keep the OOD path convention in sync with src/config.py's model dirs."""
    from src import ood_discriminator
    from src.config import settings

    transformer_dir = (Path(ood_discriminator.__file__).resolve().parents[1] / settings.TRANSFORMER_MODEL_PATH).parent
    assert Path(ood_discriminator.OOD_DIR).resolve().parent == transformer_dir.resolve()