"""Persistent home for models the bot trains at runtime.

Why this exists: on the VM the bot runs in a Coolify-built image. `models/`
comes from that image (i.e. from git) and is replaced on every redeploy, while
`data/` is a named volume that survives. Trainers used to write their promoted
models straight into `models/`, so every nightly/weekly improvement was wiped
by the next deploy, and the monthly cull wrote to `data/` where no brain ever
looked.

How it works:
- Trainers call `promote(bundle, {filename: freshly_written_path})`. The files
  land in the store (`data/models/` by default) together with the rest of the
  bundle, so weights, architecture config and scaler always travel as a set.
- Loaders call `bundle_path(bundle, filename)`. They get the store's copy when
  the store holds a complete bundle, otherwise the baked copy from `models/`.
- A model committed to git is a deliberate deployment, so it wins again: each
  promotion records the hash of the baked model that was live at the time, and
  if the baked file has changed since, the baked bundle is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from src.logging_config import get_logger

logger = get_logger("model_store")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BAKED_DIR = os.path.join(REPO_ROOT, "models")
MANIFEST_NAME = "store_manifest.json"

# First file in each bundle is the primary weights file used for the
# "has git shipped a newer model?" comparison.
BUNDLES: dict[str, tuple[str, ...]] = {
    "transformer": ("grok_gqa_v9_best.pth", "feature_scaler.pkl", "transformer_config.json"),
    "decision_transformer": ("decision_transformer.pth", "decision_transformer_config.json"),
    "ppo": ("ppo_meta_weights.zip",),
}

_sha_cache: dict[tuple[str, float, int], str] = {}
_superseded_logged: set[str] = set()


def store_dir() -> str:
    """Directory for runtime-trained models; overridable for tests."""
    return os.environ.get("APEX_MODEL_STORE_DIR") or os.path.join(REPO_ROOT, "data", "models")


def state_dir(name: str) -> str:
    """Directory for a trainer's own persistent state (e.g. PBT, Bayesian
    optimisation) inside the store, created on demand."""
    path = os.path.join(store_dir(), name)
    os.makedirs(path, exist_ok=True)
    return path


def _sha(path: str) -> str | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime, st.st_size)
    if key not in _sha_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _sha_cache[key] = h.hexdigest()
    return _sha_cache[key]


def _manifest_path() -> str:
    return os.path.join(store_dir(), MANIFEST_NAME)


def _load_manifest() -> dict:
    try:
        with open(_manifest_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_manifest(manifest: dict) -> None:
    os.makedirs(store_dir(), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=store_dir(), prefix=".manifest_", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, _manifest_path())


def bundle_dir(bundle: str) -> str:
    """Directory holding the active copy of `bundle`: the store when it has a
    complete, non-superseded bundle, otherwise the baked `models/` directory."""
    files = BUNDLES[bundle]
    store = store_dir()
    if not all(os.path.exists(os.path.join(store, f)) for f in files):
        return BAKED_DIR
    entry = _load_manifest().get(bundle) or {}
    trained_on = entry.get("baked_sha")
    current = _sha(os.path.join(BAKED_DIR, files[0]))
    if trained_on and current and current != trained_on:
        if bundle not in _superseded_logged:
            _superseded_logged.add(bundle)
            logger.warning(
                f"Model store: a newer baked '{bundle}' model was deployed from git since the "
                f"stored one was promoted -- using the baked model."
            )
        return BAKED_DIR
    return store


def promoted_at(bundle: str) -> str | None:
    """Timestamp of the last promotion of `bundle` into the store, if any.
    The bot compares this before/after a trainer subprocess to know whether
    it must drop its cached copy of the model."""
    return (_load_manifest().get(bundle) or {}).get("promoted_at")


def bundle_path(bundle: str, filename: str) -> str:
    return os.path.join(bundle_dir(bundle), filename)


def promote(bundle: str, sources: dict[str, str]) -> str:
    """Install freshly trained files as the active `bundle` in the store.

    `sources` maps bundle filenames to paths of newly written files. Bundle
    files not given are carried over from the currently active bundle, so the
    stored set is always complete. Each file is copied to a temp name and then
    renamed, so a crash mid-promotion never leaves a half-written model.
    Returns the store directory.
    """
    files = BUNDLES[bundle]
    unknown = set(sources) - set(files)
    if unknown:
        raise ValueError(f"{sorted(unknown)} are not part of the '{bundle}' bundle {files}")
    active = bundle_dir(bundle)
    store = store_dir()
    os.makedirs(store, exist_ok=True)
    plan = {}
    for name in files:
        src = sources.get(name) or os.path.join(active, name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"cannot promote '{bundle}': no source for {name} ({src})")
        plan[name] = src
    for name, src in plan.items():
        dst = os.path.join(store, name)
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        tmp = dst + ".tmp"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    manifest = _load_manifest()
    manifest[bundle] = {
        "promoted_at": datetime.now(UTC).isoformat(),
        "baked_sha": _sha(os.path.join(BAKED_DIR, files[0])),
        "files": {name: _sha(os.path.join(store, name)) for name in files},
    }
    _save_manifest(manifest)
    logger.info(f"Model store: promoted '{bundle}' to {store}")
    return store


@contextmanager
def staging(bundle: str) -> Iterator[str]:
    """Scratch directory (on the store's filesystem) for a trainer to write a
    new bundle into before calling `promote`; removed afterwards."""
    os.makedirs(store_dir(), exist_ok=True)
    path = tempfile.mkdtemp(dir=store_dir(), prefix=f".staging_{bundle}_")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def transformer_paths() -> tuple[str, str, str]:
    """(weights, scaler, architecture config) for the live transformer.

    An explicit TRANSFORMER_MODEL_PATH / TRANSFORMER_SCALER_PATH override in the
    environment is respected as-is; the defaults resolve through the store.
    """
    from src.config import settings

    fields = type(settings).model_fields
    model_override = settings.TRANSFORMER_MODEL_PATH != fields["TRANSFORMER_MODEL_PATH"].default
    scaler_override = settings.TRANSFORMER_SCALER_PATH != fields["TRANSFORMER_SCALER_PATH"].default
    if model_override or scaler_override:
        model = settings.TRANSFORMER_MODEL_PATH
        return model, settings.TRANSFORMER_SCALER_PATH, os.path.join(os.path.dirname(model), "transformer_config.json")
    d = bundle_dir("transformer")
    return (
        os.path.join(d, "grok_gqa_v9_best.pth"),
        os.path.join(d, "feature_scaler.pkl"),
        os.path.join(d, "transformer_config.json"),
    )
