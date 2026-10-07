"""Install the git-tracked transformer replay buffer into the data volume
when the volume's copy has nothing trainable in it.

On the VM, data/ is a Docker named volume. Docker fills a named volume from
the image only when the volume is first created, so a buffer fixed in git
never reaches it afterwards. The Dockerfile keeps a copy of the tracked buffer
at /app/seed; at startup this replaces the volume's buffer with it, but only
when the volume's buffer has zero usable records (e.g. the 6,312 flat
random-noise vectors from REPLAY_FAST_MODE). A buffer with any real record is
never touched, and a replaced file is kept as a timestamped backup.
"""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime

from src.live_buffer import has_usable_tensor, read_jsonl_tolerant
from src.logging_config import get_logger

logger = get_logger("seed_data")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUFFER_NAME = "historical_experiences.jsonl"


def count_usable_records(path: str) -> int:
    """Records whose tensor is a non-empty 2-D (timesteps x features) array,
    the only shape retrain_transformer.py can train on."""
    records, _ = read_jsonl_tolerant(path)
    return sum(1 for record in records if has_usable_tensor(record))


def bootstrap_replay_buffer(data_dir: str | None = None, seed_dir: str | None = None) -> bool:
    """Returns True if the seed buffer was installed."""
    data_dir = data_dir or os.path.join(REPO_ROOT, "data")
    seed_dir = seed_dir or os.environ.get("APEX_SEED_DIR") or os.path.join(REPO_ROOT, "seed")
    seed = os.path.join(seed_dir, BUFFER_NAME)
    target = os.path.join(data_dir, BUFFER_NAME)

    if not os.path.exists(seed):
        return False
    current = count_usable_records(target)
    if current > 0:
        logger.info(f"Replay buffer has {current} usable record(s); leaving it as is.")
        return False
    seeded = count_usable_records(seed)
    if seeded == 0:
        return False

    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(target):
        backup_dir = os.path.join(data_dir, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(backup_dir, f"historical_experiences_unusable_{stamp}.jsonl")
        shutil.copyfile(target, backup)
        logger.warning(f"Replay buffer had no usable records; backed it up to {backup}")
    tmp = target + ".tmp"
    shutil.copyfile(seed, tmp)
    os.replace(tmp, target)
    logger.info(f"Installed seed replay buffer ({seeded} records) at {target}")
    return True
