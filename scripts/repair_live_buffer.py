#!/usr/bin/env python3
"""Repair the JSONL experience buffers (data/live_experiences.jsonl etc.).

Rewrites a buffer keeping only well-formed records, after a tolerant read that
strips a UTF-16 BOM and stray NUL bytes. The original is backed up to a
timestamped ``.bak`` next to it unless ``--no-backup`` is passed. Safe to run
on a live bot: the write is atomic (temp file + ``os.replace``) and only
happens when something was actually dropped.

Usage:
    python scripts/repair_live_buffer.py                 # repair the live buffer
    python scripts/repair_live_buffer.py --dry-run       # report only, no writes
    python scripts/repair_live_buffer.py --require-tensor # also drop non-tensor rows
    python scripts/repair_live_buffer.py data/live_experiences.jsonl data/historical_experiences.jsonl
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.live_buffer import repair_jsonl
from src.logging_config import configure_structlog, get_logger

logger = get_logger("repair_live_buffer")

DEFAULT_BUFFERS = [
    "data/live_experiences.jsonl",
    "data/historical_experiences.jsonl",
]


def main(argv: list[str]) -> int:
    configure_structlog()
    args = [a for a in argv[1:] if not a.startswith("--")]
    dry_run = "--dry-run" in argv
    no_backup = "--no-backup" in argv
    require_tensor = "--require-tensor" in argv
    paths = args or DEFAULT_BUFFERS

    exit_code = 0
    for path in paths:
        if not os.path.exists(path):
            logger.warning(f"[LIVE_BUFFER] {path} does not exist; nothing to repair")
            continue
        stats = repair_jsonl(
            path, backup=not no_backup, dry_run=dry_run, require_tensor=require_tensor
        )
        removed = stats["skipped"] + stats.get("dropped_shape", 0)
        logger.info(
            f"[LIVE_BUFFER] {path}: total={stats['total']} kept={stats.get('kept', stats['loaded'])} "
            f"blank={stats['blank']} removed={removed} dry_run={dry_run}"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
