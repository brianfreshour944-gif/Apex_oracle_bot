"""Tolerant reader/repair for the JSONL experience buffers.

The live buffer (``data/live_experiences.jsonl``) is appended to by the running
bot and was historically committed/edited on Windows, which left a UTF-16 BOM
and stray NUL bytes that make a bare ``json.loads``/``readlines`` fail. A single
unreadable byte used to abort an entire nightly retrain, discarding every valid
record in the file. This module is the single reader every consumer uses: it
strips the BOM and NULs, decodes with ``errors="replace"``, and counts + logs
every line it skips with a ``[LIVE_BUFFER]`` marker so a silent drop becomes
visible in ``data/logs/apex_bot.log`` instead of vanishing.
"""

from __future__ import annotations

import json
import os
from typing import Any

from src.logging_config import get_logger

logger = get_logger("live_buffer")


def read_jsonl_tolerant(
    path: str,
    *,
    max_lines: int | None = None,
    buffer_name: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read a JSONL file into a list of parsed dicts, skipping unreadable lines.

    Robust to a UTF-16 BOM (``encoding="utf-8-sig"``), embedded NUL bytes
    (stripped), and undecodable bytes (``errors="replace"`` -> the line is
    rejected individually by ``json.loads`` rather than aborting the read).

    Args:
        path: file to read.
        max_lines: keep only the last N lines (the buffers are append-only, so
            the tail is the newest data).
        buffer_name: label used in log lines (defaults to the basename).

    Returns:
        ``(records, stats)`` where ``stats`` has ``total``/``loaded``/``skipped``/
        ``blank`` counts. Every skipped line is logged as a ``[LIVE_BUFFER]``
        warning with its index and a truncated preview.
    """
    name = buffer_name or os.path.basename(path)
    stats = {"total": 0, "loaded": 0, "skipped": 0, "blank": 0}
    records: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return records, stats
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        logger.warning(f"[LIVE_BUFFER] could not read {name} ({path}): {e}")
        return records, stats

    if max_lines is not None:
        lines = lines[-max_lines:]

    for line_index, raw in enumerate(lines):
        stats["total"] += 1
        # A UTF-16 newline leaves a stray NUL glued onto the next
        # (UTF-8-appended) record; raw NUL is never valid in JSON text.
        line = raw.replace("\x00", "").strip()
        if not line:
            stats["blank"] += 1
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"record is {type(record).__name__}, expected object")
        except Exception as e:
            stats["skipped"] += 1
            logger.warning(
                f"[LIVE_BUFFER] {name}: skipped line {line_index} ({e}): {line[:120]!r}"
            )
            continue
        records.append(record)
        stats["loaded"] += 1

    if stats["skipped"]:
        logger.warning(
            f"[LIVE_BUFFER] {name}: skipped {stats['skipped']}/{stats['total']} "
            f"unreadable line(s); run scripts/repair_live_buffer.py to clean it"
        )
    logger.info(
        f"[LIVE_BUFFER] {name}: loaded {stats['loaded']} record(s) from {stats['total']} line(s)"
    )
    return records, stats


def has_usable_tensor(record: dict[str, Any]) -> bool:
    """True when the record's tensor is a non-empty 2-D (timesteps x features)
    array -- the only shape retrain_transformer.py can train on."""
    tensor = record.get("tensor")
    return bool(
        isinstance(tensor, list) and tensor and isinstance(tensor[0], list)
        and tensor[0] and not isinstance(tensor[0][0], list)
    )


def is_usable_experience(record: dict[str, Any]) -> bool:
    """A record the transformer can train on: a non-empty 2-D (timesteps x
    features) tensor plus a numeric label. Matches the schema
    generate_replay_dataset.py writes and retrain_transformer.py consumes."""
    if not has_usable_tensor(record):
        return False
    label = record.get("label")
    return isinstance(label, (int, float)) and not isinstance(label, bool)


def repair_jsonl(
    path: str,
    *,
    backup: bool = True,
    dry_run: bool = False,
    require_tensor: bool = False,
) -> dict[str, int]:
    """Rewrite a JSONL buffer keeping only well-formed records.

    Reads the file tolerantly, drops lines that are not JSON objects (and, when
    ``require_tensor`` is set, records without a usable tensor/label), and writes
    the survivors back. The original is preserved as a timestamped ``.bak`` next
    to the file unless ``backup=False``. Returns the reader's stats plus
    ``dropped_shape`` when ``require_tensor`` is set.

    Idempotent and atomic: writes to a temp file then ``os.replace``.
    """
    records, stats = read_jsonl_tolerant(path)
    out_stats = dict(stats)
    kept = records
    if require_tensor:
        kept = [r for r in records if is_usable_experience(r)]
        out_stats["dropped_shape"] = len(records) - len(kept)
        if out_stats["dropped_shape"]:
            logger.warning(
                f"[LIVE_BUFFER] {os.path.basename(path)}: dropped "
                f"{out_stats['dropped_shape']} record(s) without a usable tensor/label"
            )
    out_stats["kept"] = len(kept)

    if dry_run or (out_stats["skipped"] == 0 and out_stats.get("dropped_shape", 0) == 0):
        return out_stats

    if backup and os.path.exists(path):
        from datetime import UTC, datetime
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        bak = f"{path}.{stamp}.bak"
        try:
            os.replace(path, bak)
            logger.info(f"[LIVE_BUFFER] backed up {path} -> {bak}")
        except OSError as e:
            logger.warning(f"[LIVE_BUFFER] backup of {path} failed: {e}")

    tmp = f"{path}.repair.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record) + "\n")
    os.replace(tmp, path)
    logger.info(
        f"[LIVE_BUFFER] repaired {os.path.basename(path)}: kept {len(kept)} record(s), "
        f"removed {out_stats['skipped'] + out_stats.get('dropped_shape', 0)}"
    )
    return out_stats
