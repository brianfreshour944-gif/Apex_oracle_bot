"""Crash-recovery state persistence for Apex Oracle Bot.

Persists the small, per-symbol in-memory state whose loss across a restart
changes trading behavior (audit findings F1-F3):

- RiskManager.peak_prices / TradingStrategy._trailing_peaks  -> without this, a
  trailing stop that should have fired before a crash never fires after it,
  and the peak silently re-anchors at the already-fallen price.
- BotState.cooldowns -> without this, the bot re-enters a symbol immediately
  after a stop-loss that fired seconds before the restart.
- BotState.position_adds -> without this, a restart resets the scale-in cap
  and size decay, letting the bot exceed MAX_POSITION_ADDS at full size.

Design: one JSON file, atomic temp-file-rename writes, debounced (dirty flag
flushed on a timer / at safe points) so the hot path never blocks on disk.
Everything is fail-safe: any read/write error logs and continues with the
in-memory state -- persistence problems must never block trading.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict

from src.logging_config import get_logger

logger = get_logger("persistent_state")

STATE_FILENAME = "bot_state.json"


def _state_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", STATE_FILENAME)


def load_persistent_state() -> Dict[str, Any]:
    """Load persisted bot state. Returns {} on any error (fail-safe)."""
    path = _state_path()
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Could not load persistent bot state from {path} (starting fresh): {e}")
        return {}


def save_persistent_state(data: Dict[str, Any]) -> bool:
    """Atomically persist bot state (temp file + os.replace). Fail-safe."""
    path = _state_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".bot_state_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return True
    except Exception as e:
        logger.warning(f"Could not persist bot state to {path} (non-fatal): {e}")
        return False


class PersistentBotState:
    """Debounced writer for the crash-recovery state file.

    Callers mutate their own in-memory dicts as before, then call
    ``mark_dirty()``. The actual disk write happens at most once per
    ``flush_interval`` seconds, from whatever thread calls ``flush_if_due()``
    (the trading loop; synchronous and cheap). ``flush()`` forces a write.
    """

    def __init__(self, flush_interval: float = 5.0):
        self.flush_interval = flush_interval
        self._dirty = False
        self._last_flush = 0.0

    def mark_dirty(self) -> None:
        self._dirty = True

    def flush(self, snapshot: Dict[str, Any]) -> bool:
        """Force-write the given snapshot. Returns True if written."""
        ok = save_persistent_state(snapshot)
        self._dirty = False
        self._last_flush = time.monotonic()
        return ok

    def flush_if_due(self, snapshot: Dict[str, Any], force: bool = False) -> bool:
        """Write when dirty (immediately) or as a periodic heartbeat.

        The heartbeat exists because some tracked dicts (e.g.
        TradingStrategy._trailing_peaks) are mutated without a dirty-flag
        hook; a periodic write bounds the worst-case loss after a hard kill
        to `flush_interval` seconds regardless.
        """
        now = time.monotonic()
        if force or self._dirty or (now - self._last_flush >= self.flush_interval):
            return self.flush(snapshot)
        return False
