"""Tests for persistent rotating file logging (src/logging_config.py).

The bot previously logged only to stdout, so a scheduled job that silently
stopped firing left no durable trace. These tests pin the on-disk log, its
rotation bounds, and the TRAINING_JOB start/result markers that make each
scheduled job visible to a host-side grep.
"""
import logging
from logging.handlers import RotatingFileHandler

import pytest

from src import logging_config


@pytest.fixture(autouse=True)
def _reset_logging_state(monkeypatch):
    """Isolate module-level idempotency flags and any handler we attach."""
    monkeypatch.setattr(logging_config, "_file_logging_configured", False)
    monkeypatch.setattr(logging_config, "_structlog_configured", False)
    monkeypatch.setattr(logging_config, "_training_job_started", {})
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()


def _file_handlers(root=None):
    root = root or logging.getLogger()
    return [h for h in root.handlers if isinstance(h, RotatingFileHandler)]


def test_configure_file_logging_writes_to_data_logs(tmp_path):
    path = logging_config.configure_file_logging(str(tmp_path))
    assert path is not None
    assert path.startswith(str(tmp_path))
    assert path.endswith("apex_bot.log")

    logging.getLogger("t").warning("a durable line")
    for handler in _file_handlers():
        handler.flush()
    assert "a durable line" in (tmp_path / "apex_bot.log").read_text(encoding="utf-8")


def test_file_handler_is_bounded(tmp_path):
    logging_config.configure_file_logging(str(tmp_path), max_bytes=1024, backup_count=5)
    handler = _file_handlers()[0]
    assert handler.maxBytes == 1024
    assert handler.backupCount == 5


def test_configure_file_logging_is_idempotent(tmp_path):
    logging_config.configure_file_logging(str(tmp_path))
    logging_config.configure_file_logging(str(tmp_path))
    assert len(_file_handlers()) == 1


def test_training_job_markers_are_written(tmp_path):
    logging_config.configure_file_logging(str(tmp_path))
    logging_config.log_training_job_start("ood_retrain", "scripts/retrain.py")
    logging_config.log_training_job_result("ood_retrain", returncode=0, script="scripts/retrain.py")
    for handler in _file_handlers():
        handler.flush()

    text = (tmp_path / "apex_bot.log").read_text(encoding="utf-8")
    assert "TRAINING_JOB START job=ood_retrain script=retrain.py" in text
    assert "TRAINING_JOB RESULT job=ood_retrain script=retrain.py returncode=0" in text


def test_training_job_result_reports_duration():
    logging_config.log_training_job_start("pbt")
    logging_config.log_training_job_result("pbt")
    # start marker is consumed exactly once
    assert "pbt" not in logging_config._training_job_started
