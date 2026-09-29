"""Regression tests for the transformer replay-buffer loader
(scripts/retrain_transformer.py ReplayBufferDataset).

Each case reproduces a failure that stopped nightly training:
- the tracked live buffer was a UTF-16 file, and a bare open() raised
  UnicodeDecodeError outside the per-line try, aborting every retrain;
- the tracked historical buffer held 6,312 flat 128-value random-noise
  vectors, which crashed training at `seq_len, input_dim = sample_x.shape`;
- records were written one symbol at a time, so the "temporal" holdout was
  actually one whole coin.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from retrain_transformer import ReplayBufferDataset

SHAPE = (32, 11)


def _record(label, entry_time=None, shape=SHAPE, fill=0.1):
    rec = {"tensor": np.full(shape, fill).tolist(), "label": label}
    if entry_time:
        rec["entry_time"] = entry_time
    return json.dumps(rec)


def test_utf16_header_does_not_abort_and_appended_records_load(tmp_path):
    path = tmp_path / "live.jsonl"
    utf16_empty = b"\xff\xfe\r\x00\n\x00"  # exact bytes of the committed file
    appended = (_record(1.0) + "\n" + _record(0.0) + "\n").encode("utf-8")
    path.write_bytes(utf16_empty + appended)

    ds = ReplayBufferDataset([str(path)])

    assert len(ds) == 2


def test_unreadable_file_does_not_discard_other_buffers(tmp_path):
    good = tmp_path / "hist.jsonl"
    good.write_text(_record(1.0) + "\n" + _record(0.0) + "\n", encoding="utf-8")
    bad = tmp_path / "live.jsonl"
    bad.write_bytes(b"\xff\xfe\r\x00\n\x00")

    ds = ReplayBufferDataset([str(good), str(bad)])

    assert len(ds) == 2


def test_wrong_shaped_records_are_skipped_not_crashing(tmp_path):
    path = tmp_path / "hist.jsonl"
    noise = [json.dumps({"tensor": np.random.randn(128).tolist(), "label": 1.0}) for _ in range(5)]
    real = [_record(1.0), _record(0.0), _record(1.0)]
    path.write_text("\n".join(noise + real) + "\n", encoding="utf-8")

    ds = ReplayBufferDataset([str(path)])

    assert len(ds) == 3
    assert ds.samples[0][0].shape == SHAPE


def test_records_are_ordered_by_entry_time_across_symbols(tmp_path):
    path = tmp_path / "hist.jsonl"
    # Written symbol by symbol, like the generator: BTC's late trade comes
    # before ETH's early one in the file.
    lines = [
        _record(1.0, "2026-09-10 00:00:00+00:00", fill=0.9),  # BTC, late
        _record(0.0, "2026-09-01 00:00:00+00:00", fill=0.1),  # ETH, early
        _record(1.0, "2026-09-05 00:00:00+00:00", fill=0.5),  # SOL, middle
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ds = ReplayBufferDataset([str(path)])

    fills = [float(x[0, 0]) for x, _ in ds.samples]
    assert fills == pytest.approx([0.1, 0.5, 0.9])


def test_untimestamped_live_records_come_after_timestamped_history(tmp_path):
    hist = tmp_path / "hist.jsonl"
    hist.write_text(_record(1.0, "2026-09-01 00:00:00+00:00", fill=0.1) + "\n", encoding="utf-8")
    live = tmp_path / "live.jsonl"
    live.write_text(_record(0.0, fill=0.7) + "\n", encoding="utf-8")

    ds = ReplayBufferDataset([str(live), str(hist)])

    fills = [float(x[0, 0]) for x, _ in ds.samples]
    assert fills == pytest.approx([0.1, 0.7])
