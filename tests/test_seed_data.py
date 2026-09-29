"""src/seed_data.py installs the image's replay buffer into the data volume
only when the volume's buffer is untrainable, and never touches real data."""
import json

import numpy as np

from src.seed_data import BUFFER_NAME, bootstrap_replay_buffer, count_usable_records


def _real(n):
    return "".join(json.dumps({"tensor": np.zeros((32, 11)).tolist(), "label": 1.0}) + "\n" for _ in range(n))


def _noise(n):
    return "".join(json.dumps({"tensor": np.random.randn(128).tolist(), "label": 1.0}) + "\n" for _ in range(n))


def _dirs(tmp_path, volume_content, seed_content=None):
    data, seed = tmp_path / "data", tmp_path / "seed"
    data.mkdir()
    seed.mkdir()
    if volume_content is not None:
        (data / BUFFER_NAME).write_text(volume_content, encoding="utf-8")
    (seed / BUFFER_NAME).write_text(seed_content if seed_content is not None else _real(3), encoding="utf-8")
    return data, seed


def test_noise_only_buffer_is_replaced_and_backed_up(tmp_path):
    data, seed = _dirs(tmp_path, _noise(5))

    assert bootstrap_replay_buffer(str(data), str(seed)) is True

    assert count_usable_records(str(data / BUFFER_NAME)) == 3
    backups = list((data / "backups").iterdir())
    assert len(backups) == 1 and count_usable_records(str(backups[0])) == 0


def test_missing_buffer_is_installed(tmp_path):
    data, seed = _dirs(tmp_path, None)

    assert bootstrap_replay_buffer(str(data), str(seed)) is True
    assert count_usable_records(str(data / BUFFER_NAME)) == 3


def test_buffer_with_any_real_record_is_left_alone(tmp_path):
    original = _noise(5) + _real(1)
    data, seed = _dirs(tmp_path, original)

    assert bootstrap_replay_buffer(str(data), str(seed)) is False
    assert (data / BUFFER_NAME).read_text(encoding="utf-8") == original


def test_no_seed_in_image_is_a_no_op(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / BUFFER_NAME).write_text(_noise(2), encoding="utf-8")

    assert bootstrap_replay_buffer(str(data), str(tmp_path / "no_seed_here")) is False


def test_utf16_header_counts_as_unusable(tmp_path):
    data, seed = _dirs(tmp_path, None)
    (data / BUFFER_NAME).write_bytes(b"\xff\xfe\r\x00\n\x00")

    assert bootstrap_replay_buffer(str(data), str(seed)) is True
