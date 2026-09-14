import json

import numpy as np
import pytest

from adaptive_audio.training_data import iter_batches, read_windows


def write_dataset(root):
    cache = root / "prepared-16000/train/000001.npz"
    cache.parent.mkdir(parents=True)
    arrays = {
        "mixture": np.arange(12, dtype=np.float32),
        "speech": np.arange(12, dtype=np.float32) + 1,
        "music": np.arange(12, dtype=np.float32) + 2,
        "sfx": np.arange(12, dtype=np.float32) + 3,
    }
    np.savez(cache, **arrays)
    index = root / "windows-16000-train.jsonl"
    entries = [
        {"split": "train", "clip_id": "000001", "cache": "prepared-16000/train/000001.npz",
         "start": 0, "frames": 4, "sample_rate": 16000},
        {"split": "train", "clip_id": "000001", "cache": "prepared-16000/train/000001.npz",
         "start": 4, "frames": 4, "sample_rate": 16000},
    ]
    index.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")


def test_batches_load_aligned_stems_and_preserve_window_order(tmp_path):
    write_dataset(tmp_path)

    batches = list(iter_batches(tmp_path, "train", 2, shuffle=False))

    assert len(batches) == 1
    np.testing.assert_array_equal(batches[0]["mixture"], [[0, 1, 2, 3], [4, 5, 6, 7]])
    np.testing.assert_array_equal(batches[0]["speech"], [[1, 2, 3, 4], [5, 6, 7, 8]])


def test_batches_are_deterministically_shuffled(tmp_path):
    write_dataset(tmp_path)

    first = list(iter_batches(tmp_path, "train", 1, seed=7))
    second = list(iter_batches(tmp_path, "train", 1, seed=7))

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left["mixture"], right["mixture"])


def test_invalid_index_and_batch_size_are_rejected(tmp_path):
    write_dataset(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        list(iter_batches(tmp_path, "train", 0))
    (tmp_path / "windows-16000-val.jsonl").write_text(
        json.dumps({"split": "train", "sample_rate": 16000, "cache": "x", "start": 0, "frames": 1}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mismatch"):
        read_windows(tmp_path, "val")