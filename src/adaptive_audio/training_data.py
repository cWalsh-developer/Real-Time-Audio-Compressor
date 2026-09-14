"""Load prepared DnR windows into deterministic training batches."""

import json
from pathlib import Path

import numpy as np

from .dataset_catalog import SPLITS, safe_path


STEMS = ("mixture", "speech", "music", "sfx")


def read_windows(root, split, sample_rate=16000):
    """Read and validate the indexed windows for one dataset split."""
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if sample_rate not in (16000, 48000):
        raise ValueError("sample rate must be 16000 or 48000")
    index_path = Path(root) / f"windows-{sample_rate}-{split}.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    windows = []
    with index_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                window = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid window index at line {line_number}") from error
            if window.get("split") != split or window.get("sample_rate") != sample_rate:
                raise ValueError(f"window index mismatch at line {line_number}")
            if not isinstance(window.get("cache"), str) or not isinstance(window.get("start"), int):
                raise ValueError(f"invalid window entry at line {line_number}")
            if not isinstance(window.get("frames"), int) or window["start"] < 0 or window["frames"] < 1:
                raise ValueError(f"invalid window bounds at line {line_number}")
            windows.append(window)
    if not windows:
        raise ValueError(f"empty window index: {index_path}")
    return windows


def _load_window(root, window):
    cache_path = safe_path(root, window["cache"])
    with np.load(cache_path, allow_pickle=False) as cache:
        if set(STEMS) - set(cache.files):
            raise ValueError(f"cache is missing stems: {cache_path}")
        arrays = {stem: np.asarray(cache[stem], dtype=np.float32) for stem in STEMS}
    start, end = window["start"], window["start"] + window["frames"]
    if any(array.ndim != 1 or end > array.size for array in arrays.values()):
        raise ValueError(f"window exceeds cache bounds: {cache_path}")
    return {stem: arrays[stem][start:end] for stem in STEMS}


def iter_batches(root, split, batch_size, *, sample_rate=16000, shuffle=True, seed=0):
    """Yield batches containing mixture and aligned target stems."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    windows = read_windows(root, split, sample_rate)
    order = np.arange(len(windows))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for offset in range(0, len(order), batch_size):
        selected = [_load_window(root, windows[index]) for index in order[offset:offset + batch_size]]
        yield {stem: np.stack([item[stem] for item in selected]) for stem in STEMS}