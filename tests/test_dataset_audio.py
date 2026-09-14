from pathlib import Path
import json

import numpy as np
import pytest

from adaptive_audio.dataset_audio import flac_info, audit_audio, cache_clip, window_index


def test_reads_flac_streaminfo_without_decoding(tmp_path):
    path = tmp_path / "test.flac"
    streaminfo = bytearray(34)
    packed = (48000 << 44) | (0 << 41) | (23 << 36) | 2880000
    streaminfo[10:18] = packed.to_bytes(8, "big")
    path.write_bytes(b"fLaC" + b"\x80\x00\x00\x22" + streaminfo)
    assert flac_info(path) == {"sample_rate": 48000, "channels": 1, "bits_per_sample": 24, "frames": 2880000}
    path.write_bytes(b"not audio")
    with pytest.raises(ValueError):
        flac_info(path)


def arrays():
    speech = np.zeros(32000, np.float32)
    speech[4000:16000] = np.sin(np.arange(12000) / 10).astype(np.float32) * 0.1
    music = np.full(32000, 0.02, np.float32)
    sfx = np.zeros(32000, np.float32)
    sfx[8000] = 0.2
    return dict(speech=speech, music=music, sfx=sfx, mixture=speech + music + sfx)


def test_aligned_stems_reconstruct_and_bad_alignment_is_rejected():
    audio = arrays()
    assert audit_audio(audio)["reconstruction_max_error"] < 1e-6
    audio["speech"] = np.roll(audio["speech"], 200)
    with pytest.raises(ValueError, match="reconstruct"):
        audit_audio(audio)


def test_invalid_audio_is_rejected():
    audio = arrays()
    audio["sfx"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        audit_audio(audio)
    audio = arrays()
    audio["music"] = audio["music"][:-1]
    with pytest.raises(ValueError, match="length"):
        audit_audio(audio)


def test_cache_retains_speech_exactly_and_rebuilds_corruption(tmp_path):
    path = tmp_path / "train/000001.npz"
    audio = arrays()
    first = cache_clip(path, audio, "fingerprint", 16000)
    with np.load(path, allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache["speech"], audio["speech"])
        np.testing.assert_array_equal(cache["mixture"], audio["mixture"])
    path.write_bytes(b"corrupt")
    repaired = cache_clip(path, audio, "fingerprint", 16000)
    assert repaired["sha256"] == first["sha256"]


def test_window_index_keeps_whole_clips_in_original_splits():
    clips = [{"split": split, "id": "000001", "frames": 60 * 16000,
              "cache": f"prepared/{split}/000001.npz"} for split in ["train", "val", "test"]]
    index = window_index(clips, sample_rate=16000, seconds=4)
    assert len(index) == 45
    for entry in index:
        assert f"/{entry['split']}/" in entry["cache"]
        assert entry["frames"] == 64000
        assert 0 <= entry["start"] < 60 * 16000
