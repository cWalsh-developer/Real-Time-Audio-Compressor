import hashlib
from pathlib import Path

import pytest

from adaptive_audio.dataset_catalog import (
    audio_paths, metadata_paths, eligible_clips, ranked_ids, source_group,
    validate_plan, verified_download, DATASET, REVISION,
)


def test_only_complete_examples_are_eligible_and_splits_stay_separate():
    names = audio_paths("train", "000001") + metadata_paths("train", "000001")
    names += audio_paths("val", "000001") + metadata_paths("val", "000001")[:-1]
    names += audio_paths("test", "000002") + metadata_paths("test", "000002")
    assert eligible_clips(names) == {"train": ["000001"], "val": [], "test": ["000002"]}
    assert ranked_ids(["000003", "000002", "000001"], "train", 7) == ranked_ids(
        ["000001", "000003", "000002"], "train", 7)


def test_music_variants_from_one_recording_share_a_group():
    assert source_group("/old/root/music-fma/audio/train/144788_mo003_right.wav") == source_group(
        "/new/root/music-fma/audio/test/144788_mo001_left.wav")
    assert source_group("/root/effects-fsd50k/audio/train/123.wav") != source_group(
        "/root/effects-fsd50k/audio/train/456.wav")
    with pytest.raises(ValueError):
        source_group("/unknown/not-a-dataset/example.wav")


def fixture_plan():
    clips = []
    for split in ["train", "val", "test"]:
        files = [{"path": p, "size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
                 for p in audio_paths(split, "000001") + metadata_paths(split, "000001")]
        clips.append({"split": split, "id": "000001", "source_groups": [f"source/{split}"], "files": files})
    return {"schema": 1, "dataset": DATASET, "revision": REVISION, "clips": clips}


def test_plan_rejects_cross_split_sources_and_path_escape():
    plan = fixture_plan()
    validate_plan(plan)
    plan["clips"][1]["source_groups"] = ["source/train"]
    with pytest.raises(ValueError, match="source.*split"):
        validate_plan(plan)
    plan = fixture_plan()
    plan["clips"][0]["files"][0]["path"] = "../../outside.flac"
    with pytest.raises(ValueError, match="files"):
        validate_plan(plan)


def test_plan_rejects_missing_stems_and_duplicate_examples():
    plan = fixture_plan()
    plan["clips"][0]["files"].pop()
    with pytest.raises(ValueError):
        validate_plan(plan)
    plan = fixture_plan()
    plan["clips"].append(plan["clips"][0])
    with pytest.raises(ValueError, match="duplicate"):
        validate_plan(plan)


def test_verified_download_reuses_complete_file_and_rejects_corrupt_response(tmp_path):
    import io

    entry = {"path": "flac/train/000001/speech.flac", "size": 3,
             "sha256": hashlib.sha256(b"abc").hexdigest()}
    calls = []
    def good(url):
        calls.append(url)
        return io.BytesIO(b"abc")
    assert verified_download(entry, tmp_path, REVISION, opener=good) == "downloaded"
    assert verified_download(entry, tmp_path, REVISION, opener=good) == "cached"
    assert len(calls) == 1
    target = tmp_path / entry["path"]
    target.write_bytes(b"old")
    with pytest.raises(ValueError, match="hash"):
        verified_download(entry, tmp_path, REVISION, opener=lambda _: io.BytesIO(b"bad"))
    assert target.read_bytes() == b"old"
    assert verified_download(entry, tmp_path, REVISION, opener=good) == "downloaded"
    assert target.read_bytes() == b"abc"


def test_download_rejects_oversize_response_and_symlink_escape(tmp_path):
    import io

    entry = {"path": "flac/train/000001/speech.flac", "size": 3,
             "sha256": hashlib.sha256(b"abc").hexdigest()}
    with pytest.raises(ValueError, match="size"):
        verified_download(entry, tmp_path, REVISION, opener=lambda _: io.BytesIO(b"abcd"))
    assert not (tmp_path / entry["path"]).exists()
