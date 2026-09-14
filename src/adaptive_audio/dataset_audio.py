"""Verify aligned DnR stems and prepare a float32 cache for initial training."""

import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np

from .dataset_catalog import STEMS, SPLITS, safe_path, sha256_file, source_group, validate_plan


PREPARATION_VERSION = 1


def flac_info(path):
    with Path(path).open("rb") as stream:
        header = stream.read(8)
        if len(header) != 8 or header[:4] != b"fLaC" or header[4] & 0x7f != 0 or int.from_bytes(header[5:8], "big") != 34:
            raise ValueError(f"invalid FLAC STREAMINFO: {path}")
        info = stream.read(34)
    if len(info) != 34:
        raise ValueError(f"truncated FLAC STREAMINFO: {path}")
    packed = int.from_bytes(info[10:18], "big")
    return {"sample_rate": packed >> 44, "channels": ((packed >> 41) & 7) + 1,
            "bits_per_sample": ((packed >> 36) & 31) + 1, "frames": packed & ((1 << 36) - 1)}


def decode_mono(path, sample_rate):
    result = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-threads", "1", "-i", str(path), "-map", "0:a:0", "-ar", str(sample_rate),
        "-ac", "1", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ], capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def audit_audio(audio):
    if set(audio) != set(STEMS):
        raise ValueError("expected mixture, speech, music and sfx")
    shapes = {array.shape for array in audio.values()}
    if len(shapes) != 1 or not all(array.ndim == 1 and array.size for array in audio.values()):
        raise ValueError("all stems must have the same nonzero mono length")
    if not all(np.all(np.isfinite(array)) for array in audio.values()):
        raise ValueError("audio must be finite")
    difference = audio["mixture"].astype(np.float64) - sum(audio[name].astype(np.float64) for name in ("speech", "music", "sfx"))
    error = float(np.max(np.abs(difference)))
    if error > 1e-5:
        raise ValueError(f"stems do not reconstruct the mixture (max error {error:.6g})")
    return {
        "frames": len(audio["mixture"]), "reconstruction_max_error": error,
        "levels_dbfs": {name: float(10 * np.log10(np.mean(array.astype(np.float64) ** 2) + 1e-20)) for name, array in audio.items()},
        "peaks": {name: float(np.max(np.abs(array))) for name, array in audio.items()},
    }


def cache_clip(path, audio, fingerprint, sample_rate):
    stats = audit_audio(audio)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **audio, sample_rate=np.array(sample_rate, np.int32))
    temporary.replace(path)
    record = {"fingerprint": fingerprint, "sha256": sha256_file(path), "sample_rate": sample_rate, **stats}
    path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def prepare_clip(clip, root, revision, sample_rate=16000):
    import csv

    if sample_rate not in (16000, 48000):
        raise ValueError("cache sample rate must be 16000 or 48000")
    root = Path(root)
    raw = root / "raw"
    actual_groups = set()
    for entry in clip["files"]:
        path = safe_path(raw, entry["path"])
        if not path.is_file() or path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"missing or corrupt source: {entry['path']}; rerun download")
        if entry["path"].endswith(".csv"):
            with path.open(encoding="utf-8", newline="") as stream:
                actual_groups.update(source_group(row["file"]) for row in csv.DictReader(stream))
    if actual_groups != set(clip["source_groups"]):
        raise ValueError("source metadata does not match the plan's recording audit")
    signature = {"version": PREPARATION_VERSION, "revision": revision, "sample_rate": sample_rate,
                 "files": clip["files"], "ffmpeg": imageio_ffmpeg.get_ffmpeg_version()}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    relative = f"prepared-{sample_rate}/{clip['split']}/{clip['id']}.npz"
    target = safe_path(root, relative)
    sidecar = safe_path(root, str(Path(relative).with_suffix(".json")))
    if target.is_file() and sidecar.is_file():
        try:
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            if saved["fingerprint"] == fingerprint and saved["sha256"] == sha256_file(target):
                return {"split": clip["split"], "id": clip["id"], "cache": relative, "reused": True, **saved}
        except (ValueError, KeyError):
            pass
    audio = {}
    for stem in STEMS:
        path = safe_path(raw, f"flac/{clip['split']}/{clip['id']}/{stem}.flac")
        info = flac_info(path)
        if info != {"sample_rate": 48000, "channels": 1, "bits_per_sample": 24, "frames": 60 * 48000}:
            raise ValueError(f"unexpected source format: {path}: {info}")
        audio[stem] = decode_mono(path, sample_rate)
        if len(audio[stem]) != 60 * sample_rate:
            raise ValueError(f"unexpected decoded duration: {path}")
    record = cache_clip(target, audio, fingerprint, sample_rate)
    return {"split": clip["split"], "id": clip["id"], "cache": relative, "reused": False, **record}


def window_index(clips, sample_rate=16000, seconds=4):
    frames = round(seconds * sample_rate)
    if frames < 1:
        raise ValueError("window duration must be positive")
    return [{"split": clip["split"], "clip_id": clip["id"], "cache": clip["cache"],
             "start": start, "frames": frames, "sample_rate": sample_rate}
            for clip in clips for start in range(0, clip["frames"] - frames + 1, frames)]


def prepare_dataset(plan, root, sample_rate=16000):
    validate_plan(plan)
    records = []
    for index, clip in enumerate(plan["clips"], 1):
        records.append(prepare_clip(clip, root, plan["revision"], sample_rate))
        if index % 4 == 0 or index == len(plan["clips"]):
            print(f"Prepared {index}/{len(plan['clips'])} clips", flush=True)
    windows = window_index(records, sample_rate)
    for split in SPLITS:
        target = Path(root) / f"windows-{sample_rate}-{split}.jsonl"
        target.write_text("".join(json.dumps(item) + "\n" for item in windows if item["split"] == split), encoding="utf-8")
    report = {
        "dataset": plan["dataset"], "revision": plan["revision"], "sample_rate": sample_rate,
        "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "minutes": {split: sum(item["frames"] / sample_rate / 60 for item in records if item["split"] == split) for split in SPLITS},
        "windows": {split: sum(item["split"] == split for item in windows) for split in SPLITS},
        "max_reconstruction_error": max(item["reconstruction_max_error"] for item in records),
        "clips": records,
        "limitations": ["Mono source; no stereo validation", "Initial small subset, not a production training corpus",
                        "Preserved upstream splits with source-recording audit; speaker disjointness is not universally verified",
                        "No training or augmentation has run; 4-second windows are an index, not extra independent recordings"],
    }
    (Path(root) / f"preparation-{sample_rate}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
