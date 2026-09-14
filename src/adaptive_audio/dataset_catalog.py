"""Select and download a reproducible, split-preserving DnR v3 subset."""

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import time
import urllib.error
import urllib.request


DATASET = "kwatcharasupat/dnr-v3-multilingual"
REVISION = "24b089134ed926a364e4d165c47938cc76afbbd1"
STEMS = ("mixture", "speech", "music", "sfx")
EVENT_STEMS = ("speech", "music", "sfx_fg", "sfx_bg")
SPLITS = ("train", "val", "test")
FILE_PATTERN = re.compile(r"(?:flac/(?:train|val|test)/\d{6}/(?:mixture|speech|music|sfx)\.flac|manifest/(?:train|val|test)/\d{6}/(?:speech|music|sfx_fg|sfx_bg)\.csv)")


def audio_paths(split, clip_id):
    return [f"flac/{split}/{clip_id}/{stem}.flac" for stem in STEMS]


def metadata_paths(split, clip_id):
    return [f"manifest/{split}/{clip_id}/{stem}.csv" for stem in EVENT_STEMS]


def eligible_clips(names):
    names = set(names)
    result = {split: [] for split in SPLITS}
    for name in sorted(names):
        match = re.fullmatch(r"flac/(train|val|test)/(\d{6})/speech\.flac", name)
        if match:
            split, clip_id = match.groups()
            required = audio_paths(split, clip_id) + metadata_paths(split, clip_id)
            if all(path in names for path in required):
                result[split].append(clip_id)
    return result


def ranked_ids(ids, split, seed):
    return sorted(ids, key=lambda value: hashlib.sha256(f"{seed}/{split}/{value}".encode()).digest())


def source_group(path):
    """Identify source recordings, including FMA's derived mono excerpts.

    The absolute upstream filesystem prefix is never used for local I/O.
    This is a recording audit, not a universal speaker-identity parser.
    """
    parts = PurePosixPath(path.replace("\\", "/")).parts
    dataset = next((p for p in parts if p.startswith(("speech-", "music-", "effects-"))), None)
    if dataset is None:
        raise ValueError(f"unknown source dataset in metadata: {path}")
    stem = PurePosixPath(path).stem
    if dataset == "music-fma":
        stem = re.split(r"_mo\d+", stem)[0]
        stem = re.sub(r"_(left|right)$", "", stem)
    return f"{dataset}/{stem}"


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"path escapes dataset directory: {relative}")
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_url(url):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(
                url, headers={"User-Agent": "adaptive-audio-dataset/0.1"}), timeout=45)
        except urllib.error.HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)


def read_url(url, limit=16 * 1024 * 1024):
    with open_url(url) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"metadata response exceeds {limit} bytes")
    return data


def resolve_url(revision, path):
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{path}"


def validate_plan(plan):
    if plan.get("schema") != 1 or plan.get("dataset") != DATASET:
        raise ValueError("unexpected dataset plan schema or repository")
    if not re.fullmatch(r"[a-f0-9]{40}", plan.get("revision", "")):
        raise ValueError("dataset revision must be an immutable commit")
    seen, group_splits = set(), {}
    if not plan.get("clips"):
        raise ValueError("empty dataset plan")
    for clip in plan["clips"]:
        split, clip_id = clip["split"], clip["id"]
        if split not in SPLITS or not re.fullmatch(r"\d{6}", clip_id):
            raise ValueError("invalid clip identifier or split")
        if (split, clip_id) in seen:
            raise ValueError("duplicate clip")
        seen.add((split, clip_id))
        expected = set(audio_paths(split, clip_id) + metadata_paths(split, clip_id))
        files = clip["files"]
        if len(files) != len(expected) or {entry["path"] for entry in files} != expected:
            raise ValueError("clip files must contain exactly the expected stems and metadata")
        for entry in files:
            if type(entry["size"]) is not int or not 0 < entry["size"] <= 64 * 1024 * 1024:
                raise ValueError("unexpected file size")
            if not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]):
                raise ValueError("invalid SHA-256")
        if not clip.get("source_groups"):
            raise ValueError("missing source recording audit")
        for group in clip["source_groups"]:
            previous = group_splits.setdefault(group, split)
            if previous != split:
                raise ValueError(f"source recording spans dataset splits: {group}")


def verified_download(entry, root, revision, *, opener=open_url):
    if not FILE_PATTERN.fullmatch(entry["path"]) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("unexpected dataset file or revision")
    target = safe_path(root, entry["path"])
    if target.is_file() and target.stat().st_size == entry["size"] and sha256_file(target) == entry["sha256"]:
        return "cached"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = safe_path(root, entry["path"] + ".partial")
    digest, count = hashlib.sha256(), 0
    with opener(resolve_url(revision, entry["path"])) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > entry["size"]:
                raise ValueError(f"download size exceeds plan: {entry['path']}")
            output.write(block)
            digest.update(block)
    if count != entry["size"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"download size/hash mismatch: {entry['path']}")
    temporary.replace(target)
    return "downloaded"


def inspect_clip(split, clip_id, raw_root):
    url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/flac/{split}/{clip_id}"
    tree = json.loads(read_url(url))
    entries = {entry["path"]: entry for entry in tree}
    files, groups = [], set()
    for path in audio_paths(split, clip_id):
        entry = entries[path]
        if entry["size"] != entry["lfs"]["size"]:
            raise ValueError("inconsistent Hub file size")
        files.append({"path": path, "size": entry["size"], "sha256": entry["lfs"]["oid"]})
    for path in metadata_paths(split, clip_id):
        data = read_url(resolve_url(REVISION, path), limit=1024 * 1024)
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
        groups.update(source_group(row["file"]) for row in rows)
        files.append({"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        target = safe_path(raw_root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = safe_path(raw_root, path + ".partial")
        temporary.write_bytes(data)
        temporary.replace(target)
    return {"split": split, "id": clip_id, "source_groups": sorted(groups), "files": files}


def make_plan(raw_root, counts, seed=20260914, workers=4):
    info = json.loads(read_url(f"https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}"))
    if info["sha"] != REVISION or info.get("private") or info.get("gated"):
        raise ValueError("expected the pinned, publicly accessible dataset")
    eligible = eligible_clips(entry["rfilename"] for entry in info["siblings"])
    clips, group_splits, rejected = [], {}, []
    for split in SPLITS:
        wanted = counts[split]
        if type(wanted) is not int or wanted < 1:
            raise ValueError("each split needs at least one clip")
        candidates = ranked_ids(eligible[split], split, seed)
        accepted, offset = 0, 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while accepted < wanted:
                batch = candidates[offset:offset + wanted - accepted]
                if not batch:
                    raise ValueError(f"not enough complete, disjoint {split} examples")
                offset += len(batch)
                for clip in pool.map(lambda clip_id: inspect_clip(split, clip_id, raw_root), batch):
                    conflicts = [group for group in clip["source_groups"] if group in group_splits and group_splits[group] != split]
                    if conflicts:
                        rejected.append({"split": split, "id": clip["id"], "conflicts": conflicts})
                        continue
                    clips.append(clip)
                    for group in clip["source_groups"]:
                        group_splits[group] = split
                    accepted += 1
                    if accepted % 8 == 0 or accepted == wanted:
                        print(f"Planned {split}: {accepted}/{wanted}", flush=True)
    plan = {
        "schema": 1, "dataset": DATASET, "revision": REVISION, "seed": seed,
        "license": "CC-BY-SA-4.0; retain constituent source attribution",
        "license_url": "https://github.com/kwatcharasupat/divide-and-remaster-v3/wiki/Licenses",
        "expected_audio": {"sample_rate": 48000, "channels": 1, "bits_per_sample": 24, "seconds_per_clip": 60},
        "complete_candidates": {split: len(ids) for split, ids in eligible.items()},
        "counts": counts, "rejected_cross_split_recordings": rejected, "clips": clips,
        "download_bytes": sum(entry["size"] for clip in clips for entry in clip["files"]),
        "audit_scope": "Source recording IDs (FMA mono/excerpt variants grouped). Speaker disjointness across all languages is not certified.",
    }
    validate_plan(plan)
    return plan
