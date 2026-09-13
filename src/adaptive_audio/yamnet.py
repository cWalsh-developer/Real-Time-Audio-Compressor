"""Optional YAMNet ONNX adapter for time-stamped sound-event scores."""

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

import imageio_ffmpeg
import numpy as np

from .classification import EventScores


MODEL_REVISION = "f25b741c2f0bdc6d7e6db24b5fddda23347dbafd"
MODEL_BASE = f"https://huggingface.co/audiomagic/yamnet-onnx/resolve/{MODEL_REVISION}/"
ASSETS = {
    "yamnet.onnx": "d3835ffbbd4a1bb3e777f0ca217b5007907f5171dd5d17c4236b95b2af8f908e",
    "yamnet_class_map.csv": "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2",
}


def download_model(directory: Path) -> None:
    """Fetch pinned model files and verify their SHA-256 digests."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in ASSETS.items():
        target = directory / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_hash:
            continue
        temporary = target.with_suffix(target.suffix + ".download")
        try:
            with urlopen(MODEL_BASE + name, timeout=60) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Downloaded {name} failed SHA-256 verification")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def _category(name: str) -> str:
    lower = name.lower()
    if any(word in lower for word in ("explosion", "gunshot", "gunfire", "engine", "fireworks", "scream", "shout", "yell")):
        return "action"
    if any(word in lower for word in ("speech", "conversation", "narration", "whisper")):
        return "speech"
    if "music" in lower or "musical instrument" in lower:
        return "music"
    return "other"


def group_scores(raw: np.ndarray, names: list[str]) -> dict[str, float]:
    """Collapse multilabel AudioSet scores into four normalized categories."""
    if len(raw) != len(names):
        raise ValueError("Class score count does not match the class map")
    grouped = {label: 0.0 for label in ("speech", "music", "action", "other")}
    for score, name in zip(raw, names):
        category = _category(name)
        grouped[category] = max(grouped[category], float(score))
    total = sum(grouped.values())
    if total <= 0:
        grouped["other"] = 1.0
        return grouped
    return {label: value / total for label, value in grouped.items()}


class YamnetClassifier:
    def __init__(self, directory: Path):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Install the AI extra: python -m pip install -e '.[ai]'") from error
        with (directory / "yamnet_class_map.csv").open(encoding="utf-8") as stream:
            self.names = [row["display_name"] for row in csv.DictReader(stream)]
        self.session = ort.InferenceSession(str(directory / "yamnet.onnx"), providers=["CPUExecutionProvider"])

    def classify(self, audio: np.ndarray, sample_rate: int) -> list[EventScores]:
        if sample_rate != 16000 or audio.ndim != 1:
            raise ValueError("YAMNet requires mono audio at 16 kHz")
        if len(audio) == 0:
            return []
        waveform = np.ascontiguousarray(audio, dtype=np.float32)
        frame_scores = self.session.run(["output_0"], {"waveform": waveform})[0]
        duration = len(audio) / sample_rate
        result = []
        for index, raw in enumerate(frame_scores):
            start = index * 0.48
            if start >= duration:
                break
            result.append(EventScores(start, min(start + 0.96, duration), group_scores(raw, self.names)))
        return result


def classify_wav(path: Path, model_directory: Path) -> list[EventScores]:
    """Decode and resample a WAV with FFmpeg, then run YAMNet."""
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path),
               "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", "16000", "pipe:1"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip())
    samples = np.frombuffer(result.stdout, dtype="<f4")
    return YamnetClassifier(model_directory).classify(samples, 16000)


def write_scores(path: Path, scores: list[EventScores]) -> None:
    path.write_text(json.dumps({"segments": [
        {"start_seconds": item.start_seconds, "end_seconds": item.end_seconds, "scores": item.scores}
        for item in scores
    ]}, indent=2), encoding="utf-8")
