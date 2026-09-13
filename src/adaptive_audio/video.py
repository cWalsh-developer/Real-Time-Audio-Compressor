"""MOV extraction and remuxing around the WAV processing engine."""

import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

from .streaming import process_wav
from .classification import EventScores
from .yamnet import classify_wav, download_model


def _run_ffmpeg(arguments: list[str]) -> None:
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *arguments]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "FFmpeg failed")


def process_video(source: Path, target: Path, mode: str, labels: list[EventScores] | None = None,
                  use_ai: bool = False, model_directory: Path = Path("models/yamnet")) -> None:
    """Replace the first audio track while copying the original video stream."""
    if source.resolve() == target.resolve():
        raise ValueError("Output must differ from input")
    with tempfile.TemporaryDirectory(prefix="adaptive-audio-") as directory:
        original = Path(directory) / "original.wav"
        processed = Path(directory) / "processed.wav"
        _run_ffmpeg(["-i", str(source), "-map", "0:a:0", "-ac", "2", "-c:a", "pcm_s16le", str(original)])
        if use_ai:
            download_model(model_directory)
            labels = classify_wav(original, model_directory)
        process_wav(original, processed, mode, labels=labels)
        _run_ffmpeg([
            "-i", str(source), "-i", str(processed), "-map", "0:v:0", "-map", "1:a:0",
            "-map", "0:s?", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
            "-c:s", "copy", "-map_metadata", "0", str(target),
        ])
