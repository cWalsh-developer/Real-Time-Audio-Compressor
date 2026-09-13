"""MOV extraction and remuxing around the WAV processing engine."""

import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

from .streaming import process_wav


def _run_ffmpeg(arguments: list[str]) -> None:
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *arguments]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "FFmpeg failed")


def process_video(source: Path, target: Path, mode: str) -> None:
    """Replace the first audio track while copying the original video stream."""
    if source.resolve() == target.resolve():
        raise ValueError("Output must differ from input")
    with tempfile.TemporaryDirectory(prefix="adaptive-audio-") as directory:
        original = Path(directory) / "original.wav"
        processed = Path(directory) / "processed.wav"
        _run_ffmpeg(["-i", str(source), "-map", "0:a:0", "-ac", "2", "-c:a", "pcm_s16le", str(original)])
        process_wav(original, processed, mode)
        _run_ffmpeg([
            "-i", str(source), "-i", str(processed), "-map", "0:v:0", "-map", "1:a:0",
            "-map", "0:s?", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
            "-c:s", "copy", "-map_metadata", "0", str(target),
        ])
