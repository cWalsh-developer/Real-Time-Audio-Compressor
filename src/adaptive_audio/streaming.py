"""Two-pass PCM WAV processing with bounded sample buffers."""

import wave
from pathlib import Path

import numpy as np

from .processing import MODES, apply_gain, frame_measurements, plan_gain
from .classification import EventScores
from .content import semantic_adjustments


def process_wav(source: Path, target: Path, mode: str, chunk_frames: int = 65536,
                labels: list[EventScores] | None = None) -> None:
    """Scan frame levels, then render audio in chunks with one full-file gain plan."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    with wave.open(str(source), "rb") as wav:
        if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2:
            raise ValueError("Only uncompressed 16-bit PCM WAV is supported")
        channels = wav.getnchannels()
        rate = wav.getframerate()
        total = wav.getnframes()
        window = max(1, round(rate * 0.1))
        levels = []
        peaks = []
        while True:
            raw = wav.readframes(window)
            if not raw:
                break
            frame = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            level, peak = frame_measurements(frame)
            levels.append(level)
            peaks.append(peak)
        adjustment = semantic_adjustments(labels, len(levels), rate) if labels is not None else None
        gain_db = plan_gain(np.asarray(levels), np.asarray(peaks), mode, adjustment)
        wav.rewind()
        with wave.open(str(target), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(rate)
            position = 0
            while position < total:
                raw = wav.readframes(min(chunk_frames, total - position))
                frame = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels) / 32768.0
                samples = frame[:, 0] if channels == 1 else frame
                processed = apply_gain(samples, rate, gain_db, position)
                pcm = np.round(np.clip(processed, -1, 32767 / 32768) * 32768).astype("<i2")
                output.writeframesraw(pcm.tobytes())
                position += len(frame)
