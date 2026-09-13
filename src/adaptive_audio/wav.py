"""PCM WAV input and output without external audio libraries."""

import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getcomptype() != "NONE" or source.getsampwidth() != 2:
            raise ValueError("Only uncompressed 16-bit PCM WAV is supported")
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        data = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    samples = data.astype(np.float32).reshape(-1, channels) / 32768.0
    return (samples[:, 0] if channels == 1 else samples), sample_rate


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    pcm = np.round(np.clip(audio, -1, 32767 / 32768) * 32768).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(pcm.tobytes())
