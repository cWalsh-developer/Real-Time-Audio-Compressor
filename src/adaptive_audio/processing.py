"""Deterministic short-window gain planning and application."""

import numpy as np


MODES = {
    "cinema": ((-32.0, -14.0), (1.0, -2.0)),
    "balanced": ((-30.0, -20.0, -10.0), (6.0, 0.0, -7.5)),
    "night": ((-32.0, -20.0), (8.0, -12.0)),
}


def frame_measurements(audio: np.ndarray) -> tuple[float, float]:
    rms = np.sqrt(np.mean(np.square(audio, dtype=np.float64)))
    level = 20 * np.log10(max(rms, 1e-8))
    peak = np.max(np.abs(audio))
    return level, peak


def plan_gain(levels: np.ndarray, peaks: np.ndarray, mode: str, adjustment_db: np.ndarray | None = None) -> np.ndarray:
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if len(levels) == 0:
        return np.empty(0)
    level_points, gain_points = MODES[mode]
    gain_db = np.interp(levels, level_points, gain_points)
    if adjustment_db is not None:
        if adjustment_db.shape != gain_db.shape:
            raise ValueError("Semantic adjustment must match analysis frames")
        gain_db += adjustment_db
    radius = 5
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / 2.0) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(gain_db, (radius, radius), mode="edge")
    smooth = np.convolve(padded, kernel, mode="valid")
    ceiling = 10 ** (-1 / 20)
    peak_cap_db = 20 * np.log10(ceiling / np.maximum(peaks, 1e-8))
    return np.minimum(smooth, peak_cap_db)


def apply_gain(audio: np.ndarray, sample_rate: int, gain_db: np.ndarray, start: int = 0) -> np.ndarray:
    window = max(1, round(sample_rate * 0.1))
    frame_centers = np.arange(len(gain_db)) * window + window / 2
    per_sample_db = np.interp(np.arange(start, start + len(audio)), frame_centers, gain_db)
    gain = 10 ** (per_sample_db / 20)
    result = audio.astype(np.float64) * (gain[:, None] if audio.ndim == 2 else gain)
    ceiling = 10 ** (-1 / 20)
    sample_peaks = np.max(np.abs(result), axis=1) if audio.ndim == 2 else np.abs(result)
    final_scale = np.minimum(1.0, ceiling / np.maximum(sample_peaks, 1e-8))
    result *= final_scale[:, None] if audio.ndim == 2 else final_scale
    return result.astype(np.float32)


def process_audio(audio: np.ndarray, sample_rate: int, mode: str) -> np.ndarray:
    """Apply one shared, smoothly varying gain to mono or multichannel audio.

    Input and output are floating-point samples in the [-1, 1] range. This
    baseline uses RMS dBFS, not perceptual LUFS or semantic classification.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if sample_rate <= 0 or audio.ndim not in (1, 2):
        raise ValueError("Expected mono or multichannel audio and a positive sample rate")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains non-finite samples")
    if len(audio) == 0:
        return audio.astype(np.float32, copy=True)

    window = max(1, round(sample_rate * 0.1))
    frame_count = (len(audio) + window - 1) // window
    levels = np.empty(frame_count)
    peaks = np.empty(frame_count)
    for index in range(frame_count):
        chunk = audio[index * window:(index + 1) * window]
        levels[index], peaks[index] = frame_measurements(chunk)
    return apply_gain(audio, sample_rate, plan_gain(levels, peaks, mode))
