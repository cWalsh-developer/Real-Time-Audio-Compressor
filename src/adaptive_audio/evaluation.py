"""Reproducible level measurements for listening comparisons."""

import numpy as np


def measure_audio(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Measure sample peak and 100 ms RMS-window levels in dBFS.

    Window range is P90 minus P10; it is not the standardized LUFS LRA.
    """
    if sample_rate <= 0 or audio.ndim not in (1, 2) or len(audio) == 0:
        raise ValueError("Expected non-empty mono or multichannel audio and a positive sample rate")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains non-finite samples")
    window = max(1, round(sample_rate * 0.1))
    levels = []
    for start in range(0, len(audio), window):
        chunk = audio[start:start + window].astype(np.float64)
        rms = np.sqrt(np.mean(chunk * chunk))
        levels.append(20 * np.log10(max(rms, 1e-8)))
    p10, p50, p90 = np.percentile(levels, [10, 50, 90])
    peak = np.max(np.abs(audio))
    return {
        "duration_seconds": round(len(audio) / sample_rate, 3),
        "sample_peak_dbfs": round(float(20 * np.log10(max(peak, 1e-8))), 2),
        "window_p10_dbfs": round(float(p10), 2),
        "window_p50_dbfs": round(float(p50), 2),
        "window_p90_dbfs": round(float(p90), 2),
        "window_level_range_db": round(float(p90 - p10), 2),
    }
