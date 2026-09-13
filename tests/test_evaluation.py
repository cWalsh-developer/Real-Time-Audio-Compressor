import json
import subprocess
import sys
import wave

import numpy as np

from adaptive_audio.evaluation import measure_audio


def test_metrics_capture_peak_and_window_level_spread():
    rate = 8000
    time = np.arange(rate) / rate
    tone = np.sin(2 * np.pi * 440 * time)
    audio = np.concatenate([0.1 * tone, 0.5 * tone]).astype(np.float32)
    metrics = measure_audio(audio, rate)
    assert abs(metrics["sample_peak_dbfs"] - (-6.02)) < 0.05
    assert 13 < metrics["window_level_range_db"] < 15
    assert abs(metrics["duration_seconds"] - 2) < 0.001


def test_report_cli_compares_files_as_json(tmp_path):
    paths = [tmp_path / "original.wav", tmp_path / "processed.wav"]
    for path, amplitude in zip(paths, [10000, 5000]):
        samples = (np.sin(2 * np.pi * 440 * np.arange(8000) / 8000) * amplitude).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(samples.tobytes())
    result = subprocess.run(
        [sys.executable, "-m", "adaptive_audio.report", str(paths[0]), str(paths[1]), "--json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["processed"]["sample_peak_dbfs"] < report["original"]["sample_peak_dbfs"]
    assert abs(report["peak_change_db"] - (-6.02)) < 0.1
