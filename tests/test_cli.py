import subprocess
import sys
import wave
import json

import numpy as np


def test_cli_writes_a_playable_wav(tmp_path):
    source = tmp_path / "sample.wav"
    target = tmp_path / "processed.wav"
    samples = (np.sin(2 * np.pi * 440 * np.arange(8000) / 8000) * 10000).astype("<i2")
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(samples.tobytes())
    result = subprocess.run(
        [sys.executable, "-m", "adaptive_audio", str(source), "--mode", "night", "--output", str(target)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    with wave.open(str(target), "rb") as wav:
        assert wav.getnframes() == 8000
        assert wav.getframerate() == 8000
        assert wav.getnchannels() == 1


def test_cli_can_apply_saved_semantic_scores(tmp_path):
    source = tmp_path / "sample.wav"
    baseline = tmp_path / "baseline.wav"
    informed = tmp_path / "informed.wav"
    labels = tmp_path / "labels.json"
    samples = (np.sin(2 * np.pi * 440 * np.arange(8000) / 8000) * 1000).astype("<i2")
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(samples.tobytes())
    labels.write_text(json.dumps({"segments": [{"start_seconds": 0, "end_seconds": 1,
        "scores": {"speech": 1, "music": 0, "action": 0, "other": 0}}]}))
    for output, extra in [(baseline, []), (informed, ["--labels", str(labels)])]:
        result = subprocess.run([sys.executable, "-m", "adaptive_audio", str(source),
                                 "--output", str(output), *extra], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    assert informed.read_bytes() != baseline.read_bytes()
