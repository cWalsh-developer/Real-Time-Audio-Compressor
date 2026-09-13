import subprocess
import sys
import wave

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
