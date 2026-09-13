import numpy as np
import pytest
import subprocess
import sys

from adaptive_audio.processing import process_audio
from adaptive_audio.rolling import RollingProcessor
from adaptive_audio.wav import read_wav, write_wav


def test_rolling_processor_waits_for_lookahead_and_bounds_buffer():
    processor = RollingProcessor(sample_rate=8000, mode="balanced", lookahead_seconds=1.0)
    block = np.full(800, 0.02, dtype=np.float32)
    for _ in range(10):
        assert processor.push(block) == []
        assert processor.buffered_frames <= 10
    first = processor.push(block)
    assert len(first) == 1
    assert len(first[0]) == 800
    assert processor.buffered_frames == 10
    for _ in range(100):
        assert len(processor.push(block)) == 1
        assert processor.buffered_frames == 10
    assert sum(len(frame) for frame in processor.flush()) == 8000


def test_rolling_output_matches_offline_with_full_and_partial_frames():
    rate = 8000
    time = np.arange(rate * 4 + 113) / rate
    level = np.where(time < 1, 0.02, np.where(time < 2, 0.4, 0.08))
    mono = (level * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    audio = np.column_stack([mono, mono * 0.6])
    processor = RollingProcessor(sample_rate=rate, mode="balanced", lookahead_seconds=1.0)
    chunks = []
    for start in range(0, len(audio), 800):
        chunks.extend(processor.push(audio[start:start + 800]))
    chunks.extend(processor.flush())
    actual = np.concatenate(chunks)
    expected = process_audio(audio, rate, "balanced")
    assert actual.shape == audio.shape
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert np.max(np.abs(actual)) <= 10 ** (-1 / 20) + 1e-6


def test_rolling_rejects_insufficient_lookahead():
    with pytest.raises(ValueError, match="at least"):
        RollingProcessor(sample_rate=8000, mode="balanced", lookahead_seconds=0.2)


def test_rolling_wav_command_matches_offline_samples(tmp_path):
    rate = 8000
    time = np.arange(rate * 3 + 77) / rate
    audio = (np.where(time < 1.5, 0.03, 0.4) * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    source = tmp_path / "source.wav"
    target = tmp_path / "rolling.wav"
    write_wav(source, audio, rate)
    result = subprocess.run([sys.executable, "-m", "adaptive_audio.rolling_cli", str(source),
                             "--mode", "balanced", "--lookahead", "1", "--output", str(target)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    decoded, _ = read_wav(source)
    rolled, _ = read_wav(target)
    np.testing.assert_allclose(rolled, process_audio(decoded, rate, "balanced"), atol=2 / 32768)
    assert "1.00 s" in result.stdout
