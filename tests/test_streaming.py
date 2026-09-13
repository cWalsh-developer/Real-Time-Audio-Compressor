import wave

import numpy as np

from adaptive_audio.processing import process_audio
from adaptive_audio.streaming import process_wav
from adaptive_audio.wav import read_wav, write_wav


def test_streamed_output_matches_in_memory_across_chunk_boundaries(tmp_path):
    rate = 8000
    time = np.arange(rate * 3 + 113) / rate
    envelope = np.where(time < 1, 0.02, np.where(time < 2, 0.4, 0.06))
    mono = (envelope * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    audio = np.column_stack([mono, mono * 0.6])
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    write_wav(source, audio, rate)
    decoded, _ = read_wav(source)
    process_wav(source, target, "balanced", chunk_frames=913)
    streamed, output_rate = read_wav(target)
    expected = process_audio(decoded, rate, "balanced")
    assert output_rate == rate
    assert streamed.shape == expected.shape
    np.testing.assert_allclose(streamed, expected, atol=2 / 32768)
    assert np.max(np.abs(streamed)) <= 10 ** (-1 / 20) + 2 / 32768


def test_streaming_preserves_sample_count_for_partial_final_frame(tmp_path):
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    write_wav(source, np.zeros(1234, dtype=np.float32), 8000)
    process_wav(source, target, "night", chunk_frames=317)
    with wave.open(str(target), "rb") as wav:
        assert wav.getnframes() == 1234
