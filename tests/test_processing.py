import numpy as np

from adaptive_audio.processing import process_audio


def test_balanced_reduces_loud_section_but_preserves_contrast():
    rate = 8000
    time = np.arange(rate) / rate
    tone = np.sin(2 * np.pi * 440 * time)
    audio = np.concatenate([0.02 * tone, 0.5 * tone]).astype(np.float32)
    result = process_audio(audio, rate, "balanced")
    quiet = np.sqrt(np.mean(result[2000:6000] ** 2))
    loud = np.sqrt(np.mean(result[10000:14000] ** 2))
    assert loud / quiet < 25
    assert loud / quiet > 2
    assert np.max(np.abs(result)) <= 1


def test_night_constrains_more_than_cinema():
    rate = 8000
    time = np.arange(rate) / rate
    tone = np.sin(2 * np.pi * 440 * time)
    audio = np.concatenate([0.02 * tone, 0.5 * tone]).astype(np.float32)
    cinema = process_audio(audio, rate, "cinema")
    night = process_audio(audio, rate, "night")
    def contrast(x):
        return np.sqrt(np.mean(x[10000:14000] ** 2)) / np.sqrt(np.mean(x[2000:6000] ** 2))
    assert contrast(night) < contrast(cinema)


def test_stereo_channels_keep_same_gain_and_peak_ceiling():
    audio = np.ones((8000, 2), dtype=np.float32)
    audio[:, 1] *= 0.5
    result = process_audio(audio, 8000, "night")
    assert np.max(np.abs(result)) <= 10 ** (-1 / 20) + 1e-6
    np.testing.assert_allclose(result[:, 1], result[:, 0] * 0.5)


def test_isolated_peak_does_not_turn_down_distant_quiet_audio():
    rate = 8000
    audio = np.full(rate * 3, 0.01, dtype=np.float32)
    audio[-100] = 1.0
    result = process_audio(audio, rate, "night")
    assert np.mean(np.abs(result[1000:4000])) > 0.02
    assert np.max(np.abs(result)) <= 10 ** (-1 / 20) + 1e-6
