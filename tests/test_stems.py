import numpy as np
import pytest

from adaptive_audio.stems import remix_estimated_stems


def test_zero_reduction_is_exact_original_even_with_imperfect_estimates():
    original = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)
    wrong_estimate = np.full_like(original, 0.7)

    result = remix_estimated_stems(original, wrong_estimate, wrong_estimate)

    np.testing.assert_array_equal(result, original)


def test_only_estimated_music_and_effects_are_reduced():
    speech = np.array([[0.2, 0.1], [0.15, 0.25]], dtype=np.float32)
    music = np.array([[0.3, 0.2], [0.4, 0.1]], dtype=np.float32)
    effects = np.array([[0.1, 0.3], [0.2, 0.1]], dtype=np.float32)
    original = speech + music + effects

    result = remix_estimated_stems(original, music, effects, music_gain=0.5, effects_gain=0.25)

    np.testing.assert_allclose(result, speech + 0.5 * music + 0.25 * effects, atol=1e-7)


def test_invalid_shape_or_gain_is_rejected():
    original = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        remix_estimated_stems(original, np.zeros(10), original)
    with pytest.raises(ValueError, match="between 0 and 1"):
        remix_estimated_stems(original, original, original, music_gain=1.1)
