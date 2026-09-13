"""Remix estimated cinematic stems while retaining the source mix as reference."""

import numpy as np


def remix_estimated_stems(
    original: np.ndarray,
    music: np.ndarray,
    effects: np.ndarray,
    *,
    music_gain: float = 1.0,
    effects_gain: float = 1.0,
) -> np.ndarray:
    """Reduce only estimated non-dialogue content; unity gains return input exactly.

    All arrays have the same shape: samples or samples x channels. Stem leakage
    can still reduce dialogue, so this is a remix primitive, not a quality claim.
    """
    if original.shape != music.shape or original.shape != effects.shape:
        raise ValueError("original, music, and effects must have the same shape")
    if original.ndim not in (1, 2):
        raise ValueError("audio must be mono or samples x channels")
    if not (0 <= music_gain <= 1 and 0 <= effects_gain <= 1):
        raise ValueError("stem gains must be between 0 and 1")
    if music_gain == 1 and effects_gain == 1:
        return original.copy()
    return original + (music_gain - 1) * music + (effects_gain - 1) * effects
