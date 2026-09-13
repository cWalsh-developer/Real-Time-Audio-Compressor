"""Conservative semantic adjustment to the deterministic gain plan."""

import numpy as np

from .classification import EventScores


def semantic_adjustments(labels: list[EventScores], frame_count: int, sample_rate: int) -> np.ndarray:
    """Return a dB bias for each 100 ms analysis frame."""
    windows = sorted(labels, key=lambda item: item.start_seconds)
    frame_seconds = max(1, round(sample_rate * 0.1)) / sample_rate
    adjustment = np.zeros(frame_count)
    first = 0
    for index in range(frame_count):
        midpoint = (index + 0.5) * frame_seconds
        while first < len(windows) and windows[first].end_seconds <= midpoint:
            first += 1
        covering = []
        for window in windows[first:]:
            if window.start_seconds > midpoint:
                break
            if midpoint < window.end_seconds:
                covering.append(window)
        if covering:
            speech = np.mean([item.scores["speech"] for item in covering])
            action = np.mean([item.scores["action"] for item in covering])
            adjustment[index] = 3.0 * speech - 2.0 * action
    return adjustment
