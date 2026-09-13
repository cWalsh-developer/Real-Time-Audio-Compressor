"""Model-independent semantic event scores and manual test labels."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


LABELS = ("speech", "music", "action", "other")
OTHER = {"speech": 0.0, "music": 0.0, "action": 0.0, "other": 1.0}


@dataclass(frozen=True)
class EventScores:
    start_seconds: float
    end_seconds: float
    scores: dict[str, float]

    def __post_init__(self) -> None:
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("Event interval must have positive duration")
        if set(self.scores) != set(LABELS):
            raise ValueError("Scores must contain speech, music, action, and other")
        if any(value < 0 or value > 1 for value in self.scores.values()):
            raise ValueError("Scores must be probabilities between 0 and 1")
        if abs(sum(self.scores.values()) - 1) > 1e-6:
            raise ValueError("Scores must sum to 1")


class AudioClassifier(Protocol):
    def classify(self, audio: np.ndarray, sample_rate: int) -> list[EventScores]: ...


def read_scores(path: Path) -> list[EventScores]:
    """Read score windows; model windows may overlap."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EventScores(**item) for item in data["segments"]]


class ManualClassifier:
    """A deterministic timeline for validating downstream decisions."""

    def __init__(self, segments: list[EventScores]):
        self.segments = sorted(segments, key=lambda item: item.start_seconds)
        for previous, current in zip(self.segments, self.segments[1:]):
            if current.start_seconds < previous.end_seconds:
                raise ValueError("Manual segments must not overlap")

    @classmethod
    def from_json(cls, path: Path) -> "ManualClassifier":
        return cls(read_scores(path))

    def classify_duration(self, duration_seconds: float, window_seconds: float = 1.0) -> list[EventScores]:
        if duration_seconds < 0 or window_seconds <= 0:
            raise ValueError("Duration must be nonnegative and window must be positive")
        result = []
        start = 0.0
        while start < duration_seconds:
            end = min(start + window_seconds, duration_seconds)
            midpoint = (start + end) / 2
            match = next((item for item in self.segments if item.start_seconds <= midpoint < item.end_seconds), None)
            result.append(EventScores(start, end, dict(match.scores if match else OTHER)))
            start = end
        return result

    def classify(self, audio: np.ndarray, sample_rate: int) -> list[EventScores]:
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        return self.classify_duration(len(audio) / sample_rate)
