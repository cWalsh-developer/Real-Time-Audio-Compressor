"""Bounded-latency, block-by-block look-ahead audio processing."""

import math
from collections import deque

import numpy as np

from .processing import MODES, apply_gain, frame_measurements, plan_gain


class RollingProcessor:
    """Accept 100 ms PCM blocks and emit processed blocks after look-ahead.

    Keep five past measurement frames for smoothing and at most the configured
    number of future audio frames. The caller owns pacing and playback.
    """

    def __init__(self, sample_rate: int, mode: str = "balanced", lookahead_seconds: float = 2.0):
        if sample_rate <= 0 or mode not in MODES:
            raise ValueError("Expected a positive sample rate and known mode")
        self.sample_rate = sample_rate
        self.mode = mode
        self.frame_samples = max(1, round(sample_rate * 0.1))
        self.frame_seconds = self.frame_samples / sample_rate
        self.lookahead_frames = math.ceil(lookahead_seconds / self.frame_seconds)
        if self.lookahead_frames < 6:
            raise ValueError("Look-ahead must be at least six analysis frames (about 0.6 s)")
        self.latency_seconds = self.lookahead_frames * self.frame_seconds
        self._pending = deque()
        self._history = deque(maxlen=5)
        self._previous_gain = None
        self._channels = None
        self._partial_seen = False
        self._flushed = False

    @property
    def buffered_frames(self) -> int:
        return len(self._pending)

    def push(self, audio: np.ndarray) -> list[np.ndarray]:
        if self._flushed or self._partial_seen:
            raise ValueError("Cannot push after a final partial frame or flush")
        if audio.ndim not in (1, 2) or not 0 < len(audio) <= self.frame_samples:
            raise ValueError("Expected one non-empty 100 ms mono or multichannel block")
        channels = 1 if audio.ndim == 1 else audio.shape[1]
        if self._channels is None:
            self._channels = channels
        if channels != self._channels or not np.all(np.isfinite(audio)):
            raise ValueError("Channel count must stay fixed and samples must be finite")
        if len(audio) < self.frame_samples:
            self._partial_seen = True
        samples = np.array(audio, dtype=np.float32, copy=True)
        level, peak = frame_measurements(samples)
        self._pending.append((samples, level, peak))
        if len(self._pending) > self.lookahead_frames:
            return [self._emit_one()]
        return []

    def flush(self) -> list[np.ndarray]:
        if self._flushed:
            raise ValueError("Already flushed")
        self._flushed = True
        return [self._emit_one() for _ in range(len(self._pending))]

    def _emit_one(self) -> np.ndarray:
        measured = list(self._history) + [(level, peak) for _, level, peak in self._pending]
        levels = np.asarray([item[0] for item in measured])
        peaks = np.asarray([item[1] for item in measured])
        gains = plan_gain(levels, peaks, self.mode)
        current_index = len(self._history)
        current = gains[current_index]
        following = gains[current_index + 1] if len(self._pending) > 1 else current
        previous = self._previous_gain if self._previous_gain is not None else current
        samples, level, peak = self._pending.popleft()
        processed = apply_gain(samples, self.sample_rate, np.asarray([previous, current, following]),
                               start=self.frame_samples)
        self._history.append((level, peak))
        self._previous_gain = current
        return processed
