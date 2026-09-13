"""Compare audio levels for model-labeled event windows in a local WAV."""

import argparse
import json
import wave
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, help="16-bit PCM WAV")
    parser.add_argument("labels", type=Path, help="Classifier scores JSON")
    parser.add_argument("--confidence", type=float, default=0.6)
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")

    segments = json.loads(args.labels.read_text(encoding="utf-8"))["segments"]
    levels: dict[str, list[float]] = {name: [] for name in ("speech", "music", "action", "other")}
    with wave.open(str(args.wav), "rb") as source:
        if source.getsampwidth() != 2:
            parser.error("WAV must contain 16-bit PCM samples")
        rate = source.getframerate()
        channels = source.getnchannels()
        for segment in segments:
            category = max(segment["scores"], key=segment["scores"].get)
            if segment["scores"][category] < args.confidence or category not in levels:
                continue
            start = max(0, round(segment["start_seconds"] * rate))
            end = min(source.getnframes(), round(segment["end_seconds"] * rate))
            if end <= start:
                continue
            source.setpos(start)
            samples = np.frombuffer(source.readframes(end - start), dtype="<i2")
            if len(samples) != (end - start) * channels:
                continue
            normalized = samples.astype(np.float64) / 32768
            rms = np.sqrt(np.mean(normalized * normalized))
            levels[category].append(20 * np.log10(max(rms, 1e-8)))

    print("Category  Count   P10    P50    P90    P99   >-12dB  >-10dB")
    for category, measured in levels.items():
        if not measured:
            continue
        values = np.asarray(measured)
        p10, p50, p90, p99 = np.percentile(values, [10, 50, 90, 99])
        above12 = 100 * np.mean(values > -12)
        above10 = 100 * np.mean(values > -10)
        print(f"{category:<8} {len(values):5d} {p10:6.1f} {p50:6.1f} {p90:6.1f} "
              f"{p99:6.1f} {above12:7.1f}% {above10:7.1f}%")


if __name__ == "__main__":
    main()
