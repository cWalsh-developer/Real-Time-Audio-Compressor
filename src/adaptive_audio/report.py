"""Compare original and processed WAV measurements."""

import argparse
import json
from pathlib import Path

from .evaluation import measure_audio
from .wav import read_wav


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare 100 ms RMS-window audio levels")
    parser.add_argument("original", type=Path)
    parser.add_argument("processed", type=Path)
    parser.add_argument("--json", action="store_true", help="print machine-readable report")
    args = parser.parse_args(argv)
    try:
        original, original_rate = read_wav(args.original)
        processed, processed_rate = read_wav(args.processed)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if original_rate != processed_rate or original.shape != processed.shape:
        parser.error("Files must have the same sample rate, channel count, and duration")
    first = measure_audio(original, original_rate)
    second = measure_audio(processed, processed_rate)
    report = {
        "original": first,
        "processed": second,
        "peak_change_db": round(second["sample_peak_dbfs"] - first["sample_peak_dbfs"], 2),
        "window_range_change_db": round(second["window_level_range_db"] - first["window_level_range_db"], 2),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Metric                         Original  Processed")
        for key in first:
            print(f"{key:30} {first[key]:8.2f}  {second[key]:9.2f}")
        print("All levels are dBFS; range uses 100 ms RMS-window P90-P10, not LUFS LRA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
