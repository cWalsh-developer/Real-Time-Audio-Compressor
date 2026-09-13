"""Run the optional YAMNet adapter on a WAV file."""

import argparse
from pathlib import Path

from .yamnet import classify_wav, download_model, write_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify WAV audio with YAMNet")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("models/yamnet"))
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    output = args.output or args.input.with_suffix(".labels.json")
    try:
        download_model(args.model_dir)
        scores = classify_wav(args.input, args.model_dir)
        write_scores(output, scores)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote {len(scores)} scored windows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
