"""Command-line entry point for the runnable WAV baseline."""

import argparse
from pathlib import Path

from .processing import MODES
from .streaming import process_wav
from .video import process_video
from .classification import read_scores
from .yamnet import classify_wav, download_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adaptive audio WAV and MOV processor")
    parser.add_argument("input", type=Path, help="16-bit PCM WAV or MOV file")
    parser.add_argument("--mode", choices=MODES, default="balanced")
    parser.add_argument("--output", type=Path)
    semantic = parser.add_mutually_exclusive_group()
    semantic.add_argument("--labels", type=Path, help="JSON file of time-stamped semantic scores")
    semantic.add_argument("--ai", action="store_true", help="classify audio with the optional YAMNet model")
    parser.add_argument("--model-dir", type=Path, default=Path("models/yamnet"))
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    if args.input.suffix.lower() not in (".wav", ".mov"):
        parser.error("Input must be a 16-bit PCM WAV or MOV file")
    output = args.output or args.input.with_name(f"{args.input.stem}_{args.mode}.wav")
    if output.resolve() == args.input.resolve():
        parser.error("Output must differ from input")
    try:
        labels = read_scores(args.labels) if args.labels else None
        if args.input.suffix.lower() == ".mov":
            process_video(args.input, output, args.mode, labels=labels,
                          use_ai=args.ai, model_directory=args.model_dir)
        else:
            if args.ai:
                download_model(args.model_dir)
                labels = classify_wav(args.input, args.model_dir)
            process_wav(args.input, output, args.mode, labels=labels)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Wrote {output}")
    return 0
