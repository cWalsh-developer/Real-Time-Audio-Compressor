"""Command-line entry point for the runnable WAV baseline."""

import argparse
from pathlib import Path

from .processing import MODES
from .streaming import process_wav


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adaptive audio WAV baseline")
    parser.add_argument("input", type=Path, help="16-bit PCM WAV file")
    parser.add_argument("--mode", choices=MODES, default="balanced")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    if args.input.suffix.lower() != ".wav":
        parser.error("This baseline accepts 16-bit PCM WAV only; convert video audio with FFmpeg first")
    output = args.output or args.input.with_name(f"{args.input.stem}_{args.mode}.wav")
    if output.resolve() == args.input.resolve():
        parser.error("Output must differ from input")
    try:
        process_wav(args.input, output, args.mode)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {output}")
    return 0
