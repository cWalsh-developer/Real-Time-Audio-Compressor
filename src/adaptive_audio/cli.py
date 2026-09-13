"""Command-line entry point for the runnable WAV baseline."""

import argparse
from pathlib import Path

from .processing import MODES
from .streaming import process_wav
from .video import process_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adaptive audio WAV and MOV processor")
    parser.add_argument("input", type=Path, help="16-bit PCM WAV or MOV file")
    parser.add_argument("--mode", choices=MODES, default="balanced")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    if args.input.suffix.lower() not in (".wav", ".mov"):
        parser.error("Input must be a 16-bit PCM WAV or MOV file")
    output = args.output or args.input.with_name(f"{args.input.stem}_{args.mode}.wav")
    if output.resolve() == args.input.resolve():
        parser.error("Output must differ from input")
    try:
        if args.input.suffix.lower() == ".mov":
            process_video(args.input, output, args.mode)
        else:
            process_wav(args.input, output, args.mode)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {output}")
    return 0
