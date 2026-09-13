"""WAV-file driver for the bounded-look-ahead processor."""

import argparse
import time
import wave
from pathlib import Path

import numpy as np

from .processing import MODES
from .rolling import RollingProcessor


def process_rolling_wav(source: Path, target: Path, mode: str, lookahead_seconds: float) -> tuple[float, float, float]:
    """Feed one 100 ms input block at a time; return duration, elapsed, latency."""
    if source.resolve() == target.resolve():
        raise ValueError("Output must differ from input")
    start_time = time.perf_counter()
    with wave.open(str(source), "rb") as wav:
        if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2:
            raise ValueError("Only uncompressed 16-bit PCM WAV is supported")
        channels = wav.getnchannels()
        rate = wav.getframerate()
        duration = wav.getnframes() / rate
        processor = RollingProcessor(rate, mode, lookahead_seconds)
        with wave.open(str(target), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(rate)

            def write_blocks(blocks: list[np.ndarray]) -> None:
                for block in blocks:
                    pcm = np.round(np.clip(block, -1, 32767 / 32768) * 32768).astype("<i2")
                    output.writeframesraw(pcm.tobytes())

            while raw := wav.readframes(processor.frame_samples):
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels) / 32768.0
                block = samples[:, 0] if channels == 1 else samples
                write_blocks(processor.push(block))
            write_blocks(processor.flush())
    elapsed = time.perf_counter() - start_time
    return duration, elapsed, processor.latency_seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate a rolling look-ahead audio stream from a WAV file")
    parser.add_argument("input", type=Path)
    parser.add_argument("--mode", choices=MODES, default="balanced")
    parser.add_argument("--lookahead", type=float, default=2.0, help="buffer duration in seconds (default: 2)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    output = args.output or args.input.with_name(f"{args.input.stem}_{args.mode}_rolling.wav")
    try:
        duration, elapsed, latency = process_rolling_wav(args.input, output, args.mode, args.lookahead)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    factor = duration / elapsed if elapsed else float("inf")
    print(f"Wrote {output} | lookahead {latency:.2f} s | audio {duration:.2f} s | "
          f"processing {elapsed:.2f} s | speed {factor:.1f}x real time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
