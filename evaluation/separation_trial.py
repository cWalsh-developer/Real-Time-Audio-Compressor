"""Offline BandIt v2 listening trial; outputs are deliberately outside Git."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from adaptive_audio.stems import remix_estimated_stems
from adaptive_audio.wav import read_wav, write_wav


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="48 kHz 16-bit PCM WAV")
    parser.add_argument("output_dir", type=Path, help="folder for stems and audition mix")
    parser.add_argument("--start", type=float, default=0, help="start time in seconds")
    parser.add_argument("--duration", type=float, default=12, help="trial duration in seconds")
    parser.add_argument("--reduction-db", type=float, default=6, help="fixed music/effects reduction for audition")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.start < 0 or args.duration <= 0 or args.reduction_db < 0:
        parser.error("start and reduction must be nonnegative; duration must be positive")

    audio, sample_rate = read_wav(args.input)
    if sample_rate != 48000:
        parser.error("BandIt v2 Multi requires 48 kHz audio; resample the WAV before this trial")
    start = round(args.start * sample_rate)
    end = min(len(audio), start + round(args.duration * sample_rate))
    if start >= end:
        parser.error("selected interval is outside the input")
    original = audio[start:end]
    if original.ndim == 1:
        original = original[:, None]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from bandit_infer import BanditSession  # optional, large research dependency

    load_start = time.perf_counter()
    with BanditSession("v2-multi", device=args.device, weights_dir=Path("models/bandit-weights")) as session:
        load_seconds = time.perf_counter() - load_start
        infer_start = time.perf_counter()
        stems = session.infer(original.T, sample_rate=sample_rate)
        infer_seconds = time.perf_counter() - infer_start

    speech, music, effects = (stems[name].T for name in ("speech", "music", "effects"))
    gain = 10 ** (-args.reduction_db / 20)
    preview = remix_estimated_stems(original, music, effects, music_gain=gain, effects_gain=gain)
    for name, signal in (
        ("original", original),
        ("speech_estimate", speech),
        ("music_estimate", music),
        ("effects_estimate", effects),
        ("reduced_preview", preview),
    ):
        write_wav(args.output_dir / f"{name}.wav", signal, sample_rate)

    seconds = len(original) / sample_rate
    report = {
        "model": "v2-multi",
        "device": args.device,
        "source": str(args.input),
        "start_seconds": args.start,
        "audio_seconds": seconds,
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(infer_seconds, 3),
        "inference_time_per_audio_second": round(infer_seconds / seconds, 3),
        "preview_reduction_db": args.reduction_db,
        "preview_peak_before_wav_clipping": float(np.max(np.abs(preview))),
        "note": "Fixed-gain audition only. Inference throughput is not live end-to-end latency.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
