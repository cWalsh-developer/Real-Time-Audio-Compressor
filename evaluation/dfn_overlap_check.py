"""Make a controlled speech/music overlap and measure a rendered preview.

Run ``prepare`` first, render music.wav and mixture.wav with
dfn_browser_probe.py --preview-only --output-dir, then run ``analyze``.
The files are local test audio and stay ignored by Git.
"""

import argparse
from pathlib import Path

import numpy as np

from adaptive_audio.wav import read_wav, write_wav


def rms_db(samples: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(samples.astype(np.float64) ** 2)) + 1e-12))


def prepare(speech_path: Path, music_path: Path, output_dir: Path) -> None:
    speech, speech_rate = read_wav(speech_path)
    music, music_rate = read_wav(music_path)
    if speech_rate != 48000 or music_rate != 48000:
        raise ValueError("both inputs must be 48 kHz")
    if speech.ndim != 2 or music.ndim != 2 or speech.shape[1] != 2 or music.shape[1] != 2:
        raise ValueError("both inputs must be stereo")
    if len(music) < len(speech):
        raise ValueError("music must be at least as long as speech")
    music = music[:len(speech)] * 0.6
    speech = speech * 0.6
    mixture = speech + music
    if np.max(np.abs(mixture)) >= 1:
        raise ValueError("mixture would clip; choose quieter inputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, samples in (("speech", speech), ("music", music), ("mixture", mixture)):
        write_wav(output_dir / f"{name}.wav", samples, 48000)
    print(f"speech {rms_db(speech):.2f} dBFS, music {rms_db(music):.2f} dBFS")


def analyze(output_dir: Path, delay_samples: int) -> None:
    def load(name: str) -> np.ndarray:
        samples, rate = read_wav(output_dir / name)
        if rate != 48000 or samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError(f"expected stereo 48 kHz WAV: {name}")
        return samples.astype(np.float64)

    speech = load("speech.wav")
    music = load("music.wav")
    mixture = load("mixture.wav")
    if not (speech.shape == music.shape == mixture.shape):
        raise ValueError("input files have different shapes")
    if np.max(np.abs(mixture - (speech + music))) > 2 / 32768:
        raise ValueError("mixture is not the sum of the two sources")
    music_preview = load(f"{output_dir.name}_music_residual_preview.wav")
    mixture_preview = load(f"{output_dir.name}_mixture_residual_preview.wav")
    if music_preview.shape != speech.shape or mixture_preview.shape != speech.shape:
        raise ValueError("preview files have different shapes")
    if delay_samples < 0 or delay_samples >= len(speech) - 9600:
        raise ValueError("invalid delay for audio length")
    reference = speech[:len(speech) - delay_samples]
    contribution = (mixture_preview - music_preview)[delay_samples:]
    # Exclude startup and tail where the model's state and delayed samples differ.
    reference = reference[4800:-4800].ravel()
    contribution = contribution[4800:-4800].ravel()
    change_db = rms_db(contribution) - rms_db(reference)
    correlation = float(np.corrcoef(reference, contribution)[0, 1])
    error_db = rms_db(contribution - reference) - rms_db(reference)
    print(f"speech contribution change: {change_db:+.2f} dB")
    print(f"speech contribution correlation: {correlation:.4f}")
    print(f"relative difference: {error_db:+.2f} dB")
    print("This is a counterfactual speech contribution, not an isolated output stem.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    make = subparsers.add_parser("prepare")
    make.add_argument("speech", type=Path)
    make.add_argument("music", type=Path)
    make.add_argument("output_dir", type=Path)
    check = subparsers.add_parser("analyze")
    check.add_argument("output_dir", type=Path)
    check.add_argument("--delay-samples", type=int, default=1888)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.speech, args.music, args.output_dir)
    else:
        analyze(args.output_dir, args.delay_samples)


if __name__ == "__main__":
    main()
