"""Evaluate the optional spectral-mask baseline on held-out prepared windows."""

import json
from pathlib import Path

import numpy as np
import torch

from .training import HOP_LENGTH, N_FFT, SpectralMaskNet
from .training_data import _load_window, iter_batches, read_windows
from .wav import write_wav


def si_sdr(reference, estimate):
    """Return scale-invariant SDR in dB for one-dimensional signals."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape or reference.ndim != 1 or not reference.size:
        raise ValueError("reference and estimate must be non-empty one-dimensional arrays")
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    reference_energy = np.dot(reference, reference)
    if reference_energy <= 1e-20:
        raise ValueError("reference has no measurable energy")
    projection = np.dot(estimate, reference) / reference_energy * reference
    noise = estimate - projection
    return float(10 * np.log10(max(np.dot(projection, projection), 1e-20) /
                         max(np.dot(noise, noise), 1e-20)))


def _estimate(model, mixture, device):
    window = torch.hann_window(N_FFT, device=device)
    spectrum = torch.stft(mixture, N_FFT, HOP_LENGTH, N_FFT, window, return_complex=True)
    mask = model(spectrum.abs())
    estimate = torch.istft(mask * spectrum, N_FFT, HOP_LENGTH, N_FFT, window,
                           length=mixture.shape[-1])
    return estimate


def evaluate_checkpoint(checkpoint_path, root, *, split="test", batch_size=4,
                        device="auto", max_batches=None):
    """Evaluate a checkpoint and return aggregate held-out metrics."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=True)
    model = SpectralMaskNet().to(torch_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mixture_scores, estimate_scores, speech_losses = [], [], []
    silent_windows = 0
    with torch.inference_mode():
        for batch_number, batch in enumerate(iter_batches(root, split, batch_size, shuffle=False)):
            mixture = torch.from_numpy(batch["mixture"]).to(torch_device)
            speech = torch.from_numpy(batch["speech"]).to(torch_device)
            estimate = _estimate(model, mixture, torch_device).cpu().numpy()
            for index in range(len(estimate)):
                speech_energy = np.mean(batch["speech"][index].astype(np.float64) ** 2)
                if speech_energy <= 1e-20:
                    silent_windows += 1
                else:
                    mixture_scores.append(si_sdr(batch["speech"][index], batch["mixture"][index]))
                    estimate_scores.append(si_sdr(batch["speech"][index], estimate[index]))
                speech_losses.append(float(np.mean(np.abs(estimate[index] - batch["speech"][index]))))
            if max_batches is not None and batch_number + 1 >= max_batches:
                break
    report = {
        "checkpoint": str(Path(checkpoint_path)),
        "split": split,
        "windows": len(speech_losses),
        "silent_speech_windows": silent_windows,
        "estimate_mean_absolute_error": float(np.mean(speech_losses)),
    }
    if estimate_scores:
        report.update({
            "scored_windows": len(estimate_scores),
            "mixture_si_sdr_db": float(np.mean(mixture_scores)),
            "estimate_si_sdr_db": float(np.mean(estimate_scores)),
            "si_sdr_improvement_db": float(np.mean(estimate_scores) - np.mean(mixture_scores)),
        })
    return report


def write_evaluation_report(checkpoint_path, root, output, **options):
    report = evaluate_checkpoint(checkpoint_path, root, **options)
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def render_preview(checkpoint_path, root, output_dir, *, split="test", index=0,
                   background_gain=0.35, device="auto"):
    """Render one indexed window and a conservative estimated-background remix."""
    if not 0 <= background_gain <= 1:
        raise ValueError("background gain must be between 0 and 1")
    windows = read_windows(root, split)
    if type(index) is not int or not 0 <= index < len(windows):
        raise ValueError(f"preview index must be between 0 and {len(windows) - 1}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=True)
    model = SpectralMaskNet().to(torch_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    window = _load_window(root, windows[index])
    mixture = torch.from_numpy(window["mixture"])[None].to(torch_device)
    with torch.inference_mode():
        estimate = _estimate(model, mixture, torch_device)[0].cpu().numpy()
    background = window["mixture"] - estimate
    reduced = estimate + background_gain * background
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = windows[index]["sample_rate"]
    outputs = {"mixture": window["mixture"], "speech_reference": window["speech"],
               "speech_estimate": estimate, "background_reduced": reduced}
    for name, audio in outputs.items():
        write_wav(output_dir / f"{split}-{index:04d}-{name}.wav", audio, sample_rate)
    report = {
        "checkpoint": str(Path(checkpoint_path)), "split": split, "index": index,
        "background_gain": background_gain, "sample_rate": sample_rate,
        "frames": len(window["mixture"]),
        "files": {name: f"{split}-{index:04d}-{name}.wav" for name in outputs},
        "metrics": {name: _preview_metrics(audio, sample_rate) for name, audio in outputs.items()},
    }
    (output_dir / f"{split}-{index:04d}-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _preview_metrics(audio, sample_rate):
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return {"peak": peak, "rms_dbfs": float(20 * np.log10(max(rms, 1e-8))),
            "clipped_samples": int(np.count_nonzero(np.abs(audio) >= 1))}