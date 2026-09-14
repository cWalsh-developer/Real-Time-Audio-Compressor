"""Evaluate the optional spectral-mask baseline on held-out prepared windows."""

import json
from pathlib import Path

import numpy as np
import torch

from .training import HOP_LENGTH, N_FFT, SpectralMaskNet
from .training_data import iter_batches


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