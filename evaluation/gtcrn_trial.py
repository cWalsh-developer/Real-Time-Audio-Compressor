"""Isolated GTCRN streaming-model quality trial; not the live extension."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import scipy
from scipy.signal import resample_poly

from adaptive_audio.wav import read_wav, write_wav


ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/gtcrn/stream/onnx_models/gtcrn_simple.onnx"
MODEL_SHA256 = "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87"
UPSTREAM_COMMIT = "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73"
FFT_SIZE, HOP, MODEL_RATE = 512, 256, 16000


def create_session() -> ort.InferenceSession:
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("GTCRN model hash mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])


def enhance(samples: np.ndarray, session: ort.InferenceSession) -> tuple[np.ndarray, list[float]]:
    """Use one ONNX call per 16 ms hop, with independent state per channel.

    File boundary reflection matches upstream PyTorch's centered STFT. This
    offline driver aligns output for comparison; a live resampler/OLA scheduler
    and its delay still need implementation and measurement.
    """
    channels = samples.shape[1]
    window = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FFT_SIZE) / FFT_SIZE))
    padded = np.pad(samples, ((HOP, HOP + (-len(samples)) % HOP), (0, 0)), mode="reflect")
    output = np.zeros_like(padded, dtype=np.float64)
    normalization = np.zeros(len(padded), dtype=np.float64)
    states = [{
        "conv_cache": np.zeros((2, 1, 16, 16, 33), np.float32),
        "tra_cache": np.zeros((2, 3, 1, 1, 16), np.float32),
        "inter_cache": np.zeros((2, 1, 33, 16), np.float32),
    } for _ in range(channels)]
    timings = []
    for start in range(0, len(padded) - FFT_SIZE + 1, HOP):
        began = time.perf_counter()
        for channel, state in enumerate(states):
            spectrum = np.fft.rfft(padded[start:start + FFT_SIZE, channel] * window)
            features = np.stack((spectrum.real, spectrum.imag), axis=-1).astype(np.float32)[None, :, None, :]
            enhanced, conv, tra, inter = session.run(None, {"mix": features, **state})
            state.update(conv_cache=conv, tra_cache=tra, inter_cache=inter)
            spectrum_out = enhanced[0, :, 0, 0] + 1j * enhanced[0, :, 0, 1]
            output[start:start + FFT_SIZE, channel] += np.fft.irfft(spectrum_out, n=FFT_SIZE) * window
        normalization[start:start + FFT_SIZE] += window ** 2
        timings.append(time.perf_counter() - began)
    output /= np.maximum(normalization[:, None], 1e-12)
    return output[HOP:HOP + len(samples)].astype(np.float32), timings


def level_db(samples: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(samples.astype(np.float64) ** 2) + 1e-20))


def verify_reference(session: ort.InferenceSession) -> dict:
    directory = ROOT / "models/gtcrn/stream/test_wavs"
    original, rate = read_wav(directory / "mix.wav")
    reference, reference_rate = read_wav(directory / "enh_stream.wav")
    if rate != MODEL_RATE or reference_rate != MODEL_RATE:
        raise ValueError("unexpected upstream reference sample rate")
    enhanced, _ = enhance(original.reshape(-1, 1), session)
    count = min(len(reference), len(enhanced))
    difference = enhanced[:count, 0] - reference[:count]
    report = {"reference_max_error": float(np.max(np.abs(difference))),
              "reference_rms_error": float(np.sqrt(np.mean(difference ** 2)))}
    if report["reference_max_error"] > 0.001:
        raise ValueError(f"upstream parity failed: {report}")
    return report


def verify_future_independence(session: ort.InferenceSession) -> dict:
    """Changing future audio must not change output before the STFT look-ahead."""
    rng = np.random.default_rng(0)
    original = rng.normal(0, 0.03, (MODEL_RATE * 2, 2)).astype(np.float32)
    changed = original.copy()
    changed[MODEL_RATE:] = rng.normal(0, 0.1, changed[MODEL_RATE:].shape)
    first, _ = enhance(original, session)
    second, _ = enhance(changed, session)
    # An overlap-added output sample can depend on the whole later window
    # covering it, not just the centered STFT's half-window. Bound that by
    # the full 512-sample analysis window (32 ms), excluding resampling.
    error = float(np.max(np.abs(first[:MODEL_RATE - FFT_SIZE] - second[:MODEL_RATE - FFT_SIZE])))
    if error > 1e-7:
        raise ValueError(f"future input changed earlier output: {error}")
    return {"prefix_max_error": error, "excluded_stft_lookahead_samples": FFT_SIZE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--reduction-db", type=float, default=6)
    args = parser.parse_args()
    if not np.isfinite(args.reduction_db) or not 0 <= args.reduction_db <= 30:
        parser.error("reduction must be between 0 and 30 dB")
    audio, rate = read_wav(args.input)
    if rate != 48000 or audio.ndim != 2 or audio.shape[1] != 2 or len(audio) < rate:
        parser.error("input must be stereo 48 kHz PCM16 WAV, at least one second")
    session = create_session()
    parity = verify_reference(session)
    future_independence = verify_future_independence(session)
    began = time.perf_counter()
    lowband = resample_poly(audio, 1, 3, axis=0)
    enhanced, timings = enhance(lowband, session)
    # Only remove the estimated low-band background. Original high frequencies
    # stay in the mix; GTCRN's 16 kHz model cannot identify them reliably.
    residual = resample_poly(lowband - enhanced, 3, 1, axis=0)[:len(audio)]
    gain = 10 ** (-args.reduction_db / 20)
    preview = audio + (gain - 1) * residual
    elapsed = time.perf_counter() - began
    peak = float(np.max(np.abs(preview)))
    if not np.all(np.isfinite(preview)) or peak >= 1:
        raise ValueError(f"preview would clip or contains invalid samples (peak {peak}); use quieter input")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_wav(args.output_dir / "reduced_preview.wav", preview, rate)
    write_wav(args.output_dir / "speech_estimate_16k.wav", enhanced, MODEL_RATE)
    report = {
        "model": "GTCRN DNS3 streaming ONNX", "upstream_commit": UPSTREAM_COMMIT,
        "model_sha256": MODEL_SHA256, "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source": str(args.input), "audio_seconds": len(audio) / rate,
        "reduction_db": args.reduction_db, "process_seconds": elapsed,
        "processing_time_per_audio_second": elapsed / (len(audio) / rate),
        "stereo_hop_ms_p50": float(np.percentile(timings, 50) * 1000),
        "stereo_hop_ms_p99": float(np.percentile(timings, 99) * 1000),
        "stereo_hop_ms_max": max(timings) * 1000,
        "hops_over_16ms": sum(value > 0.016 for value in timings),
        "output_level_change_db": level_db(preview) - level_db(audio),
        "peak": peak, "upstream_parity": parity,
        "future_independence": future_independence,
        "versions": {"numpy": np.__version__, "onnxruntime": ort.__version__, "scipy": scipy.__version__},
        "note": "Native CPU quality/throughput trial; resampling is offline. No browser latency claim.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
