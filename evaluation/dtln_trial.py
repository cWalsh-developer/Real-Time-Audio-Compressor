"""Pinned, isolated DTLN quality trial; this is not the live Chrome processor.

Streaming algorithm follows Nils L. Westhausen's MIT-licensed DTLN example:
https://github.com/breizhn/DTLN/blob/1de1f15a8b5b7e1c44905618ff2ef70ca8277fbc/real_time_processing_onnx.py
See evaluation/licenses/DTLN.txt for the upstream copyright and MIT licence.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from adaptive_audio.wav import read_wav, write_wav


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/dtln"
COMMIT = "1de1f15a8b5b7e1c44905618ff2ef70ca8277fbc"
ASSETS = {
    "model_1.onnx": ("pretrained_model/model_1.onnx", "22b91cae3855e5a0620e66a917ca6c82c58db0e842c770f58d86751c5e8d4ae3"),
    "model_2.onnx": ("pretrained_model/model_2.onnx", "e20c92f9233fccf29cddf86970d0d0161a03aebccc26d6f4d5639c4d5ec2e639"),
    "upstream_reference.py": ("real_time_processing_onnx.py", "e15f0fc092f03f372dff34770a9a7d1f4b664337f514bf6f0880c0b87f3a2016"),
    "LICENSE": ("LICENSE", "aa95acd8c8a7341bfcdb2823694dcbb27a2a4143e860bb52db1ff0f29c85e5a1"),
}
BLOCK, HOP, RATE = 512, 128, 16000
# Position offset in the upstream file driver, not end-to-end playback latency.
# The next 128-sample hop must also be collected: 384 + 128 = 512 (32 ms).
ALIGNMENT = BLOCK - HOP


def fetch_assets() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, (remote, digest) in ASSETS.items():
        target = MODEL_DIR / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            continue
        url = f"https://raw.githubusercontent.com/breizhn/DTLN/{COMMIT}/{remote}"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"download hash mismatch: {name}")
        target.write_bytes(data)


def create_sessions():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    sessions = []
    for name in ("model_1.onnx", "model_2.onnx"):
        path = MODEL_DIR / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != ASSETS[name][1]:
            raise ValueError(f"model hash mismatch: {name}")
        sessions.append(ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]))
    return tuple(sessions)


def enhance(samples, sessions, *, align=True):
    """Run 8 ms hops with isolated channel state and enough zeros to flush OLA.

    ``align=False`` retains the driver's 384-sample offset and padded tail.
    Alignment is only for file comparison. A live implementation cannot remove
    the time needed to collect a frame. Resampling is deliberately outside here.
    """
    if samples.ndim != 2 or not all(samples.shape) or not np.all(np.isfinite(samples)):
        raise ValueError("expected finite, nonempty frames-by-channels audio")
    first, second = sessions
    names = [[entry.name for entry in session.get_inputs()] for session in sessions]
    states = [[np.zeros(session.get_inputs()[1].shape, np.float32) for session in sessions]
              for _ in range(samples.shape[1])]
    count = (len(samples) + ALIGNMENT + HOP - 1) // HOP
    padded = np.pad(samples, ((0, count * HOP - len(samples)), (0, 0)))
    input_buffer = np.zeros((BLOCK, samples.shape[1]), np.float32)
    output_buffer = np.zeros_like(input_buffer)
    output = np.zeros_like(padded, dtype=np.float32)
    timings = []
    for start in range(0, len(padded), HOP):
        began = time.perf_counter()
        input_buffer[:-HOP] = input_buffer[HOP:]
        input_buffer[-HOP:] = padded[start:start + HOP]
        output_buffer[:-HOP] = output_buffer[HOP:]
        output_buffer[-HOP:] = 0
        for channel, state in enumerate(states):
            spectrum = np.fft.rfft(input_buffer[:, channel])
            magnitude = np.abs(spectrum).reshape(1, 1, -1).astype(np.float32)
            mask, state[0] = first.run(None, {names[0][0]: magnitude, names[0][1]: state[0]})
            estimated = np.fft.irfft(magnitude * mask * np.exp(1j * np.angle(spectrum)), n=BLOCK)
            block, state[1] = second.run(None, {
                names[1][0]: estimated.astype(np.float32), names[1][1]: state[1],
            })
            output_buffer[:, channel] += block.reshape(BLOCK)
        output[start:start + HOP] = output_buffer[:HOP]
        timings.append(time.perf_counter() - began)
    if not np.all(np.isfinite(output)):
        raise ValueError("model produced non-finite audio")
    return (output[ALIGNMENT:ALIGNMENT + len(samples)] if align else output), timings


def verify_streaming(sessions):
    rng = np.random.default_rng(42)
    audio = rng.normal(0, 0.02, (RATE, 2)).astype(np.float32)
    changed = audio.copy()
    boundary = 64 * HOP
    changed[boundary:] *= -2
    original, _ = enhance(audio, sessions, align=False)
    modified, _ = enhance(changed, sessions, align=False)
    error = float(np.max(np.abs(original[:boundary] - modified[:boundary])))
    mono, _ = enhance(audio[:, :1], sessions, align=False)
    channel_error = float(np.max(np.abs(original[:, :1] - mono)))
    if error > 1e-7 or channel_error > 1e-7:
        raise ValueError("streaming state or channel isolation check failed")
    return {"future_prefix_max_error": error, "independent_channel_max_error": channel_error}


def level_db(audio):
    return float(10 * np.log10(np.mean(audio.astype(np.float64) ** 2) + 1e-20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--download", action="store_true", help="fetch and verify pinned public model assets")
    parser.add_argument("--reduction-db", type=float, default=6)
    args = parser.parse_args()
    if not np.isfinite(args.reduction_db) or not 0 <= args.reduction_db <= 30:
        parser.error("reduction must be between 0 and 30 dB")
    audio, rate = read_wav(args.input)
    if rate != 48000 or audio.ndim != 2 or audio.shape[1] != 2 or len(audio) < rate:
        parser.error("input must be stereo 48 kHz PCM16 WAV, at least one second")
    if args.input.resolve().parent == args.output_dir.resolve():
        parser.error("use a separate output directory to protect source audio")
    if args.download:
        fetch_assets()
    import onnxruntime as ort
    import scipy
    from scipy.signal import resample_poly

    sessions = create_sessions()
    verification = verify_streaming(sessions)
    began = time.perf_counter()
    lowband = resample_poly(audio, 1, 3, axis=0)
    speech, timings = enhance(lowband, sessions)
    # Retain original frequencies above the 16 kHz model's band. This is a
    # quality-screening limitation, not a full-band cinematic separator.
    background = resample_poly(lowband - speech, 3, 1, axis=0)[:len(audio)]
    preview = audio + (10 ** (-args.reduction_db / 20) - 1) * background
    elapsed = time.perf_counter() - began
    peak = float(np.max(np.abs(preview)))
    if not np.all(np.isfinite(preview)) or peak >= 1 or np.max(np.abs(speech)) >= 1:
        raise ValueError("preview/estimate is invalid or would clip; use quieter input")
    report = {
        "model": "DTLN DNS-Challenge streaming ONNX", "upstream_commit": COMMIT,
        "asset_sha256": {name: entry[1] for name, entry in ASSETS.items()},
        "source": str(args.input), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "audio_seconds": len(audio) / rate, "reduction_db": args.reduction_db,
        "process_seconds": elapsed, "processing_time_per_audio_second": elapsed / (len(audio) / rate),
        "stereo_hop_ms_p50": float(np.percentile(timings, 50) * 1000),
        "stereo_hop_ms_p99": float(np.percentile(timings, 99) * 1000),
        "stereo_hop_ms_max": max(timings) * 1000,
        "hops_over_8ms": sum(value > 0.008 for value in timings),
        "output_level_change_db": level_db(preview) - level_db(audio),
        "speech_estimate_level_change_db": level_db(speech) - level_db(lowband),
        "peak": peak, "streaming_checks": verification,
        "file_alignment_removed_samples_16k": ALIGNMENT,
        "documented_model_delay_ms": 32,
        "versions": {"numpy": np.__version__, "onnxruntime": ort.__version__, "scipy": scipy.__version__},
        "note": "Native CPU screening with offline resampling/alignment; not measured Chrome latency. No boosting or normalization.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_wav(args.output_dir / "reduced_preview.wav", preview, rate)
    write_wav(args.output_dir / "speech_estimate_16k.wav", speech, RATE)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
