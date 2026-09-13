"""Benchmark the licensed local Rapidly SDK through its public C API.

Demo mode records performance only: watermarked output is not a quality test.
"""

import argparse
import ctypes as ct
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from adaptive_audio.wav import read_wav, write_wav


ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT / "models/rapidly-sdk"
SDK_COMMIT = "a75155656556b38709fe77a706958dcbe1f13624"
DLL_HASH = "e47e01eb3a43efa41d9de4a8e2f32edb7a63a975fc320da458fa1af8bc31337a"
MODEL_HASHES = {
    "32ms": "91f9b2af12e080dbf7fc9753662f9cec4191836098af5d255c33ff53dd08bc23",
    "96ms": "abd3e085878bb7e0b619611275d4b57cca3b514644e2d3b4552d3f4b2d80d54f",
}
PCM_POINTER = ct.POINTER(ct.c_float)


class ProcessorInfo(ct.Structure):
    _fields_ = [("sampleRate", ct.c_double), ("numOfModelChannels", ct.c_int32),
                ("latencyInSamples", ct.c_int32), ("blockSize", ct.c_int32),
                ("hopSize", ct.c_int32)]


def verify_hash(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"SDK asset hash mismatch: {path.name}")


def load_api():
    if platform.system() != "Windows" or ct.sizeof(ct.c_void_p) != 8:
        raise ValueError("this isolated trial requires Windows x64")
    dll = SDK / "bin/windows-x64/RapidlyEngine.dll"
    verify_hash(dll, DLL_HASH)
    api = ct.CDLL(str(dll))
    signatures = {
        "rapidlyAddLicense": ([ct.c_char_p], ct.c_bool),
        "rapidlyCreateProcessor": ([ct.c_char_p, ct.c_int32, ct.c_double], ct.c_void_p),
        "rapidlyDeleteProcessor": ([ct.c_void_p], None),
        "rapidlyGetProcessorInfo": ([ct.c_void_p, ct.POINTER(ProcessorInfo)], None),
        "rapidlyGetNumOfOutputBusses": ([ct.c_void_p], ct.c_int32),
        "rapidlyGetOutputBusName": ([ct.c_void_p, ct.c_int32, ct.POINTER(ct.c_char), ct.c_int32], None),
        "rapidlySetParameterValue": ([ct.c_void_p, ct.c_int32, ct.c_float], None),
        "rapidlyAddAudioInterleaved": ([ct.c_void_p, PCM_POINTER, ct.c_int32], None),
        "rapidlyGetNumOfPendingSamples": ([ct.c_void_p], ct.c_int32),
        "rapidlyGetAudioInterleaved": ([ct.c_void_p, PCM_POINTER, ct.c_int32], ct.c_bool),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes = arguments
        function.restype = result
    return api


def render(api, handle, audio: np.ndarray, rate: int, reduction_db: float) -> tuple[np.ndarray, dict]:
    info = ProcessorInfo()
    api.rapidlyGetProcessorInfo(handle, ct.byref(info))
    buses = []
    for index in range(api.rapidlyGetNumOfOutputBusses(handle)):
        name = ct.create_string_buffer(256)
        api.rapidlyGetOutputBusName(handle, index, name, len(name))
        buses.append(name.value.decode("utf-8"))
    if sorted(name.lower() for name in buses) != ["dialogue", "noise"]:
        raise ValueError(f"unexpected model outputs: {buses}")
    for index, name in enumerate(buses):
        gain = 1 if name.lower() == "dialogue" else 10 ** (-reduction_db / 20)
        api.rapidlySetParameterValue(handle, 0x0100 + index, gain)
        api.rapidlySetParameterValue(handle, 0x0200 + index, 0)
    block = 480  # 10 ms at the required 48 kHz input rate.
    output = np.empty_like(audio)
    written, supplied = 0, 0
    first_output_at = None
    timings = []
    began = time.perf_counter()
    # SDK output is aligned by its queue, as in the vendor's process-file demo.
    # Supply at most one extra second of silence to drain the delayed tail.
    while written < len(audio) and supplied < len(audio) + rate:
        chunk = np.zeros((block, 2), np.float32)
        count = min(block, max(0, len(audio) - supplied))
        if count:
            chunk[:count] = audio[supplied:supplied + count]
        started = time.perf_counter()
        api.rapidlyAddAudioInterleaved(handle, chunk.ctypes.data_as(PCM_POINTER), block)
        supplied += block
        ready = api.rapidlyGetNumOfPendingSamples(handle)
        if ready < 0 or ready > supplied:
            raise ValueError("invalid SDK output queue size")
        take = min(ready, len(audio) - written)
        if take:
            destination = output[written:written + take]
            if not api.rapidlyGetAudioInterleaved(handle, destination.ctypes.data_as(PCM_POINTER), take):
                raise RuntimeError("SDK failed to return available audio")
            written += take
            if first_output_at is None:
                first_output_at = supplied
        timings.append(time.perf_counter() - started)
    if written != len(audio) or not np.all(np.isfinite(output)):
        raise RuntimeError("SDK returned incomplete or invalid output")
    report = {
        "processor_info": {name: getattr(info, name) for name, _ in info._fields_},
        "output_buses": buses, "first_output_after_input_ms": first_output_at / rate * 1000,
        "process_seconds": time.perf_counter() - began,
        "host_block_ms": block / rate * 1000,
        "block_time_ms_p50": float(np.percentile(timings, 50) * 1000),
        "block_time_ms_p99": float(np.percentile(timings, 99) * 1000),
        "block_time_ms_max": max(timings) * 1000,
        "blocks_over_10ms": sum(value > 0.01 for value in timings),
        "peak": float(np.max(np.abs(output))),
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", choices=MODEL_HASHES, default="32ms")
    parser.add_argument("--reduction-db", type=float, default=6)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="watermarked performance test; no WAV exported")
    mode.add_argument("--license-file", type=Path, help="private file containing an offline lk_ trial key")
    args = parser.parse_args()
    if not np.isfinite(args.reduction_db) or not 0 <= args.reduction_db <= 30:
        parser.error("reduction must be between 0 and 30 dB")
    audio, rate = read_wav(args.input)
    if rate != 48000 or audio.ndim != 2 or audio.shape[1] != 2 or len(audio) < rate:
        parser.error("input must be stereo 48 kHz PCM16 WAV, at least one second")
    key = None
    if args.license_file:
        if not args.license_file.is_file():
            parser.error("licence file is missing; see evaluation/rapidly-trial.md")
        key = args.license_file.read_text(encoding="utf-8-sig").strip()
        if not key.startswith("lk_"):
            parser.error("this local trial requires an offline lk_ key; subscription/activation keys are not supported")
    api = load_api()
    if key is not None and not api.rapidlyAddLicense(key.encode("utf-8")):
        parser.error("SDK rejected the licence key")
    model = SDK / "models" / f"speech-denoise-{args.model}.v1.1.rapidly"
    verify_hash(model, MODEL_HASHES[args.model])
    handle = api.rapidlyCreateProcessor(str(model).encode("utf-8"), 2, rate)
    if not handle:
        raise RuntimeError("SDK could not create the model processor")
    try:
        output, report = render(api, handle, audio, rate, args.reduction_db)
    finally:
        api.rapidlyDeleteProcessor(handle)
    report.update({
        "sdk_commit": SDK_COMMIT, "sdk_sha256": DLL_HASH, "model": model.name,
        "model_sha256": MODEL_HASHES[args.model],
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "audio_seconds": len(audio) / rate, "reduction_db": args.reduction_db,
        "licence_mode": "watermarked_demo" if args.demo else "offline_key_accepted",
        "quality_status": "not_evaluated_watermarked" if args.demo else "model_entitlement_requires_vendor_confirmation",
        "note": "Native performance only. First output availability is not measured browser end-to-end delay.",
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.demo:
        if report["peak"] >= 1:
            raise ValueError("preview would clip; use a quieter test input")
        write_wav(args.output_dir / "reduced_preview.wav", output, rate)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
