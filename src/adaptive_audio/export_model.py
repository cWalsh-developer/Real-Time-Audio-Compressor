"""Export the trained spectral mask baseline for browser/runtime experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .training import N_FFT, SpectralMaskNet


def export_checkpoint(checkpoint_path, output_path, *, opset=17):
    """Export a checkpoint's mask network with a fixed four-second frame shape."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = SpectralMaskNet().eval()
    model.load_state_dict(checkpoint["model"])
    example = torch.zeros(1, N_FFT // 2 + 1, 1 + (4 * 16000 - N_FFT) // 256)
    torch.onnx.export(
        model, (example,), output_path, input_names=["mixture_magnitude"],
        output_names=["speech_mask"], opset_version=opset,
        dynamo=False, do_constant_folding=True,
    )
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata = {
        "format": "onnx",
        "opset": opset,
        "sample_rate": 16000,
        "fft_size": N_FFT,
        "hop_length": 256,
        "window_seconds": 4,
        "input_shape": [1, N_FFT // 2 + 1, 1 + (4 * 16000 - N_FFT) // 256],
        "output_shape": [1, N_FFT // 2 + 1, 1 + (4 * 16000 - N_FFT) // 256],
        "sha256": digest,
        "checkpoint": str(Path(checkpoint_path)),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(export_checkpoint(args.checkpoint, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())