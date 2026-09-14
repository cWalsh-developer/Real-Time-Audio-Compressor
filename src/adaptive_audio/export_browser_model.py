"""Export the trained spectral mask network for browser inference."""

import argparse
from pathlib import Path

import torch

from .training import SpectralMaskNet


def export_model(checkpoint, output, *, opset=17):
    """Export a checkpoint as an ONNX model accepting [batch, frequency, frames]."""
    checkpoint = Path(checkpoint)
    output = Path(output)
    device = torch.device("cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = SpectralMaskNet().to(device).eval()
    model.load_state_dict(state["model"])
    example = torch.zeros((1, 257, 251), dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        output,
        input_names=["mixture_magnitude"],
        output_names=["speech_mask"],
        dynamic_axes={"mixture_magnitude": {0: "batch", 2: "frames"},
                      "speech_mask": {0: "batch", 2: "frames"}},
        opset_version=opset,
        dynamo=False,
    )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(export_model(args.checkpoint, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())