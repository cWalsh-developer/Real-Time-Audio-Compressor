"""Command-line entry point for measuring trained-model inference throughput."""

import argparse
import json
from pathlib import Path

from .training_eval import benchmark_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("root", type=Path, help="prepared dataset root")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    print(json.dumps(benchmark_checkpoint(
        args.checkpoint, args.root, split=args.split, batch_size=args.batch_size,
        batches=args.batches, device=args.device,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())