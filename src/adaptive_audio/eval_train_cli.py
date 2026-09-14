"""Command-line entry point for evaluating the optional trained baseline."""

import argparse
import json
from pathlib import Path

from .training_eval import write_evaluation_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("root", type=Path, help="prepared dataset root")
    parser.add_argument("output", type=Path, help="JSON report path")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args(argv)
    report = write_evaluation_report(
        args.checkpoint, args.root, args.output, split=args.split,
        batch_size=args.batch_size, device=args.device, max_batches=args.max_batches,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())