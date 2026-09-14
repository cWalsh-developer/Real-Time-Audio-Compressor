"""Command-line entry point for the optional PyTorch baseline trainer."""

import argparse
import json
from pathlib import Path

from .training import train_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="prepared dataset root")
    parser.add_argument("output", type=Path, help="checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    history = train_model(
        args.root, args.output, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, device=args.device, resume=args.resume,
        max_batches=args.max_batches, seed=args.seed,
    )
    print(json.dumps(history[-1], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())