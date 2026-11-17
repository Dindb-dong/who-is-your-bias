from __future__ import annotations

import argparse
import json

from .pipeline import fit_driver, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reviewed Jensen face/FLAME pairs and fit a private expression driver")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--recipe", required=True)
    prepare.add_argument("--output", required=True)
    train = sub.add_parser("fit")
    train.add_argument("--dataset", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--ridge", type=float, default=0.01)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--max-batches", type=int)
    train.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_dataset(args.recipe, args.output)
    else:
        result = fit_driver(args.dataset, args.checkpoint, args.output, ridge=args.ridge, batch_size=args.batch_size, resume=args.resume, max_batches=args.max_batches)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") == "rejected_validation":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
