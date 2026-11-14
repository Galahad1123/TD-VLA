"""Verify selected source files and write an immutable episode split plus train statistics."""

import argparse
import json
from pathlib import Path

from td_vla.data.multistream import prepare_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    split = prepare_split(args.data, args.validation_per_task, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        file.write(json.dumps(split, indent=2) + "\n")
    print(json.dumps({"train": len(split["train"]), "validation": len(split["validation"])}))


if __name__ == "__main__":
    main()
