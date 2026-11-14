"""Compare trusted self-generated checkpoints after continuous and resumed training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def compare(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or not torch.equal(a, b):
            raise AssertionError(f"Tensor mismatch: {path}")
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            compare(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]")
    else:
        assert a == b, path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("continuous", type=Path)
    p.add_argument("resumed", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    first = torch.load(a.continuous, map_location="cpu", weights_only=False, mmap=True)
    second = torch.load(a.resumed, map_location="cpu", weights_only=False, mmap=True)
    compare(first, second)
    result = {
        "exact_resume": True,
        "step": first["step"],
        "checked": [
            "model",
            "optimizer",
            "scheduler",
            "ema",
            "per-rank RNG",
            "metadata",
            "data fingerprint",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
