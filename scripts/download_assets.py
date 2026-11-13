"""Download explicit immutable asset files, preserving checksum and source receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[1]


def verify_cached_asset(path, spec, filename):
    receipt = json.loads(path.with_suffix(path.suffix + ".receipt.json").read_text())
    for key in ("repo_id", "repo_type", "revision"):
        if receipt.get(key) != spec[key]:
            raise ValueError(f"Asset origin mismatch ({key}): {path}")
    if receipt.get("file") != filename or receipt.get("sha256") != digest(path):
        raise ValueError(f"Asset checksum or filename mismatch: {path}")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", choices=["robotwin", "reference_policy"])
    args = parser.parse_args()
    spec = json.loads((ROOT / "configs/assets.json").read_text())[args.asset]
    storage = Path(os.environ["TD_VLA_STORAGE_ROOT"])
    dest = storage / ("assets" if args.asset == "robotwin" else "pretrained") / args.asset
    dest.mkdir(parents=True, exist_ok=True)
    prefix = "datasets/" if spec["repo_type"] == "dataset" else ""
    receipts = []
    for filename in spec["files"]:
        path = dest / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path = path.with_suffix(path.suffix + ".receipt.json")
        if path.exists() and receipt_path.exists():
            receipt = verify_cached_asset(path, spec, filename)
            receipts.append(receipt)
            print(f"verified existing {filename}", flush=True)
            continue
        partial = path.with_suffix(path.suffix + ".partial")
        for endpoint in dict.fromkeys([os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
                                       "https://huggingface.co"]):
            url = f"{endpoint}/{prefix}{spec['repo_id']}/resolve/{spec['revision']}/{filename}"
            command = ["curl", "--fail", "--location", "--retry", "3",
                                     "--connect-timeout", "30", "--speed-time", "120",
                                     "--speed-limit", "1024", "--continue-at", "-",
                                     "--output", str(partial), url]
            if shutil.which("aria2c"):
                command = ["aria2c", "--continue=true", "--allow-overwrite=true",
                           "--auto-file-renaming=false", "--max-connection-per-server=16",
                           "--split=16", "--min-split-size=1M", "--max-tries=3",
                           "--connect-timeout=30", "--timeout=120", "--summary-interval=60",
                           "--console-log-level=warn", "--dir", str(path.parent),
                           "--out", partial.name, url]
            result = subprocess.run(command)
            if result.returncode == 0:
                partial.replace(path)
                receipt = {**spec, "file": filename, "endpoint": endpoint,
                           "sha256": digest(path), "bytes": path.stat().st_size}
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
                receipts.append(receipt)
                break
        else:
            raise RuntimeError(f"Download failed: {filename}; partial retained for resume")
    (dest / "manifest.json").write_text(json.dumps(receipts, indent=2) + "\n")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


if __name__ == "__main__":
    main()
