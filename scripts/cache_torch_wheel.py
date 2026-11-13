"""Fetch the Linux CPython 3.10 CUDA wheel and validate against the uv lock."""
import hashlib
import os
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlparse

import tomli

ROOT = Path(__file__).resolve().parents[1]
lock = tomli.loads((ROOT / "envs/policy/uv.lock").read_text())
package = next(p for p in lock["package"] if p["name"] == "torch")
wheel = next(w for w in package["wheels"] if "cp310-cp310-manylinux_2_28_x86_64" in w["url"])

folder = Path(os.environ["TD_VLA_STORAGE_ROOT"]) / "cache/wheels"
folder.mkdir(parents=True, exist_ok=True)
filename = unquote(Path(urlparse(wheel["url"]).path).name)
path = folder / filename
subprocess.run(["aria2c", "--continue=true", "--allow-overwrite=true",
                "--auto-file-renaming=false", "-x16", "-s32", "-k1M", "--max-tries=5",
                "--connect-timeout=30", "--timeout=120", "--summary-interval=60",
                "--console-log-level=warn", "--dir", str(folder), "--out", filename,
                wheel["url"], wheel["url"].replace("download-r2.pytorch.org",
                                                    "download.pytorch.org")], check=True)
sha = hashlib.sha256()
with path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        sha.update(block)
assert "sha256:" + sha.hexdigest() == wheel["hash"], "Wheel hash mismatch; do not install"
print("verified", filename, wheel["hash"], flush=True)
