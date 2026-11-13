"""Collect reproducibility evidence without recording environment secrets."""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def capture(command):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
    return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
packages = {}
for name in ("td-vla", "torch", "torchvision", "transformers", "sapien", "mplib", "nvidia-curobo"):
    try:
        packages[name] = version(name)
    except PackageNotFoundError:
        packages[name] = None
report = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
          "packages": packages, "commit": capture(["git", "rev-parse", "HEAD"]),
          "worktree": capture(["git", "status", "--porcelain"]), "locks": {},
          "storage_root": os.environ.get("TD_VLA_STORAGE_ROOT"), "upstreams": {}}
for path in (ROOT / "uv.lock", ROOT / "envs/policy/uv.lock", ROOT / "envs/sim/uv.lock"):
    report["locks"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
for name in json.loads((ROOT / "configs/upstreams.json").read_text()):
    report["upstreams"][name] = capture(["git", "-C", str(ROOT / "external" / name),
                                        "rev-parse", "HEAD"])
for command in (["uv", "--version"], ["nvidia-smi"], ["df", "-h", str(ROOT)]):
    try:
        report[command[0]] = capture(command)
    except FileNotFoundError:
        report[command[0]] = {"available": False}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(args.output)
