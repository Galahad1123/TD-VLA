"""Fetch immutable upstream revisions; never reset a dirty or different checkout."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main():
    specs = json.loads((ROOT / "configs/upstreams.json").read_text())
    names = sys.argv[1:] or list(specs)
    for name in names:
        spec = specs[name]
        path = ROOT / "external" / name
        if path.exists():
            actual = git("rev-parse", "HEAD", cwd=path)
            if actual != spec["commit"]:
                raise RuntimeError(f"{path}: expected {spec['commit']}, got {actual}; inspect manually")
            if git("status", "--porcelain", "--untracked-files=no", cwd=path):
                raise RuntimeError(f"{path}: tracked changes present; inspect manually")
        else:
            path.mkdir(parents=True)
            git("init", cwd=path)
            git("remote", "add", "origin", spec["url"], cwd=path)
            git("fetch", "--depth", "1", "origin", spec["commit"], cwd=path)
            git("checkout", "--detach", spec["commit"], cwd=path)
        print(name, spec["commit"])


if __name__ == "__main__":
    main()
