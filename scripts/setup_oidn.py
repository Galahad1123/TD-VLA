"""Install the pinned Blackwell-compatible OIDN into the locked SAPIEN wheel.

Run with envs/sim/.venv/bin/python after syncing sim. Original libraries remain.
"""
import argparse
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import shutil
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "configs/oidn.json").read_text())
    assert version("sapien") == "3.0.0b1", "Unexpected SAPIEN version"
    digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise RuntimeError("OIDN archive checksum mismatch")
    root = Path(distribution("sapien").locate_file("sapien"))
    loader = root / "_oidn_tricks.py"
    original = loader.read_text()
    expected = ("libOpenImageDenoise_core.so.", "libOpenImageDenoise.so.")
    for prefix in expected:
        if original.count(prefix + "2.0.1") + original.count(prefix + "2.3.3") != 1:
            raise RuntimeError("Unexpected SAPIEN loader; refusing to patch")
    with tarfile.open(args.archive) as archive:
        for name in ("libOpenImageDenoise", "libOpenImageDenoise_core",
                     "libOpenImageDenoise_device_cuda"):
            filename = name + ".so.2.3.3"
            member = archive.getmember("oidn-2.3.3.x86_64.linux/lib/" + filename)
            if not member.isfile():
                raise RuntimeError(f"Expected a regular library: {filename}")
            with archive.extractfile(member) as source, (root / "oidn_library" / filename).open("wb") as dest:
                shutil.copyfileobj(source, dest)
    backup = loader.with_suffix(".py.original")
    if not backup.exists():
        backup.write_text(original)
    loader.write_text(original.replace("2.0.1", "2.3.3"))
    (root / "oidn_library/td-vla-receipt.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Installed OIDN 2.3.3; run check_render.py --ray-tracing in a new process")


if __name__ == "__main__":
    main()
