"""Extract verified archives on the data disk and link them into upstream."""
import json
import os
from pathlib import Path
import zipfile

from download_assets import digest

ROOT = Path(__file__).resolve().parents[1]
base = Path(os.environ["TD_VLA_STORAGE_ROOT"]) / "assets/robotwin"
for stem in ("embodiments", "objects", "background_texture"):
    archive = base / f"{stem}.zip"
    receipt = json.loads(archive.with_suffix(".zip.receipt.json").read_text())
    assert digest(archive) == receipt["sha256"], f"Checksum mismatch: {archive}"
    extracted = base / "extracted"
    extracted.mkdir(exist_ok=True)
    marker = extracted / f".{stem}-{receipt['sha256']}.done"
    if not marker.exists():
        with zipfile.ZipFile(archive) as stream:
            for member in stream.infolist():
                target = (extracted / member.filename).resolve()
                if not target.is_relative_to(extracted.resolve()):
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            stream.extractall(extracted)
        marker.touch()
    source = extracted / stem
    assert source.is_dir(), f"Unexpected archive layout: {source}"
    link = ROOT / "external/RoboTwin/assets" / stem
    if link.is_symlink() and link.resolve() == source.resolve():
        continue
    if link.exists() or link.is_symlink():
        raise RuntimeError(f"Refusing to replace existing asset path: {link}")
    link.symlink_to(source, target_is_directory=True)
    print(link, "->", source)
