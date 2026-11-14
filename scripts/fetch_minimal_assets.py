"""Range-download only unmodified assets needed by the P1 Clean stapler task."""

import argparse
import hashlib
import io
import json
import os
import zipfile
import zlib
from pathlib import Path

from remotezip import RemoteZip
import requests
from download_assets import verify_cached_asset

ROOT = Path(__file__).resolve().parents[1]


class OffsetBuffer(io.BytesIO):
    """A contiguous ZIP span retaining absolute archive offsets."""

    def __init__(self, data, offset):
        super().__init__(data)
        self.offset = offset

    def seek(self, offset, whence=0):
        return super().seek(offset - self.offset if whence == 0 else offset, whence) + self.offset

    def tell(self):
        return super().tell() + self.offset


def prefetch_span(stream, members, url):
    start = min(m.header_offset for m in members)
    end = min(stream.start_dir, max(m.header_offset + m.compress_size for m in members) + 65536) - 1
    if end - start > 64 * 1024 * 1024:
        return  # Large, sparse selections retain the original range-reader path.
    response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    response.raise_for_status()
    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(
        f"bytes {start}-{end}/"
    ):
        raise RuntimeError("Server did not honor archive byte range")
    if len(response.content) != end - start + 1:
        raise RuntimeError("Truncated archive byte range")
    stream.fp.close()
    stream.fp = OffsetBuffer(response.content, start)


def selected(name, archive, objects=("048_stapler",)):
    if archive == "embodiments.zip":
        return name.startswith("embodiments/aloha-agilex/")
    return any(name.startswith(f"objects/{obj}/") for obj in objects) or name in {
        "objects/same.json",
        "objects/objaverse/list.json",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--archive", action="append", choices=["embodiments.zip", "objects.zip"])
    parser.add_argument("--object", action="append", dest="objects")
    args = parser.parse_args()
    objects = args.objects or ["048_stapler"]
    if any("/" in obj or ".." in obj for obj in objects):
        raise ValueError("Object names must be single directory names")
    spec = json.loads((ROOT / "configs/assets.json").read_text())["robotwin"]
    base = Path(os.environ["TD_VLA_STORAGE_ROOT"]) / "assets/robotwin-minimal"
    base.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    for archive in args.archive or ("embodiments.zip", "objects.zip"):
        manifest_path = base / f"manifest-{archive}.json"
        records = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        url = f"{endpoint}/datasets/{spec['repo_id']}/resolve/{spec['revision']}/{archive}"
        local = base.parent / "robotwin" / archive
        if local.exists():
            verify_cached_asset(local, spec, archive)
        source = (
            zipfile.ZipFile(local)
            if local.exists()
            else RemoteZip(url, timeout=120, initial_buffer_size=1024 * 1024)
        )
        with source as stream:
            members = [
                i
                for i in stream.infolist()
                if selected(i.filename, archive, objects) and not i.is_dir()
            ]
            assert members, f"No matching entries in {archive}"
            print(
                json.dumps(
                    {
                        "archive": archive,
                        "files": len(members),
                        "compressed_bytes": sum(i.compress_size for i in members),
                    }
                ),
                flush=True,
            )
            if args.list_only:
                continue
            if not local.exists():
                missing = []
                for member in members:
                    cached = base / member.filename
                    content = cached.read_bytes() if cached.is_file() else b""
                    if len(content) != member.file_size or zlib.crc32(content) != member.CRC:
                        missing.append(member)
                if missing:
                    prefetch_span(stream, missing, url)
            for member in members:
                path = (base / member.filename).resolve()
                if not path.is_relative_to(base.resolve()):
                    raise ValueError(member.filename)
                old = records.get(member.filename, {})
                if (
                    path.exists()
                    and old.get("revision") == spec["revision"]
                    and hashlib.sha256(path.read_bytes()).hexdigest() == old.get("sha256")
                ):
                    continue
                content = path.read_bytes() if path.exists() else b""
                if len(content) != member.file_size or zlib.crc32(content) != member.CRC:
                    content = stream.read(member)  # ZipFile verifies member CRC.
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(path.suffix + ".partial")
                partial.write_bytes(content)
                partial.replace(path)
                records[member.filename] = {
                    "revision": spec["revision"],
                    "archive": archive,
                    "endpoint": endpoint,
                    "crc32": member.CRC,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
                manifest_path.write_text(json.dumps(records, indent=2) + "\n")
                print(member.filename, flush=True)
    if not args.list_only:
        for name in ("embodiments", "objects"):
            link = ROOT / "external/RoboTwin/assets" / name
            target = base / name
            if not target.is_dir():
                continue
            if link.is_symlink() and link.resolve() == target.resolve():
                continue
            if link.exists() or link.is_symlink():
                raise RuntimeError(f"Existing asset path: {link}; inspect manually")
            link.symlink_to(target, target_is_directory=True)


if __name__ == "__main__":
    main()
