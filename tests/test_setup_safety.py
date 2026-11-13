import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_download_module():
    spec = importlib.util.spec_from_file_location("download_assets", ROOT / "scripts/download_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("existing", ["directory", "wrong-link"])
def test_storage_conflicts_fail_without_replacing_files(tmp_path, existing):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for filename in ("env.sh", "setup_storage.sh"):
        shutil.copy(ROOT / "scripts" / filename, repo / "scripts" / filename)
    old = tmp_path / "old"
    old.mkdir()
    if existing == "directory":
        (repo / "data").mkdir()
    else:
        (repo / "data").symlink_to(old, target_is_directory=True)
    (repo / "data/keep.txt").write_text("user data")
    result = subprocess.run(["bash", str(repo / "scripts/setup_storage.sh")],
        env={**os.environ, "TD_VLA_STORAGE_ROOT": str(tmp_path / "storage")}, capture_output=True)
    assert result.returncode != 0
    assert (repo / "data/keep.txt").read_text() == "user data"


@pytest.mark.parametrize("problem", ["revision", "repo_id", "checksum", "missing_receipt", "none"])
def test_cached_asset_requires_matching_origin_and_bytes(tmp_path, problem):
    module = load_download_module()
    archive = tmp_path / "objects.zip"
    archive.write_bytes(b"fixture archive")
    expected = {"repo_id": "official/assets", "repo_type": "dataset", "revision": "fixed"}
    receipt = {**expected, "file": "objects.zip",
               "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    if problem in ("revision", "repo_id"):
        receipt[problem] = "different"
    if problem == "checksum":
        archive.write_bytes(b"wrong bytes")
    if problem != "missing_receipt":
        archive.with_suffix(".zip.receipt.json").write_text(json.dumps(receipt))
    if problem == "none":
        assert module.verify_cached_asset(archive, expected, "objects.zip") == receipt
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            module.verify_cached_asset(archive, expected, "objects.zip")
