from pathlib import Path
import subprocess


def test_storage_symlinks_ignored_but_data_package_tracked(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / ".gitignore").write_text((root / ".gitignore").read_text())
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "storage").mkdir()
    for name in ("data", "assets", "logs", "outputs", "pretrained"):
        (tmp_path / name).symlink_to(tmp_path / "storage", target_is_directory=True)
        result = subprocess.run(["git", "check-ignore", "-q", name], cwd=tmp_path)
        assert result.returncode == 0, f"Storage symlink {name} must be ignored"
    source = tmp_path / "src/td_vla/data"
    source.mkdir(parents=True)
    (source / "__init__.py").touch()
    result = subprocess.run(["git", "check-ignore", "-q", "src/td_vla/data/__init__.py"],
                            cwd=tmp_path)
    assert result.returncode == 1, "Data implementation must not be ignored"
