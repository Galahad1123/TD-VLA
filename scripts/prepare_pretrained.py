"""Fetch pinned official initialization assets; authentication stays in the HF credential store."""

import argparse
import json
import os
import subprocess
from urllib.parse import urlencode
from pathlib import Path

from huggingface_hub import hf_hub_download
from td_vla.data.multistream import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["bert", "vision", "fusion"],
        default=["bert", "vision", "fusion"],
    )
    parser.add_argument(
        "--public-distribution",
        action="store_true",
        help="Use pinned public ModelScope distribution where configured",
    )
    args = parser.parse_args()
    specs = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/pretrained.json").read_text()
    )
    for component in args.components:
        spec = specs[component]
        directory = args.output / component
        receipts = []
        directory.mkdir(parents=True, exist_ok=True)
        distribution = spec.get("public_distribution") if args.public_distribution else None
        for name in spec["files"]:
            if distribution:
                url = (
                    "https://modelscope.cn/api/v1/models/"
                    + distribution["repo_id"]
                    + "/repo?"
                    + urlencode({"Revision": distribution["revision"], "FilePath": name})
                )
                path = directory / name
                expected = distribution["files"][name]
                if not path.exists() or sha256(path) != expected:
                    subprocess.run(
                        [
                            "aria2c",
                            "--continue=true",
                            "--allow-overwrite=true",
                            "--auto-file-renaming=false",
                            "-x",
                            "16",
                            "-s",
                            "16",
                            "-k",
                            "1M",
                            "--connect-timeout=20",
                            "--timeout=60",
                            "--max-tries=3",
                            "--console-log-level=warn",
                            "--summary-interval=60",
                            "--dir=" + str(directory),
                            "--out=" + name + ".partial",
                            url,
                        ],
                        check=True,
                    )
                    partial = path.with_name(name + ".partial")
                    if sha256(partial) != expected:
                        raise ValueError(f"Public distribution checksum mismatch: {name}")
                    partial.replace(path)
            else:
                for endpoint in dict.fromkeys(
                    [
                        os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
                        "https://huggingface.co",
                    ]
                ):
                    try:
                        path = Path(
                            hf_hub_download(
                                spec["repo_id"],
                                name,
                                revision=spec["revision"],
                                local_dir=directory,
                                endpoint=endpoint,
                            )
                        )
                        break
                    except Exception as error:
                        print(f"{component}/{name}: {endpoint} {type(error).__name__}", flush=True)
                else:
                    raise RuntimeError(
                        f"Cannot fetch {component}/{name}; check official access/login"
                    )
            receipts.append(dict(file=name, sha256=sha256(path), bytes=path.stat().st_size))
            print(f"verified {component}/{name}", flush=True)
        (directory / "manifest.json").write_text(
            json.dumps(
                dict(
                    schema="td-vla-pretrained-v1",
                    **spec,
                    artifacts=receipts,
                    download_source=distribution
                    or {
                        "provider": "huggingface",
                        "repo_id": spec["repo_id"],
                        "revision": spec["revision"],
                    },
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
