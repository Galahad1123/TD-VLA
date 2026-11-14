"""Strict, provenance-checked base initialization; never load a RoboTwin policy here."""

import json
from pathlib import Path

import torch
from td_vla.data.multistream import sha256

MAPPINGS = (
    ("bert.", "text_encoder.bert."),
    ("feat_map.", "text_encoder.text_projection."),
    ("transformer.encoder.text_layers.", "vision_language_interaction.text_layers."),
    ("transformer.encoder.fusion_layers.", "vision_language_interaction.fusion_layers."),
)


def map_grounding_state(source, target):
    required = {key for key in target if any(key.startswith(dst) for _, dst in MAPPINGS)}
    loaded = {}
    for key, value in source.items():
        key = key.removeprefix("module.")
        for src, dst in MAPPINGS:
            if key.startswith(src):
                dest = dst + key[len(src) :]
                if dest in required:
                    if value.shape != target[dest].shape:
                        raise ValueError(f"Initialization shape mismatch: {dest}")
                    if dest in loaded:
                        raise ValueError(f"Duplicate initialization tensor: {dest}")
                    loaded[dest] = value
                break
    missing = sorted(required - loaded.keys())
    if not required or missing:
        raise ValueError(f"Incomplete GroundingDINO initialization: {missing[:10]}")
    report = dict(
        loaded_tensors=len(loaded),
        groups={dst: sum(k.startswith(dst) for k in loaded) for _, dst in MAPPINGS},
    )
    return loaded, report


def verify_pretrained(root, specs):
    root = Path(root)
    provenance = {}
    for name, spec in specs.items():
        directory = root / name
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("schema") != "td-vla-pretrained-v1":
            raise ValueError(f"Invalid initialization manifest: {name}")
        if any(manifest.get(k) != spec[k] for k in ("repo_id", "revision", "files")):
            raise ValueError(f"Initialization origin mismatch: {name}")
        artifacts = manifest["artifacts"]
        if len(artifacts) != len(spec["files"]) or {a["file"] for a in artifacts} != set(
            spec["files"]
        ):
            raise ValueError(f"Incomplete initialization files: {name}")
        for artifact in artifacts:
            path = directory / artifact["file"]
            expected = spec.get("public_distribution", {}).get("files", {}).get(artifact["file"])
            if expected is not None and artifact["sha256"] != expected:
                raise ValueError(
                    f"Pinned initialization checksum mismatch: {name}/{artifact['file']}"
                )
            if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
                raise ValueError(f"Initialization checksum mismatch: {name}/{artifact['file']}")
        provenance[name] = manifest
    return provenance


def load_grounding_initialization(model, path):
    # Trusted official checkpoint, after verify_pretrained validated origin and SHA256.
    source = torch.load(path, map_location="cpu", weights_only=False)
    source = source.get("model", source)
    loaded, report = map_grounding_state(source, model.state_dict())
    model.load_state_dict(loaded, strict=False)
    return report
