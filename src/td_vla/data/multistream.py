"""Manifest-only v4 dataset: lazy RGB reads, episode splits, train-only statistics."""

from bisect import bisect_right
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from td_vla.annotation.geometry import future_indices
from td_vla.annotation.stages import VOCAB
from td_vla.data.pilot import INSTRUCTIONS

# RoboTwin raw: left joints, left grip, right joints, right grip.
# TurboVLA: left joints, right joints, left grip, right grip.
POLICY_ORDER = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_commands(raw, split):
    value = np.asarray(raw, dtype=np.float32)[..., POLICY_ORDER].copy()
    low, high = np.asarray(split["min"]), np.asarray(split["max"])
    span = high[:12] - low[:12]
    value[..., :12] = np.where(
        span > 1e-6, 2 * (value[..., :12] - low[:12]) / np.maximum(span, 1e-6) - 1, 0
    )
    value[..., 12:] = value[..., 12:] > 0.49
    return value.astype(np.float32)


def denormalize_actions(value, split):
    value = np.asarray(value, dtype=np.float32).copy()
    low, high = np.asarray(split["min"]), np.asarray(split["max"])
    value[..., :12] = (np.clip(value[..., :12], -1, 1) + 1) * 0.5 * (high[:12] - low[:12]) + low[
        :12
    ]
    value[..., 12:] = value[..., 12:] > 0.49
    return value[..., np.argsort(POLICY_ORDER)].astype(np.float32)


def prepare_split(root, validation_per_task=5, seed=42):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["schema"] != "td-vla-selected-episodes-v1"
        or manifest["annotation_schema"] != "pilot-labels-v4"
    ):
        raise ValueError("Expected selected manifest with v4 labels")
    episodes = sorted(manifest["episodes"], key=lambda e: (e["task"], e["seed"]))
    identities, groups, commands = set(), {}, []
    for i, e in enumerate(episodes):
        identity = (e["task"], e["seed"])
        if identity in identities:
            raise ValueError("Duplicate task/seed episode")
        identities.add(identity)
        groups.setdefault(e["task"], []).append(i)
        raw, labels = root / e["raw_path"], root / e["label_path"]
        if sha256(raw) != e["raw_sha256"] or sha256(labels / "labels.npz") != e["labels_sha256"]:
            raise ValueError(f"Episode checksum mismatch: {identity}")
        report = json.loads((labels / "report.json").read_text())
        if any(
            report[k] != value
            for k, value in {
                "schema": "pilot-labels-v4",
                "task": e["task"],
                "seed": e["seed"],
                "frames": e["frames"],
                "raw_sha256": e["raw_sha256"],
                "vocab": VOCAB[e["task"]],
            }.items()
        ):
            raise ValueError(f"Annotation identity mismatch: {identity}")
        with h5py.File(raw) as f:
            if (
                not f.attrs["expert_success"]
                or f.attrs["task"] != e["task"]
                or int(f.attrs["seed"]) != e["seed"]
                or f["qpos"].shape != (e["frames"], 14)
            ):
                raise ValueError("Raw identity/shape mismatch")
            q = f["qpos"][:]
            times = f["time_seconds"][:]
            if (
                not np.isfinite(q).all()
                or not np.isfinite(times).all()
                or (np.diff(times) <= 0).any()
            ):
                raise ValueError("Invalid raw commands/timestamps")
            commands.append(q[:, POLICY_ORDER])
        with np.load(labels / "labels.npz", allow_pickle=False) as data:
            n = e["frames"]
            if (
                data["track"].shape != (n, 10, 2, 2)
                or data["track_valid"].shape != (n, 10, 2)
                or data["stage"].shape != (n,)
                or data["stage_valid"].shape != (n,)
            ):
                raise ValueError("Invalid v4 label shapes")
            if (
                data["track_valid"].dtype != bool
                or data["stage_valid"].dtype != bool
                or not np.issubdtype(data["stage"].dtype, np.integer)
            ):
                raise ValueError("Invalid v4 label dtypes")
            valid_stage = data["stage"][data["stage_valid"]]
            if (valid_stage < 0).any() or (valid_stage >= len(VOCAB[e["task"]])).any():
                raise ValueError("Invalid task-local stage label")
            if not np.isfinite(data["track"][data["track_valid"]]).all():
                raise ValueError("Nonfinite valid track")
            if not np.allclose(data["track_offsets_seconds"], np.arange(1, 11) * 0.3):
                raise ValueError("Expected 10 track points at 0.3-second offsets")
    if validation_per_task < 1 or not groups:
        raise ValueError("Nonempty data and positive validation count required")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for task in sorted(groups):
        indices = rng.permutation(groups[task]).tolist()
        if len(indices) <= validation_per_task:
            raise ValueError("Need training episodes after validation split")
        validation.extend(indices[:validation_per_task])
        train.extend(indices[validation_per_task:])
    q = np.concatenate([commands[i] for i in train])
    return dict(
        schema="td-vla-training-split-v1",
        dataset_id=manifest["dataset_id"],
        manifest_sha256=sha256(root / "manifest.json"),
        seed=seed,
        episodes=episodes,
        train=sorted(train),
        validation=sorted(validation),
        min=q.min(0).tolist(),
        max=q.max(0).tolist(),
        mean=q.mean(0).tolist(),
        normalization="train-only joint min/max; constant joint=0; gripper binary >0.49",
        policy_order=POLICY_ORDER,
        horizon=50,
        track_points=10,
        track_offsets_seconds=(np.arange(1, 11) * 0.3).tolist(),
        track_coordinates="original image u/W,v/H; no clipping or geometric augmentation",
        vocab={task: VOCAB[task] for task in sorted(groups)},
    )


class MultiStreamDataset:
    def __init__(self, root, split, partition="train", image_size=224):
        self.root, self.split = Path(root), split
        if split["schema"] != "td-vla-training-split-v1":
            raise ValueError("Unsupported split schema")
        if sha256(self.root / "manifest.json") != split["manifest_sha256"]:
            raise ValueError("Selected manifest changed")
        manifest = json.loads((self.root / "manifest.json").read_text())
        expected = sorted(manifest["episodes"], key=lambda e: (e["task"], e["seed"]))
        if split["episodes"] != expected:
            raise ValueError("Split episodes differ from selected manifest")
        train, validation = split["train"], split["validation"]
        indices = train + validation
        if (
            partition not in {"train", "validation"}
            or not train
            or not validation
            or any(type(i) is not int for i in indices)
            or len(set(indices)) != len(indices)
            or set(indices) != set(range(len(expected)))
        ):
            raise ValueError("Invalid or overlapping split partition")
        if split["policy_order"] != POLICY_ORDER:
            raise ValueError("Unsupported action dimension order")
        self.episodes = [split["episodes"][i] for i in split[partition]]
        self.ends = np.cumsum([e["frames"] for e in self.episodes]).tolist()
        self.image_size = image_size
        self.labels = {}
        self.task_indices = {}
        start = 0
        for e in self.episodes:
            self.task_indices.setdefault(e["task"], []).extend(range(start, start + e["frames"]))
            start += e["frames"]

    def __len__(self):
        return self.ends[-1]

    def sample_indices(self, step, rank, batch_size, *, seed=42, fixed=False):
        rng = np.random.default_rng(seed + (0 if fixed else step) * 10007 + rank)
        tasks = sorted(self.task_indices)
        return [
            int(rng.choice(self.task_indices[tasks[int(rng.integers(len(tasks)))]]))
            for _ in range(batch_size)
        ]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self.ends, index)
        e = self.episodes[episode]
        t = index - (self.ends[episode - 1] if episode else 0)
        if episode not in self.labels:
            if (
                sha256(self.root / e["raw_path"]) != e["raw_sha256"]
                or sha256(self.root / e["label_path"] / "labels.npz") != e["labels_sha256"]
            ):
                raise ValueError("Episode checksum changed after split creation")
            with np.load(self.root / e["label_path"] / "labels.npz", allow_pickle=False) as data:
                self.labels[episode] = {
                    k: data[k] for k in ("track", "track_valid", "stage", "stage_valid")
                }
        label = self.labels[episode]
        with h5py.File(self.root / e["raw_path"]) as f:
            images = []
            for camera in ("head_camera", "left_camera", "right_camera"):
                im = Image.fromarray(f[f"rgb/{camera}"][t]).resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
                images.append(np.asarray(im).transpose(2, 0, 1) / 255.0)
            # qpos is small; RGB stays lazy. No simulation Oracle fields enter inputs.
            q = normalize_commands(f["qpos"][:], self.split)
        future, valid = future_indices(t, self.split["horizon"], len(q))
        images = np.asarray(images) - np.array([0.485, 0.456, 0.406])[None, :, None, None]
        images /= np.array([0.229, 0.224, 0.225])[None, :, None, None]
        return dict(
            images=images.astype(np.float32),
            state=q[t],
            instruction=INSTRUCTIONS[e["task"]],
            task=e["task"],
            stage_count=len(VOCAB[e["task"]]),
            targets=dict(
                actions=q[future],
                action_valid=valid,
                track=label["track"][t],
                track_valid=label["track_valid"][t],
                stage=np.int64(label["stage"][t]),
                stage_valid=label["stage_valid"][t],
            ),
        )
