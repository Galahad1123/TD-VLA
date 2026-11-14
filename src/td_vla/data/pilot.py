"""Small in-memory diagnostic dataset: current RGB/command state -> future commands."""

import hashlib
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from td_vla.annotation.geometry import future_indices

INSTRUCTIONS = {
    "move_stapler_pad": "move the stapler onto the pad.",
    "lift_pot": "lift the pot with both arms.",
    "handover_block": "hand over the red block and place it on the blue pad.",
    "beat_block_hammer": "hit the block with the hammer.",
    "stack_blocks_two": "stack the green block on the red block.",
    "place_object_basket": "put the object in the basket and lift the basket.",
}


def window(qpos, current, horizon=50):
    indices, valid = future_indices(current, horizon, len(qpos))
    return qpos[indices], valid


def make_split(root, tasks=None):
    root = Path(root)
    groups = {}
    fingerprints = {}
    identities = set()
    for path in sorted(root.glob("*/*.h5")):
        if ".partial." in path.name or (tasks and path.parent.name not in tasks):
            continue
        with h5py.File(path) as f:
            if not f.attrs["expert_success"]:
                continue
            task = str(f.attrs["task"])
            identity = (task, int(f.attrs["seed"])) if "seed" in f.attrs else (task, str(path))
            if identity in identities:
                raise ValueError("Duplicate task/seed episode")
            identities.add(identity)
        relative = str(path.relative_to(root))
        groups.setdefault(task, []).append(relative)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in fingerprints.values():
            raise ValueError("Duplicate episode content")
        fingerprints[relative] = digest
    if not groups or (tasks and set(groups) != set(tasks)):
        raise ValueError("Missing requested tasks")
    if any(len(paths) < 2 for paths in groups.values()):
        raise ValueError("Need at least two successful episodes per task")
    train = [path for task in sorted(groups) for path in groups[task][:-1]]
    validation = [groups[task][-1] for task in sorted(groups)]
    q = []
    for path in train:
        with h5py.File(root / path) as f:
            q.append(f["qpos"][:])
    q = np.concatenate(q)
    return dict(
        train=train,
        validation=validation,
        mean=q.mean(0).tolist(),
        std=np.maximum(q.std(0), 0.01).tolist(),
        files=fingerprints,
        normalization="train-only mean/std, floor=0.01",
        horizon=50,
    )


class PilotData:
    def __init__(self, root, split, partition="train"):
        self.episodes = []
        self.mean, self.std = np.asarray(split["mean"]), np.asarray(split["std"])
        for path in split[partition]:
            with h5py.File(Path(root) / path) as f:
                # Explicit allowlist: no stage, geometry, contacts or future RGB enter inputs.
                views = []
                for name in ("head_camera", "left_camera", "right_camera"):
                    views.append(
                        np.stack(
                            [
                                np.asarray(
                                    Image.fromarray(rgb).resize(
                                        (224, 224), Image.Resampling.BILINEAR
                                    )
                                )
                                for rgb in f[f"rgb/{name}"]
                            ]
                        )
                    )
                images = np.stack(views, axis=1)
                self.episodes.append((str(f.attrs["task"]), images, f["qpos"][:]))
        self.tasks = sorted({e[0] for e in self.episodes})

    def batch(self, step, rank, batch_size, seed=42, fixed=False):
        import torch

        rng = np.random.default_rng(seed + (0 if fixed else step) * 10007 + rank)
        images, states, targets, masks, texts = [], [], [], [], []
        for _ in range(batch_size):
            task = self.tasks[int(rng.integers(len(self.tasks)))]
            group = [e for e in self.episodes if e[0] == task]
            name, rgb, q = group[int(rng.integers(len(group)))]
            t = int(rng.integers(len(q) - 1))
            target, valid = window(q, t)
            images.append(rgb[t])
            states.append((q[t] - self.mean) / self.std)
            targets.append((target - self.mean) / self.std)
            masks.append(valid)
            texts.append(INSTRUCTIONS[name])
        image = torch.from_numpy(np.stack(images)).permute(0, 1, 4, 2, 3).float() / 255
        image = (
            image - torch.tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
        ) / torch.tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
        return (
            texts,
            image,
            torch.tensor(np.asarray(states), dtype=torch.float32),
            torch.tensor(np.asarray(targets), dtype=torch.float32),
            torch.tensor(np.asarray(masks)),
        )
