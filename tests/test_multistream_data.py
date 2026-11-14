import hashlib
import json

import h5py
import numpy as np
import pytest

from td_vla.data.multistream import MultiStreamDataset, prepare_split


def fixture_dataset(tmp_path):
    episodes = []
    for seed in range(4):
        raw = tmp_path / f"raw/{seed}.h5"
        raw.parent.mkdir(exist_ok=True)
        labels = tmp_path / f"labels/{seed}"
        labels.mkdir(parents=True)
        with h5py.File(raw, "w") as f:
            f.attrs.update(task="lift_pot", seed=seed, expert_success=True)
            f["qpos"] = np.full((3, 14), seed, np.float32)
            f["time_seconds"] = [0.0, 0.06, 0.12]
            for camera in ("head_camera", "left_camera", "right_camera"):
                f[f"rgb/{camera}"] = np.full((3, 8, 8, 3), seed, np.uint8)
        np.savez(
            labels / "labels.npz",
            stage=[0, 1, 2],
            stage_valid=[True] * 3,
            track=np.full((3, 10, 2, 2), 1.5, np.float32),
            track_valid=np.ones((3, 10, 2), bool),
            track_offsets_seconds=np.arange(1, 11) * 0.3,
        )
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        (labels / "report.json").write_text(
            json.dumps(
                dict(
                    schema="pilot-labels-v4",
                    task="lift_pot",
                    seed=seed,
                    frames=3,
                    raw_sha256=digest,
                    vocab=["approach_grasp", "bilateral_hold_lift", "complete"],
                )
            )
        )
        episodes.append(
            dict(
                task="lift_pot",
                seed=seed,
                frames=3,
                raw_path=str(raw.relative_to(tmp_path)),
                label_path=str(labels.relative_to(tmp_path)),
                raw_sha256=digest,
                labels_sha256=hashlib.sha256((labels / "labels.npz").read_bytes()).hexdigest(),
            )
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                schema="td-vla-selected-episodes-v1",
                annotation_schema="pilot-labels-v4",
                dataset_id="test",
                episodes=episodes,
            )
        )
    )
    return tmp_path


def test_split_and_sample_contract(tmp_path):
    root = fixture_dataset(tmp_path)
    split = prepare_split(root, validation_per_task=1)
    assert set(split["train"]).isdisjoint(split["validation"])
    assert len(split["train"]) == 3
    expected_mean = np.mean([split["episodes"][i]["seed"] for i in split["train"]])
    np.testing.assert_allclose(split["mean"], expected_mean)
    dataset = MultiStreamDataset(root, split, "train", image_size=16)
    sample = dataset[2]  # final frame of first episode
    assert sample["images"].shape == (3, 3, 16, 16)
    assert set(sample) == {"images", "state", "instruction", "targets", "task", "stage_count"}
    assert not sample["targets"]["action_valid"].any()
    assert sample["targets"]["track"].max() > 1  # no clipping of offscreen tracks
    assert sample["targets"]["stage"] == 2
    assert sample["stage_count"] == 3
    assert prepare_split(root, validation_per_task=1) == split


def test_split_rejects_tampered_labels(tmp_path):
    root = fixture_dataset(tmp_path)
    with (root / "labels/0/labels.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        prepare_split(root, validation_per_task=1)


def test_split_rejects_duplicate_episode(tmp_path):
    root = fixture_dataset(tmp_path)
    path = root / "manifest.json"
    data = json.loads(path.read_text())
    data["episodes"].append(data["episodes"][0])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_split(root, validation_per_task=1)


def test_dataset_rejects_overlapping_split(tmp_path):
    root = fixture_dataset(tmp_path)
    split = prepare_split(root, validation_per_task=1)
    split["validation"] = [split["train"][0]]
    with pytest.raises(ValueError, match="partition"):
        MultiStreamDataset(root, split)


def test_dataset_rechecks_actual_labels_after_split(tmp_path):
    root = fixture_dataset(tmp_path)
    split = prepare_split(root, validation_per_task=1)
    e = split["episodes"][split["train"][0]]
    with (root / e["label_path"] / "labels.npz").open("ab") as file:
        file.write(b"modified after split")
    dataset = MultiStreamDataset(root, split)
    with pytest.raises(ValueError, match="checksum"):
        dataset[0]


def test_normalization_order_and_roundtrip():
    from td_vla.data.multistream import normalize_commands, denormalize_actions, POLICY_ORDER

    raw = np.arange(14, dtype=np.float32)
    raw[[6, 13]] = [0, 1]
    split = {"min": [-20] * 14, "max": [20] * 14}
    normalized = normalize_commands(raw, split)
    np.testing.assert_allclose(normalized[:12], raw[POLICY_ORDER][:12] / 20, atol=1e-6)
    np.testing.assert_allclose(denormalize_actions(normalized, split), raw, atol=1e-6)
