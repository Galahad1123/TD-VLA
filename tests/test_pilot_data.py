import h5py
import numpy as np
import pytest
from td_vla.data.pilot import make_split, window


def episode(path, value):
    with h5py.File(path, "w") as f:
        f.attrs.update(expert_success=True, task="lift_pot")
        f.create_dataset("qpos", data=np.full((4, 14), value, dtype=np.float32))
        f.create_dataset("rgb/head_camera", data=np.zeros((4, 2, 2, 3), dtype=np.uint8))


def test_episode_split_statistics_do_not_read_validation(tmp_path):
    task = tmp_path / "lift_pot"
    task.mkdir()
    episode(task / "seed-0000.h5", 2)
    episode(task / "seed-0001.h5", 100)
    split = make_split(tmp_path)
    assert split["train"] == ["lift_pot/seed-0000.h5"]
    assert split["validation"] == ["lift_pot/seed-0001.h5"]
    np.testing.assert_allclose(split["mean"], 2)


def test_window_is_next_command_not_current_and_masks_tail():
    q = np.arange(5 * 14).reshape(5, 14)
    target, mask = window(q, 3, 3)
    np.testing.assert_equal(target, q[[4, 4, 4]])
    assert mask.tolist() == [True, False, False]


def test_split_rejects_single_episode(tmp_path):
    task = tmp_path / "lift_pot"
    task.mkdir()
    episode(task / "seed-0000.h5", 2)
    with pytest.raises(ValueError):
        make_split(tmp_path)


def test_identical_episode_content_cannot_cross_split(tmp_path):
    task = tmp_path / "lift_pot"
    task.mkdir()
    episode(task / "seed-0000.h5", 2)
    (task / "seed-0001.h5").write_bytes((task / "seed-0000.h5").read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        make_split(tmp_path)
