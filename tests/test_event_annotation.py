import numpy as np
import pytest
from td_vla.annotation import geometry, stages


def test_track_keeps_offscreen_and_rejects_singular_projection():
    k = np.array([[100, 0, 160], [0, 100, 120], [0, 0, 1.0]])
    uv, _, valid = geometry.project(
        [[5, 0, 1], [0, 0, -1], [0, 0, 1e-8]],
        k,
        np.eye(4),
        320,
        240,
        include_offscreen=True,
        min_depth=1e-4,
    )
    assert valid.tolist() == [True, False, False]
    assert uv[0, 0] / 320 > 1


def test_sparse_times_do_not_turn_short_terminal_interval_into_valid_padding():
    indices, valid, requested = geometry.future_time_indices(
        np.array([0.0, 0.06, 0.12, 0.14]), 0, [0.1, 0.14, 0.15]
    )
    assert indices.tolist() == [2, 3, 3]
    assert valid.tolist() == [True, True, False]
    np.testing.assert_allclose(requested, [0.1, 0.14, 0.15])


def test_event_intervals_survive_contact_flicker_and_use_start_of_confirmation():
    times = np.arange(10) * 0.06
    flags = np.array(
        [[0, 0], [1, 0], [1, 0], [1, 0], [0, 0], [1, 0], [0, 0], [0, 1], [0, 1], [0, 1]], dtype=bool
    )
    labels, boundaries = stages.segment_events(times, flags, ["grasp", "complete"], 0.12)
    assert labels.tolist() == [0, 1, 1, 1, 1, 1, 1, 2, 2, 2]
    assert [x["frame"] for x in boundaries] == [1, 7]
    assert boundaries[0]["confirmed_frame"] == 3


def test_missing_event_does_not_fabricate_progress_from_success():
    with pytest.raises(ValueError, match="grasp"):
        stages.segment_events(
            np.arange(5) * 0.06, [[0, 0]] * 4 + [[0, 1]], ["grasp", "complete"], 0.12
        )


def test_reversed_events_are_rejected():
    with pytest.raises(ValueError):
        stages.segment_events(
            np.arange(8) * 0.06, [[0, 1]] * 3 + [[1, 0]] * 5, ["grasp", "complete"], 0.12
        )


def test_touching_basket_while_still_holding_object_is_not_a_transition():
    base = dict(held=[True, False], object_in_basket=True, basket_held=True, success=False)
    flags, names = stages.event_flags("place_object_basket", [base])
    assert flags.tolist() == [[True, False, False, False]]
    assert len(names) == 4


@pytest.mark.parametrize("offsets,depth", [([2.0], 1.0), ([0.06], -1.0)])
def test_annotation_handles_no_valid_track_points(tmp_path, offsets, depth):
    import h5py
    import runpy
    from pathlib import Path

    annotate = runpy.run_path(str(Path(__file__).parents[1] / "scripts/annotate_pilot.py"))[
        "annotate"
    ]
    raw = tmp_path / "episode.h5"
    with h5py.File(raw, "w") as f:
        f.attrs.update(expert_success=True, task="lift_pot", seed=0)
        f["time_seconds"] = np.arange(12) * 0.06
        f["object_pose"] = np.tile([[[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]]], (12, 1, 1))
        f["qpos"] = np.zeros((12, 14))
        contacts = np.zeros((12, 1, 2, 2), dtype=bool)
        contacts[2:] = True
        f["finger_contact"] = contacts
        f["object_contact"] = np.zeros((12, 1, 1), dtype=bool)
        f["success"] = np.arange(12) >= 10
        f["midpoint_world"] = np.tile([[0.0, 0.0, depth], [0.0, 0.0, depth]], (12, 1, 1))
        f["rgb/head_camera"] = np.zeros((12, 24, 32, 3), dtype=np.uint8)
        f["intrinsic"] = np.tile(np.eye(3), (12, 1, 1))
        f["extrinsic"] = np.tile(np.eye(4), (12, 1, 1))
    report = annotate(raw, tmp_path / "labels", np.array(offsets))
    assert report["status"] == "ok"
    assert report["valid_track_points"] == 0
    assert report["coordinate_min"] is None
    z = np.load(tmp_path / "labels/labels.npz")
    assert not z["track_valid"].any()
    assert np.isfinite(z["track"]).all()
    assert z["stage_valid"].all()
    if offsets == [2.0]:
        assert report["max_sampling_error_seconds"] is None
