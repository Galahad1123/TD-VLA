import numpy as np
from td_vla.annotation.geometry import project, future_indices, resize_pixels


def test_cv_projection_and_invalid_depth():
    k = np.array([[100, 0, 160], [0, 100, 120], [0, 0, 1.0]])
    uv, depth, valid = project(
        np.array([[0, 0, 1], [1, 1, 2], [0, 0, -1], [5, 0, 1]]), k, np.eye(4), 320, 240
    )
    np.testing.assert_allclose(uv[:2], [[160, 120], [210, 170]])
    assert valid.tolist() == [True, True, False, False]
    assert depth[2] == -1


def test_resize_pixel_centers():
    np.testing.assert_allclose(
        resize_pixels(np.array([[0.0, 0.0], [319, 239]]), (320, 240), (224, 224)),
        [[-0.15, -1 / 30], [223.15, 223 + 1 / 30]],
    )


def test_future_window_excludes_current_and_masks_padding():
    indices, valid = future_indices(2, 5, 4)
    assert indices.tolist() == [3, 3, 3, 3, 3]
    assert valid.tolist() == [True, False, False, False, False]
