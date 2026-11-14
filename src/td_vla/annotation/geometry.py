"""CV-camera projection. Pixels use integer-center coordinates; no image clipping."""

import numpy as np


def project(
    points, intrinsic, world_to_cv, width, height, *, include_offscreen=False, min_depth=0.0
):
    points = np.asarray(points, dtype=np.float64)
    ext = np.asarray(world_to_cv, dtype=np.float64)
    camera = points @ ext[:3, :3].T + ext[:3, 3]
    z = camera[..., 2]
    homogeneous = camera @ np.asarray(intrinsic).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = homogeneous[..., :2] / homogeneous[..., 2:3]
    valid = np.isfinite(uv).all(-1) & (z > min_depth)
    if not include_offscreen:
        valid &= (
            (uv[..., 0] >= 0) & (uv[..., 0] < width) & (uv[..., 1] >= 0) & (uv[..., 1] < height)
        )
    return uv, z, valid


def resize_pixels(uv, source_size, target_size):
    return (np.asarray(uv) + 0.5) * (np.asarray(target_size) / source_size) - 0.5


def future_indices(current, horizon, length):
    if not 0 <= current < length or horizon < 1:
        raise ValueError("Invalid window")
    raw = current + 1 + np.arange(horizon)
    return np.minimum(raw, length - 1), raw < length


def transform_point(pose, local_point):
    """Transform a local point with a SAPIEN xyz + wxyz pose."""
    pose = np.asarray(pose, dtype=np.float64)
    q = pose[3:] / np.linalg.norm(pose[3:])
    w, x, y, z = q
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rotation @ np.asarray(local_point) + pose[:3]


def future_time_indices(times, current, offsets):
    """Nearest timestamp (ties choose earlier); padding validity uses requested time."""
    times, offsets = np.asarray(times, dtype=float), np.asarray(offsets, dtype=float)
    if (
        times.ndim != 1
        or len(times) == 0
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or not 0 <= current < len(times)
        or offsets.ndim != 1
        or len(offsets) == 0
        or not np.isfinite(offsets).all()
        or np.any(offsets <= 0)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("Invalid timestamps or future offsets")
    requested = times[current] + offsets
    right = np.minimum(np.searchsorted(times, requested), len(times) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(requested - times[left] <= times[right] - requested, left, right)
    # Tiny configurable offsets must never silently select the current observation.
    indices = np.maximum(indices, min(current + 1, len(times) - 1))
    valid = (requested <= times[-1] + 1e-9) & (indices > current)
    return indices, valid, requested
