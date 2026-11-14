"""Versioned sparse Track and offline Oracle event segmentation for clean pilots."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from td_vla.annotation.geometry import future_time_indices, project, transform_point
from td_vla.annotation.stages import VOCAB, event_flags, scene_features, segment_events

SCHEMA = "pilot-labels-v4"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def annotate(path, output, offsets, confirm_seconds=0.12, min_depth=1e-4):
    with h5py.File(path, "r") as f:
        if not f.attrs["expert_success"]:
            return dict(
                task=str(f.attrs["task"]), seed=int(f.attrs["seed"]), status="skipped_failed_expert"
            )
        output.mkdir(parents=True, exist_ok=False)
        task = str(f.attrs["task"])
        times, poses, q = f["time_seconds"][:], f["object_pose"][:, :, :3], f["qpos"][:]
        geometry_path = Path(__file__).resolve().parents[1] / "configs/pilot_geometry.json"
        geometry = json.loads(geometry_path.read_text())
        evidence = []
        for t in range(len(times)):
            s = scene_features(
                task,
                poses[t],
                poses[0],
                f["finger_contact"][t],
                q[t],
                f["object_contact"][t],
                f["success"][t],
                previous=poses[t - 1] if t else None,
                dt=float(times[t] - times[t - 1]) if t else 0.06,
            )
            if task == "beat_block_hammer":
                head = transform_point(f["object_pose"][t, 0], geometry["hammer_head"]["local_xyz"])
                s["near_target"] = bool(np.linalg.norm(head[:2] - poses[t, 1, :2]) < 0.04)
            evidence.append(s)
        flags, names = event_flags(task, evidence)
        issue = None
        boundaries = []
        try:
            stages, boundaries = segment_events(times, flags, names, confirm_seconds)
            stage_valid = np.ones(len(times), dtype=bool)
        except ValueError as error:
            issue = str(error)
            stages = np.full(len(times), -1, dtype=np.int64)
            stage_valid = np.zeros(len(times), dtype=bool)
        (output / "boundaries.json").write_text(
            json.dumps(
                dict(
                    status="review_required" if issue else "ok",
                    issue=issue,
                    boundaries=boundaries,
                    evidence=[
                        dict(
                            event=b["event"],
                            start=evidence[b["frame"]],
                            confirmed=evidence[b["confirmed_frame"]],
                        )
                        for b in boundaries
                    ],
                    candidate_frames={
                        name: np.flatnonzero(flags[:, i]).tolist() for i, name in enumerate(names)
                    },
                ),
                indent=2,
            )
            + "\n"
        )
        mid = f["midpoint_world"][:]
        h, w = f["rgb/head_camera"].shape[1:3]
        tracks, projections, in_frames, indices, temporals, requested = [], [], [], [], [], []
        for t in range(len(times)):
            idx, temporal, target_times = future_time_indices(times, t, offsets)
            uv, _, valid = project(
                mid[idx],
                f["intrinsic"][t],
                f["extrinsic"][t],
                w,
                h,
                include_offscreen=True,
                min_depth=min_depth,
            )
            in_frame = (
                valid & (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)
            )
            # Sanitize mathematically invalid coordinates, never clip offscreen targets.
            tracks.append(np.where(valid[..., None], uv / [w, h], 0))
            projections.append(valid)
            in_frames.append(in_frame)
            indices.append(idx)
            temporals.append(temporal)
            requested.append(target_times)
        indices, temporal = np.asarray(indices), np.asarray(temporals)
        projection_valid, in_frame = np.asarray(projections), np.asarray(in_frames)
        track = np.asarray(tracks, dtype=np.float32)
        valid = projection_valid & temporal[..., None]
        actual_times = times[indices]
        np.savez_compressed(
            output / "labels.npz",
            stage=stages,
            stage_valid=stage_valid,
            track=track,
            track_valid=valid,
            projection_valid=projection_valid,
            in_frame=in_frame,
            future_index=indices,
            temporal_valid=temporal,
            target_time_seconds=requested,
            sample_time_seconds=actual_times,
            track_offsets_seconds=offsets,
        )
        report = dict(
            schema=SCHEMA,
            task=task,
            seed=int(f.attrs["seed"]),
            frames=len(times),
            status="review_required" if issue else "ok",
            stage_issue=issue,
            vocab=VOCAB[task],
            boundaries=boundaries,
            stage_counts=np.bincount(stages[stage_valid], minlength=len(VOCAB[task])).tolist(),
            raw_sha256=sha256(path),
            offsets_seconds=offsets.tolist(),
            confirm_seconds=confirm_seconds,
            min_depth_m=min_depth,
            valid_track_points=int(valid.sum()),
            offscreen_supervised_points=int((valid & ~in_frame).sum()),
            invalid_projection_real_points=int((~projection_valid & temporal[..., None]).sum()),
            padding_time_slots=int((~temporal).sum()),
            max_sampling_error_seconds=float(
                np.max(np.abs(actual_times - np.asarray(requested))[temporal])
            )
            if temporal.any()
            else None,
            coordinate_min=track[valid].min(axis=0).tolist() if valid.any() else None,
            coordinate_max=track[valid].max(axis=0).tolist() if valid.any() else None,
        )
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        # One compact review artifact per episode; 3 neighboring views per event.
        centers = [b["frame"] for b in boundaries] or np.linspace(0, len(times) - 1, 4, dtype=int)
        sheet = Image.new("RGB", (w * 3, h * len(centers)), "white")
        for row, center in enumerate(centers):
            for col, t in enumerate(np.clip([center - 2, center, center + 2], 0, len(times) - 1)):
                im = Image.fromarray(f["rgb/head_camera"][t])
                draw = ImageDraw.Draw(im)
                for arm, color in enumerate(("lime", "magenta")):
                    points = track[t, :, arm] * [w, h]
                    for a, b, ok in zip(
                        points[:-1], points[1:], valid[t, :-1, arm] & valid[t, 1:, arm]
                    ):
                        if ok:
                            draw.line([tuple(a), tuple(b)], fill=color, width=2)
                draw.text(
                    (4, 4),
                    f"{task} t={t} stage={stages[t]}",
                    fill="black",
                    stroke_width=1,
                    stroke_fill="white",
                )
                sheet.paste(im, (col * w, row * h))
        sheet.save(output / "review.jpg")
        return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--track-seconds", type=float, default=3.0)
    p.add_argument("--track-points", type=int, default=10)
    p.add_argument("--confirm-seconds", type=float, default=0.12)
    p.add_argument("--min-depth", type=float, default=1e-4)
    a = p.parse_args()
    if (
        not np.isfinite([a.track_seconds, a.confirm_seconds, a.min_depth]).all()
        or min(a.track_seconds, a.confirm_seconds, a.min_depth) <= 0
        or a.track_points < 1
    ):
        p.error("Time, point count and depth parameters must be finite and positive")
    paths = sorted(a.data.glob("*/*.h5"))
    paths = [path for path in paths if ".partial." not in path.name]
    if not paths:
        p.error("No input episodes")
    a.output.mkdir(parents=True, exist_ok=False)
    offsets = np.linspace(a.track_seconds / a.track_points, a.track_seconds, a.track_points)
    reports = [
        annotate(
            path, a.output / path.parent.name / path.stem, offsets, a.confirm_seconds, a.min_depth
        )
        for path in paths
    ]
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        root / "src/td_vla/annotation/geometry.py",
        root / "src/td_vla/annotation/stages.py",
        root / "configs/pilot_geometry.json",
    ]
    summary = dict(
        schema=SCHEMA,
        source_sha256={str(s.relative_to(root)): sha256(s) for s in sources},
        sampling="nearest_timestamp_ties_earlier",
        reports=reports,
    )
    (a.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(a.output),
                ok=sum(r["status"] == "ok" for r in reports),
                review_required=sum(r["status"] == "review_required" for r in reports),
                skipped=sum(r["status"] == "skipped_failed_expert" for r in reports),
            )
        )
    )


if __name__ == "__main__":
    main()
