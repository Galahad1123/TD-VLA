"""Derive auditable scene-state labels and future tracks; never modify raw rollouts."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from td_vla.annotation.geometry import future_indices, project, resize_pixels
from td_vla.annotation.stages import VOCAB, classify, stabilize


def annotate(path, output):
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as f:
        if not f.attrs["expert_success"]:
            return {"path": str(path), "skipped": "failed expert"}
        task = f.attrs["task"]
        poses, q = f["object_pose"][:, :, :3], f["qpos"][:]
        contacts = f["finger_contact"][:]
        # Both finger pads in contact and not open. Contact data is label-only.
        held = contacts.all(-1) & (q[:, None, [6, 13]] < 0.8)
        n = len(q)
        raw, evidence = [], []
        for t in range(n):
            obj = poses[t, 0]
            near = np.linalg.norm(obj[:2] - poses[t, 1, :2]) < 0.04 if poses.shape[1] > 1 else False
            features = dict(
                success=bool(f["success"][t]),
                held=held[t, 0].tolist(),
                lifted=bool(obj[2] - poses[0, 0, 2] > 0.025),
                near_target=bool(near),
                object_in_basket=False,
                basket_held=False,
                first_placed=False,
                second_held=False,
                ambiguous=bool(np.any(contacts[t, 0].sum(-1) == 1)),
            )
            if task == "stack_blocks_two":
                features.update(
                    first_placed=bool(
                        np.linalg.norm(obj[:2] - [0, -0.13]) < 0.035 and not held[t, 0].any()
                    ),
                    second_held=bool(held[t, 1].any()),
                )
            if task == "place_object_basket":
                features.update(
                    object_in_basket=bool(
                        f["object_contact"][t, 0, 1] and np.linalg.norm(obj - poses[t, 1]) < 0.15
                    ),
                    basket_held=bool(held[t, 1].any()),
                )
            raw.append(classify(task, features))
            evidence.append(features)
        stages, stage_valid = stabilize(raw, len(VOCAB[task]) - 1)
        mid = f["midpoint_world"][:]
        h, w = f["rgb/head_camera"].shape[1:3]
        tracks, track_valid, indices, temporal, current_uv, occluded = [], [], [], [], [], []
        transform_errors = []
        for t in range(n):
            idx, mask = future_indices(t, 50, n)
            uv, z, valid = project(mid[idx], f["intrinsic"][t], f["extrinsic"][t], w, h)
            now_uv, now_z, now_valid = project(mid[t], f["intrinsic"][t], f["extrinsic"][t], w, h)
            # Independent OpenGL camera transform -> CV consistency check.
            cv = np.diag([1, -1, -1, 1]) @ np.linalg.inv(f["cam2world_gl"][t])
            transform_errors.append(float(np.abs(cv[:3] - f["extrinsic"][t]).max()))
            occ = np.zeros(2, dtype=bool)
            for arm in range(2):
                if now_valid[arm]:
                    x, y = np.rint(now_uv[arm]).astype(int)
                    d = f["depth_m"][t, np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
                    occ[arm] = 0 < d < now_z[arm] - 0.02
            current_uv.append(now_uv)
            occluded.append(occ)
            tracks.append(uv / [w, h])
            track_valid.append(valid & mask[:, None])
            indices.append(idx)
            temporal.append(mask)
        np.savez_compressed(
            output / "labels.npz",
            stage=np.asarray(stages),
            stage_valid=stage_valid,
            track=np.asarray(tracks, dtype=np.float32),
            track_valid=track_valid,
            future_index=indices,
            temporal_valid=temporal,
            current_uv=current_uv,
            current_occluded=occluded,
        )
        (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        # Include stage changes and evenly sampled frames; no image-based claim without inspection.
        changes = np.flatnonzero(np.diff(stages)) + 1
        selected = sorted(set(np.linspace(0, n - 1, 8, dtype=int).tolist() + changes.tolist()))
        thumbnails = []
        for t in selected:
            image = Image.fromarray(f["rgb/head_camera"][t])
            draw = ImageDraw.Draw(image)
            for arm, color in enumerate(("lime", "magenta")):
                points = np.asarray(tracks[t])[:, arm] * [w, h]
                valid = np.asarray(track_valid[t])[:, arm]
                for a, b, ok in zip(points[:-1], points[1:], valid[:-1] & valid[1:]):
                    if ok:
                        draw.line([tuple(a), tuple(b)], fill=color, width=2)
                uv = current_uv[t][arm]
                if np.isfinite(uv).all():
                    x, y = uv
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
            draw.text(
                (4, 4),
                f"{t}: {stages[t]} valid={stage_valid[t]}",
                fill="black",
                stroke_width=1,
                stroke_fill="white",
            )
            image.save(output / f"frame-{t:04d}.png")
            thumbnails.append(image)
        sheet = Image.new("RGB", (w * 4, h * ((len(thumbnails) + 3) // 4)), "white")
        for i, image in enumerate(thumbnails):
            sheet.paste(image, ((i % 4) * w, (i // 4) * h))
        sheet.save(output / "overview.jpg")
        # Check image-input coordinate transform explicitly with a 224x224 overlay.
        image = Image.fromarray(f["rgb/head_camera"][n // 2]).resize(
            (224, 224), Image.Resampling.BILINEAR
        )
        draw = ImageDraw.Draw(image)
        for uv, color in zip(
            resize_pixels(current_uv[n // 2], (w, h), (224, 224)), ("lime", "magenta")
        ):
            if np.isfinite(uv).all():
                x, y = uv
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
        image.save(output / "model-input-overlay.png")
        report = dict(
            task=task,
            seed=int(f.attrs["seed"]),
            frames=n,
            schema="pilot-labels-v1",
            raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            vocab=VOCAB[task],
            stage_counts=np.bincount(stages, minlength=len(VOCAB[task])).tolist(),
            valid_stage_fraction=float(np.mean(stage_valid)),
            track_valid_fraction=float(np.mean(track_valid)),
            current_occluded_fraction=np.mean(occluded, axis=0).tolist(),
            stage_changes=changes.tolist(),
            terminal_saved=bool(f["success"][-1]),
            physics_step_intervals=np.unique(np.diff(f["physics_step"][:])).tolist(),
            camera_transform_max_error=max(transform_errors),
            tcp_midpoint_distance_m=np.mean(
                np.linalg.norm(f["tcp_world"][:] - mid, axis=-1), axis=0
            ).tolist(),
        )
        if report["camera_transform_max_error"] > 1e-5:
            raise RuntimeError("CV/GL camera transform mismatch")
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    reports = [
        annotate(path, a.output / path.parent.name / path.stem)
        for path in sorted(a.data.glob("*/*.h5"))
        if ".partial." not in path.name
    ]
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "reports.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports))


if __name__ == "__main__":
    main()
