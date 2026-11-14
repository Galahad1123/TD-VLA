"""Record fixed-rate Clean expert rollouts without changing expert actions or physics."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN = ROOT / "external/RoboTwin"
sys.path.insert(0, str(ROBOTWIN))
CAMERAS = ("head_camera", "left_camera", "right_camera")
OBJECTS = {
    "move_stapler_pad": ("stapler", "pad"), "lift_pot": ("pot",),
    "handover_block": ("box", "target_box"), "beat_block_hammer": ("hammer", "block"),
    "stack_blocks_two": ("block1", "block2"), "place_object_basket": ("object", "basket"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finger_geometry(robot_path):
    """Center of distal 5 mm inner pad bounding box, in each finger link frame."""
    result = []
    sources = []
    for index, side in ((7, -1), (8, 1)):
        path = robot_path / f"urdf/aloha_maniskill_sim/meshes/link{index}.STL"
        vertices = np.asarray(trimesh.load(path).vertices)
        distal = vertices[vertices[:, 0] >= vertices[:, 0].max() - 0.005]
        lo, hi = distal.min(0), distal.max(0)
        point = (lo + hi) / 2
        point[1] = lo[1] if side == -1 else hi[1]
        result.append(point)
        sources.append({"mesh": str(path.relative_to(robot_path)), "sha256": digest(path),
                        "point_local": point.tolist()})
    return np.asarray(result), sources


class Recorder:
    def __init__(self, task, name, file, robot_path):
        self.task, self.name, self.file = task, name, file
        self.scene = task.scene
        self.step_index = 0
        self.last_recorded = -1
        self.local_points, geometry = finger_geometry(robot_path)
        self.file.attrs["finger_geometry"] = json.dumps(geometry)
        self.file.attrs["object_names"] = json.dumps(OBJECTS[name])
        self.links = [[entity.find_link_by_name(f"{prefix}_link{i}") for i in (7, 8)]
                      for entity, prefix in ((task.robot.left_entity, "fl"),
                                             (task.robot.right_entity, "fr"))]
        self.contact_since_frame = set()

    def __getattr__(self, name):
        return getattr(self.scene, name)

    def step(self):
        self.scene.step()
        self.step_index += 1
        if self.step_index % 15 == 0:
            self.capture()

    def append(self, key, value):
        value = np.asarray(value)
        if key not in self.file:
            self.file.create_dataset(key, shape=(0, *value.shape), maxshape=(None, *value.shape),
                                     chunks=(1, *value.shape), dtype=value.dtype,
                                     compression="lzf" if value.ndim else None)
        ds = self.file[key]
        ds.resize(len(ds) + 1, axis=0)
        ds[-1] = value

    def capture(self):
        if self.last_recorded == self.step_index:
            return
        task = self.task
        obs = task.get_obs()
        values = {"physics_step": np.int64(self.step_index),
                  "time_seconds": np.float64(self.step_index * self.scene.get_timestep()),
                  "qpos": np.asarray(obs["joint_action"]["vector"], dtype=np.float32),
                  "success": np.bool_(task.check_success())}
        for name in CAMERAS:
            values[f"rgb/{name}"] = obs["observation"][name]["rgb"]
        camera = obs["observation"]["head_camera"]
        values["intrinsic"] = camera["intrinsic_cv"]
        values["extrinsic"] = camera["extrinsic_cv"]
        values["cam2world_gl"] = camera["cam2world_gl"]
        # SAPIEN Position uses OpenGL coordinates; depth in metres, not uint16 mm.
        head = task.cameras.static_camera_list[task.cameras.static_camera_name.index("head_camera")]
        values["depth_m"] = -head.get_picture("Position")[..., 2]
        fingers = np.asarray([[link.get_pose().to_transformation_matrix()[:3, :3] @ point
                               + link.get_pose().p for link, point in zip(links, self.local_points)]
                              for links in self.links])
        values["fingers_world"] = fingers
        values["midpoint_world"] = fingers.mean(axis=1)
        values["tcp_world"] = np.asarray([task.robot.get_left_tcp_pose()[:3],
                                           task.robot.get_right_tcp_pose()[:3]])
        names = OBJECTS[self.name]
        values["object_pose"] = np.asarray([np.r_[getattr(task, n).get_pose().p,
                                                  getattr(task, n).get_pose().q] for n in names])
        bodies = []
        for n in names:
            obj = getattr(task, n).actor
            if hasattr(obj, "get_links"):
                bodies.append({link.entity.per_scene_id for link in obj.get_links()})
            else:
                bodies.append({obj.per_scene_id})
        contacts = self.scene.get_contacts()
        pairs = [{c.bodies[0].entity.per_scene_id, c.bodies[1].entity.per_scene_id} for c in contacts]
        finger_contact = np.zeros((len(names), 2, 2), dtype=bool)
        for o, ids in enumerate(bodies):
            for arm, links in enumerate(self.links):
                for finger, link in enumerate(links):
                    finger_contact[o, arm, finger] = any(link.entity.per_scene_id in pair and bool(pair & ids)
                                                        for pair in pairs)
        values["finger_contact"] = finger_contact
        object_contact = np.zeros((len(names), len(names)), dtype=bool)
        for i, ids in enumerate(bodies):
            for j, others in enumerate(bodies):
                if i != j:
                    object_contact[i, j] = any(bool(p & ids) and bool(p & others) for p in pairs)
        values["object_contact"] = object_contact
        for key, value in values.items():
            self.append(key, value)
        self.last_recorded = self.step_index
        if len(self.file["qpos"]) % 50 == 0:
            self.file.flush()
            print(json.dumps({"frames": len(self.file["qpos"]), "step": self.step_index}), flush=True)


def run(name, seed, root):
    out = root / name / f"seed-{seed:04d}.h5"
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.h5")
    if partial.exists():
        raise FileExistsError(partial)
    cfg = yaml.safe_load((ROBOTWIN / "task_config/demo_clean.yml").read_text())
    robot_path = ROBOTWIN / "assets/embodiments/aloha-agilex"
    embodiment = yaml.safe_load((robot_path / "config.yml").read_text())
    cfg.update(task_name=name, task_config="demo_clean", need_plan=True,
               left_robot_file=str(robot_path), right_robot_file=str(robot_path),
               left_embodiment_config=embodiment, right_embodiment_config=embodiment,
               dual_arm_embodied=True, embodiment_name="aloha-agilex", render_freq=0,
               save_path=str(root), save_data=False, collect_data=False)
    task = getattr(importlib.import_module(f"envs.{name}"), name)()
    try:
        task.setup_demo(now_ep_num=seed, seed=seed, **cfg)
        with h5py.File(partial, "w") as file:
            file.attrs.update(task=name, seed=seed, schema="td-vla-pilot-raw-v1", clean=True,
                              git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                              clean_config_sha256=digest(ROBOTWIN / "task_config/demo_clean.yml"))
            recorder = Recorder(task, name, file, robot_path)
            task.scene = recorder
            recorder.capture()
            task.play_once()
            recorder.capture()  # Save terminal state even off the sampling grid.
            success = bool(task.plan_success and task.check_success())
            file.attrs["expert_success"] = success
            file.attrs["info"] = json.dumps(task.info, default=str)
            frames = len(file["qpos"])
        partial.replace(out)
        record = {"task": name, "seed": seed, "success": success, "frames": frames,
                  "path": str(out.relative_to(root)), "sha256": digest(out)}
        with (root / "manifest.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    finally:
        task.close_env()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=OBJECTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(ROBOTWIN)
    run(args.task, args.seed, root)


if __name__ == "__main__":
    main()
