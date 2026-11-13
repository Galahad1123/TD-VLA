"""Headless SAPIEN smoke test; fails if rendered RGB is empty or constant."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import sapien

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene()
scene.add_ground(0)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [1, 1, 1])
builder = scene.create_actor_builder()
builder.add_box_visual(half_size=[0.1, 0.1, 0.1], color=[1, 0, 0])
box = builder.build_static(name="test-box")
box.set_pose(sapien.Pose([1, 0, 0.1]))
camera = scene.add_camera("test", 320, 240, 1.0, 0.01, 10)
camera.set_pose(sapien.Pose([0, 0, 0.2]))
scene.step()
scene.update_render()
camera.take_picture()
rgb = camera.get_picture("Color")[..., :3]
assert np.isfinite(rgb).all() and float(rgb.std()) > 0.01
Image.fromarray((rgb.clip(0, 1) * 255).astype(np.uint8)).save(args.output / "render.png")
print({"render": "pass", "shape": rgb.shape, "std": float(rgb.std())})
