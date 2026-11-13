"""Headless SAPIEN smoke test; fails if rendered RGB is empty or constant."""
import argparse
import ctypes
from pathlib import Path

import numpy as np
from PIL import Image
import sapien

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--ray-tracing", action="store_true")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
if args.ray_tracing:
    # A rendered image alone does not prove that the CUDA denoiser initialized.
    from sapien._oidn_tricks import oidn_dll
    oidn_dll.oidnNewDevice.argtypes = [ctypes.c_int]
    oidn_dll.oidnNewDevice.restype = ctypes.c_void_p
    oidn_dll.oidnCommitDevice.argtypes = [ctypes.c_void_p]
    oidn_dll.oidnGetDeviceError.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)]
    oidn_dll.oidnGetDeviceError.restype = ctypes.c_int
    oidn_dll.oidnReleaseDevice.argtypes = [ctypes.c_void_p]
    device = oidn_dll.oidnNewDevice(3)  # OIDN_DEVICE_TYPE_CUDA
    message = ctypes.c_char_p()
    error = oidn_dll.oidnGetDeviceError(device, ctypes.byref(message))
    if error or not device:
        raise RuntimeError(f"OIDN CUDA device failed: {message.value!r}")
    try:
        oidn_dll.oidnCommitDevice(device)
        error = oidn_dll.oidnGetDeviceError(device, ctypes.byref(message))
        if error:
            raise RuntimeError(f"OIDN CUDA initialization failed: {message.value!r}")
    finally:
        oidn_dll.oidnReleaseDevice(device)
    print("OIDN CUDA device initialized", flush=True)
    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_denoiser("oidn")
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene()
scene.add_ground(0)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [1, 1, 1])
builder = scene.create_actor_builder()
material = sapien.render.RenderMaterial(base_color=[1, 0, 0, 1])
builder.add_box_visual(half_size=[0.1, 0.1, 0.1], material=material)
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
