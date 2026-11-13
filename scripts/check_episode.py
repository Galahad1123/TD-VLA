"""One successful Clean expert episode; no dataset creation or policy score claim."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN = ROOT / "external/RoboTwin"
sys.path.insert(0, str(ROBOTWIN))
sys.path.insert(0, str(ROOT / "external/TurboVLA/third_party/starvla_runtime"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="move_stapler_pad")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--echo-port", type=int)
    options = parser.parse_args()
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROBOTWIN)
    cfg = yaml.safe_load((ROBOTWIN / "task_config/demo_clean.yml").read_text())
    robot_file = str(ROBOTWIN / "assets/embodiments/aloha-agilex")
    embodiment = yaml.safe_load((Path(robot_file) / "config.yml").read_text())
    cfg.update(task_name=options.task, task_config="demo_clean", need_plan=True,
               left_robot_file=robot_file, right_robot_file=robot_file,
               left_embodiment_config=embodiment, right_embodiment_config=embodiment,
               dual_arm_embodied=True, embodiment_name="aloha-agilex", render_freq=0,
               save_path=str(output), save_data=False, collect_data=False)
    task_class = getattr(importlib.import_module(f"envs.{options.task}"), options.task)
    records = []
    for seed in range(options.seed, options.seed + options.attempts):
        task = task_class()
        try:
            task.setup_demo(now_ep_num=0, seed=seed, **cfg)
            obs = task.get_obs()
            for camera in ("head_camera", "left_camera", "right_camera"):
                rgb = obs["observation"][camera]["rgb"]
                assert np.isfinite(rgb).all() and float(np.std(rgb)) > 0
                Image.fromarray(rgb).save(output / f"seed-{seed}-{camera}.png")
            if options.echo_port is not None:
                from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
                client = WebsocketClientPolicy("127.0.0.1", options.echo_port)
                try:
                    images = [np.asarray(Image.fromarray(obs["observation"][name]["rgb"])
                                         .resize((224, 224)))
                              for name in ("head_camera", "left_camera", "right_camera")]
                    result = client.predict_action({"type": "infer", "payload": {
                        "state": obs["joint_action"]["vector"], "image": np.stack(images)}})
                    assert result["ok"], result
                    task.take_action(result["data"]["action"][0], action_type="qpos")
                finally:
                    client.close()
            task.play_once()
            success = bool(task.plan_success and task.check_success())
            records.append({"seed": seed, "expert_success": success,
                            "echo_action_executed": options.echo_port is not None})
            for camera in ("head_camera", "left_camera", "right_camera"):
                task.save_camera_rgb(str(output / f"seed-{seed}-final-{camera}.png"), camera)
        except Exception as error:
            records.append({"seed": seed, "error": f"{type(error).__name__}: {error}"})
            raise
        finally:
            (output / "episodes.json").write_text(json.dumps(records, indent=2) + "\n")
            task.close_env()
        if success:
            print(json.dumps({"task": options.task, "variant": "clean", "episodes": records}))
            return
    raise RuntimeError("No successful expert episode in the specified seed range")


if __name__ == "__main__":
    main()
