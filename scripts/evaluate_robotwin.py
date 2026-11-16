"""Clean subset evaluation with official expert filtering, success and step limits."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/TurboVLA/third_party/starvla_runtime"))


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def server(args):
    import torch
    from turbovla.models.turbovla import TurboVLA
    from turbovla.models.configuration import TurboVLAConfig
    from td_vla.models.multistream import TDVLA, decode_stage
    from td_vla.data.multistream import normalize_commands, denormalize_actions, sha256
    from td_vla.integrations.robotwin_policy import prepare_images
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    metadata = json.loads(args.metadata.read_text())
    config = metadata["config"]
    split = json.loads(args.split.read_text())
    assert sha256(args.split) == metadata["split_sha256"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    assert checkpoint["step"] == 45000 and checkpoint["signature"] == signature
    torch.manual_seed(42)
    model = (
        TurboVLA(TurboVLAConfig.from_mapping(config["model"]))
        if config["mode"] == "baseline"
        else TDVLA(
            TurboVLAConfig.from_mapping(config["model"]),
            track_points=config["track_points"],
            attention=config["mode"],
        )
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.cuda().eval()
    del checkpoint

    class Policy:
        @torch.inference_mode()
        def predict_action(self, **payload):
            task = payload["task"]
            images = prepare_images(payload["image"], config["model"]["vision"]["image_size"])
            state = normalize_commands(payload["state"], split)
            # Match training: model owns encoder autocast, decoder remains FP32.
            result = model(
                [payload["lang"]],
                torch.from_numpy(images[None]).cuda(),
                torch.from_numpy(state[None]).cuda(),
            )
            if isinstance(result, torch.Tensor):
                assert result.dtype == torch.float32
                return dict(actions=denormalize_actions(result[0].cpu().numpy(), split))
            assert result["actions"].dtype == torch.float32
            stage = decode_stage(result["stage_logits"], [len(split["vocab"][task])])
            return dict(
                actions=denormalize_actions(result["actions"][0].float().cpu().numpy(), split),
                track=result["track"][0].float().cpu().numpy(),
                stage=int(stage["stage"][0]),
                raw_stage=int(stage["raw_stage"][0]),
            )

    print(json.dumps(dict(ready=True, step=45000, weights="online", port=args.port)), flush=True)
    WebsocketPolicyServer(
        Policy(), host="127.0.0.1", port=args.port, idle_timeout=86400, metadata={"step": 45000}
    ).serve_forever()


def evaluate(args):
    import yaml
    from td_vla.integrations.robotwin_policy import ChunkPolicy
    from td_vla.data.pilot import INSTRUCTIONS
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    robotwin = ROOT / "external/RoboTwin"
    sys.path.insert(0, str(robotwin))
    os.chdir(robotwin)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((robotwin / "task_config/demo_clean.yml").read_text())
    robot = robotwin / "assets/embodiments/aloha-agilex"
    embodiment = yaml.safe_load((robot / "config.yml").read_text())
    cfg.update(
        task_name=args.task,
        task_config="demo_clean",
        need_plan=True,
        left_robot_file=str(robot),
        right_robot_file=str(robot),
        left_embodiment_config=embodiment,
        right_embodiment_config=embodiment,
        dual_arm_embodied=True,
        embodiment_name="aloha-agilex",
        render_freq=0,
        save_path=str(output),
        save_data=False,
        collect_data=False,
        eval_mode=True,
        eval_video_log=False,
        policy_name="td_vla",
        ckpt_setting="step-45000-online",
    )
    task_class = getattr(importlib.import_module(f"envs.{args.task}"), args.task)
    client = WebsocketClientPolicy("127.0.0.1", args.port)
    policy = ChunkPolicy(client, args.task, args.exec_horizon)
    records, rejected = [], []
    start = time.time()
    write_json(
        output / "config.json",
        dict(
            task=args.task,
            episodes=args.episodes,
            start_seed=args.seed,
            exec_horizon=args.exec_horizon,
            prediction_horizon=50,
            instruction=INSTRUCTIONS[args.task],
            instruction_mode="fixed training instruction",
            weights="online step 45000",
            variant="demo_clean",
            expert_check=True,
        ),
    )

    def status(state):
        write_json(
            output / "status.json",
            dict(
                state=state,
                pid=os.getpid(),
                episodes=len(records),
                successes=sum(r["success"] for r in records),
                rejected_expert_seeds=len(rejected),
                elapsed_seconds=time.time() - start,
            ),
        )

    status("running")
    try:
        seed = args.seed
        while len(records) < args.episodes:
            if seed >= args.seed + 1000:
                raise RuntimeError("Exceeded expert seed search limit")
            task = task_class()
            accepted = False
            try:
                task.setup_demo(now_ep_num=len(records), seed=seed, is_test=True, **cfg)
                task.play_once()
                accepted = bool(task.plan_success and task.check_success())
                if not accepted:
                    rejected.append(dict(seed=seed, reason="expert_failed"))
            except Exception as error:
                rejected.append(dict(seed=seed, reason=f"{type(error).__name__}: {error}"))
            finally:
                task.close_env()
            write_json(output / "expert-rejections.json", rejected)
            if not accepted:
                seed += 1
                continue
            episode_start = time.time()
            task = task_class()
            record = dict(seed=seed, episode=len(records), success=False)
            try:
                task.setup_demo(now_ep_num=len(records), seed=seed, is_test=True, **cfg)
                task.set_instruction(instruction=INSTRUCTIONS[args.task])
                policy.reset()
                while task.take_action_cnt < task.step_lim and not task.eval_success:
                    task.take_action(policy.action(task.get_obs()), action_type="qpos")
                record.update(
                    success=bool(task.eval_success),
                    steps=task.take_action_cnt,
                    step_limit=task.step_lim,
                )
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                record["seconds"] = time.time() - episode_start
                records.append(record)
                write_json(output / "episodes.json", records)
                write_json(output / f"predictions-{seed}.json", policy.predictions)
                task.close_env(clear_cache=True)
                status("running")
            print(json.dumps(record), flush=True)
            seed += 1
        status("complete")
    except BaseException:
        (output / "error.txt").write_text(traceback.format_exc())
        status("failed")
        raise
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["server", "evaluate"])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--exec-horizon", type=int, default=50)
    args = parser.parse_args()
    (server if args.mode == "server" else evaluate)(args)


if __name__ == "__main__":
    main()
