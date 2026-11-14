"""Unified baseline/full/block training with explicit diagnostic or verified base initialization."""

import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import default_collate
from turbovla.models.configuration import TurboVLAConfig
from turbovla.models.turbovla import TurboVLA

from td_vla.data.multistream import MultiStreamDataset, sha256
from td_vla.models.multistream import TDVLA, decode_stage
from td_vla.training.checkpoint import load_checkpoint, save_checkpoint
from td_vla.training.initialization import verify_pretrained, load_grounding_initialization
from td_vla.training.schedule import learning_rate_factor
from td_vla.evaluation.offline import evaluate_fixed_subset
from td_vla.training.losses import LossWeights, multitask_loss


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def batch(dataset, step, rank, size, device, seed, fixed=False):
    indices = dataset.sample_indices(step, rank, size, seed=seed, fixed=fixed)
    sample = default_collate([dataset[i] for i in indices])
    return (
        sample["instruction"],
        sample["images"].to(device),
        sample["state"].to(device),
        {k: v.to(device) for k, v in sample["targets"].items()},
        sample["stage_count"].to(device),
    )


def gradient_paths(model):
    modules = dict(
        vision=model.vision_encoder,
        fusion=model.vision_language_interaction,
        text_projection=model.text_encoder.text_projection,
        state=model.action_head.state_projection,
        action=model.action_head.decoder.action_projection,
    )
    if isinstance(model, TDVLA):
        modules.update(
            track=model.action_head.decoder.track_projection,
            stage=model.action_head.decoder.stage_projection,
        )
    return {
        k: sum(
            float(p.grad.float().square().sum()) for p in module.parameters() if p.grad is not None
        )
        ** 0.5
        for k, module in modules.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--pretrained-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "full", "block"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fixed-batch", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.steps = args.steps if args.steps is not None else config.get("total_steps", 16)
    args.save_every = (
        args.save_every if args.save_every is not None else config.get("save_every", 8)
    )
    if args.steps > config.get("total_steps", args.steps):
        raise ValueError("Stopping step exceeds configured schedule total")
    repo = Path(__file__).resolve().parents[1]
    initialization_provenance = None
    if config["initialization"] == "base-pretrained-v1":
        if args.pretrained_root is None or args.fixtures is not None:
            raise ValueError("Formal training requires --pretrained-root and forbids --fixtures")
        specs = json.loads((repo / "configs/pretrained.json").read_text())
        initialization_provenance = verify_pretrained(args.pretrained_root, specs)
        config["model"]["text"]["model_name_or_path"] = str(args.pretrained_root / "bert")
        config["model"]["vision"]["model_name_or_path"] = str(args.pretrained_root / "vision")
    elif config["initialization"] == "random-fixture-diagnostic":
        if args.fixtures is None or args.pretrained_root is not None:
            raise ValueError("Diagnostic mode requires only --fixtures")
        config["model"]["text"]["model_name_or_path"] = str(args.fixtures / "random-bert")
        config["model"]["vision"]["model_name_or_path"] = str(args.fixtures / "random-dino")
    else:
        raise ValueError("Unknown initialization mode")
    if args.steps < 1 or args.save_every < 1 or config["batch_size"] < 1:
        raise ValueError("Positive steps/save interval/batch size required")
    config["mode"] = args.mode or config["mode"]
    if config["mode"] not in {"baseline", "full", "block"}:
        raise ValueError("Invalid mode")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rank, world = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=device)
    torch.use_deterministic_algorithms(True)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Use an empty output directory, including for resumed runs")
        args.output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    split = json.loads(args.split.read_text())
    train = MultiStreamDataset(args.data, split, image_size=config["model"]["vision"]["image_size"])
    validation = MultiStreamDataset(
        args.data, split, "validation", image_size=config["model"]["vision"]["image_size"]
    )
    cfg = TurboVLAConfig.from_mapping(config["model"])
    if not cfg.text.frozen or cfg.vision.frozen or cfg.action.horizon != split["horizon"]:
        raise ValueError("Model freeze/horizon configuration does not match v1")
    weights = LossWeights(**config["loss_weights"])
    # Include source/config/split identity in restore contract, not the requested stopping step.
    repo = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), *sorted((repo / "src/td_vla").rglob("*.py"))]
    provenance = {str(p.relative_to(repo)): sha256(p) for p in sources}
    metadata = dict(
        config=config,
        split_sha256=sha256(args.split),
        world_size=world,
        fixed_batch=args.fixed_batch,
        sources=provenance,
        torch=torch.__version__,
        weights=asdict(weights),
        initialization_provenance=initialization_provenance,
    )
    signature = fingerprint(metadata)
    if rank == 0:
        (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    model = (
        TurboVLA(cfg)
        if config["mode"] == "baseline"
        else TDVLA(cfg, track_points=config["track_points"], attention=config["mode"])
    )
    if initialization_provenance is not None:
        report = load_grounding_initialization(
            model, args.pretrained_root / "fusion/groundingdino_swint_ogc.pth"
        )
        if rank == 0:
            (args.output / "initialization.json").write_text(json.dumps(report, indent=2) + "\n")
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=tuple(config.get("betas", [0.9, 0.999])),
        foreach=False,
    )
    if config.get("schedule", "warmup_constant") == "warmup_cosine":

        def factor(step):
            return learning_rate_factor(
                step, config["warmup_steps"], config["total_steps"], config["min_lr_factor"]
            )
    elif config.get("schedule", "warmup_constant") == "warmup_constant":

        def factor(step):
            return min(1.0, (step + 1) / max(1, config["warmup_steps"]))
    else:
        raise ValueError("Unknown learning-rate schedule")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    wrapped = (
        torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )
        if world > 1
        else model
    )
    ema = (
        {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        if rank == 0
        else {}
    )
    start = 0
    if args.resume:
        start, ema = load_checkpoint(args.resume, model, optimizer, scheduler, signature)
    if start >= args.steps:
        raise ValueError("Requested stopping step must exceed resumed step")
    records = []
    for step in range(start, args.steps):
        tick = time.monotonic()
        wrapped.train()
        texts, images, state, targets, counts = batch(
            train, step, rank, config["batch_size"], device, config["seed"], args.fixed_batch
        )
        optimizer.zero_grad(set_to_none=True)
        output = wrapped(texts, images, state)
        if isinstance(output, torch.Tensor):
            output = {"actions": output}
        losses = multitask_loss(output, targets, weights, distributed=world > 1)
        if not torch.isfinite(losses["total"]):
            raise RuntimeError("Nonfinite loss")
        losses["total"].backward()
        if any(p.grad is not None or p.requires_grad for p in model.text_encoder.bert.parameters()):
            raise RuntimeError("BERT must stay frozen")
        should_log = (
            step == start or (step + 1) % config.get("log_every", 1) == 0 or step + 1 == args.steps
        )
        paths = gradient_paths(model) if should_log else {}
        if not all(np.isfinite(v) for v in paths.values()):
            raise RuntimeError(f"Nonfinite gradient paths: {paths}")
        if step == 0 and not all(paths[k] > 0 for k in ("vision", "fusion", "state", "action")):
            raise RuntimeError(f"Missing action gradient path: {paths}")
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        used_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        if rank == 0:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in ema:
                        ema[name].lerp_(p.detach(), 1 - config["ema_decay"])
        values = torch.stack([v.detach() for v in losses.values()])
        if world > 1:
            dist.all_reduce(values)
            values /= world
        torch.cuda.synchronize()
        if rank == 0 and should_log:
            record = dict(
                step=step + 1,
                losses=dict(zip(losses, values.cpu().tolist())),
                gradient_paths=paths,
                gradient_norm=float(norm),
                learning_rate=used_lr,
                seconds=time.monotonic() - tick,
                peak_memory_bytes=torch.cuda.max_memory_allocated(),
            )
            records.append(record)
            with (args.output / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            status = dict(
                state="running",
                step=step + 1,
                target_steps=args.steps,
                updated_unix=time.time(),
                last_metrics=record,
            )
            partial_status = args.output / "status.partial"
            partial_status.write_text(json.dumps(status, indent=2) + "\n")
            partial_status.replace(args.output / "status.json")
        if config.get("validate_every", 0) and (step + 1) % config["validate_every"] == 0:
            if world > 1:
                dist.barrier()
            if rank == 0:
                result = evaluate_fixed_subset(
                    model,
                    validation,
                    device,
                    min(config["batch_size"], 32),
                    config["validation_frames_per_episode"],
                )
                with (args.output / "validation.jsonl").open("a") as file:
                    file.write(json.dumps(dict(step=step + 1, **result)) + "\n")
            if world > 1:
                dist.barrier()
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            save_checkpoint(
                args.output / f"step-{step + 1:04d}.pt",
                model,
                optimizer,
                scheduler,
                ema,
                step + 1,
                signature,
            )
    if rank == 0:
        model.eval()
        with torch.no_grad():
            texts, images, state, targets, counts = batch(
                validation, 0, 0, config["batch_size"], device, config["seed"], True
            )
            output = model(texts, images, state)
            if isinstance(output, torch.Tensor):
                output = {"actions": output}
            val = {k: float(v) for k, v in multitask_loss(output, targets, weights).items()}
            if "stage_logits" in output:
                decoded = decode_stage(output["stage_logits"], counts)
                valid = targets["stage_valid"] & (targets["stage"] >= 0)
                val["stage_accuracy"] = (
                    float((decoded["stage"][valid] == targets["stage"][valid]).float().mean())
                    if valid.any()
                    else None
                )
                val["stage_out_of_range"] = (
                    float(decoded["out_of_range"][valid].float().mean()) if valid.any() else None
                )
        summary = dict(
            start_step=start,
            end_step=args.steps,
            parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in params),
            validation_single_batch=val,
            first_losses=records[0]["losses"],
            last_losses=records[-1]["losses"],
            purpose=config["purpose"],
        )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.output / "status.json").write_text(
            json.dumps(
                dict(
                    state="completed",
                    step=args.steps,
                    target_steps=args.steps,
                    updated_unix=time.time(),
                ),
                indent=2,
            )
            + "\n"
        )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
