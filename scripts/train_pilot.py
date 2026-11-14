"""Dual-GPU upstream action-only diagnostic training. NOT a formal baseline recipe."""

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from turbovla.models.configuration import TurboVLAConfig
from turbovla.models.turbovla import TurboVLA
from td_vla.data.pilot import PilotData, make_split


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(),
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=["move_stapler_pad", "lift_pot", "handover_block"])
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--save-every", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--resume", type=Path)
    p.add_argument("--fixed-batch", action="store_true")
    p.add_argument("--initialization", choices=["random-fixture-diagnostic"], required=True)
    a = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    torch.use_deterministic_algorithms(True)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError("Use a new output directory even when resuming")
    a.output.mkdir(parents=True, exist_ok=True)
    split = make_split(a.data, a.tasks)
    cfg = TurboVLAConfig.from_mapping(
        {
            "compatibility_profile": "robotwin",
            "text": {
                "model_name_or_path": str(a.fixtures / "random-bert"),
                "attention_implementation": "eager",
            },
            "vision": {
                "model_name_or_path": str(a.fixtures / "random-dino"),
                "image_size": 224,
                "num_views": 3,
                "attention_implementation": "sdpa",
                "position_embedding": "learned_patch",
            },
            "interaction": {
                "padding_strategy": "zero_fill",
                "residual_style": "pre_norm",
                "attention_backend": "sdpa",
                "compute_precision": "bf16_autocast",
            },
            "action": {"action_dim": 14, "state_dim": 14, "horizon": 50},
        }
    )
    metadata = dict(
        schema="pilot-checkpoint-v1",
        initialization=a.initialization,
        model=cfg.to_dict(),
        split=split,
        world_size=world,
        batch_size=a.batch_size,
        fixed_batch=a.fixed_batch,
        optimizer={"name": "AdamW", "lr": 5e-5, "weight_decay": 0.01, "betas": [0.9, 0.999]},
        scheduler={"warmup_steps": 2, "type": "constant_after_warmup"},
        ema_decay=0.99,
        image_transform="bilinear 224, ImageNet mean/std",
        seed=42,
    )
    signature = fingerprint(metadata)
    model = TurboVLA(cfg).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.01, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 2))
    ema = (
        {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
        if rank == 0
        else None
    )
    start = 0
    if a.resume:
        checkpoint = torch.load(
            a.resume, map_location="cpu", weights_only=False
        )  # Trusted self-generated checkpoint only.
        if checkpoint["signature"] != signature:
            raise ValueError("Resume configuration or data fingerprint mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if rank == 0:
            ema = {k: v.cuda() for k, v in checkpoint["ema"].items()}
        start = checkpoint["step"]
        saved_rng = checkpoint["rng"][rank]
        del checkpoint
    if world > 1:
        wrapped = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )
    else:
        wrapped = model
    train = PilotData(a.data, split)
    validation = PilotData(a.data, split, "validation") if rank == 0 else None
    # Initialize all objects before restoring RNG; first resumed batch uses the same global step.
    if a.resume:
        restore_rng(saved_rng)
    if rank == 0:
        (a.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        provenance = {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "resume_from": str(a.resume) if a.resume else None,
            "start_step": start,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in params),
        }
        (a.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    loss_log = []
    for step in range(start, a.steps):
        tick = time.monotonic()
        wrapped.train()
        texts, image, state, target, mask = train.batch(
            step, rank, a.batch_size, fixed=a.fixed_batch
        )
        image, state, target, mask = [x.cuda() for x in (image, state, target, mask)]
        optimizer.zero_grad(set_to_none=True)
        output = wrapped(texts, image, state)
        error = (output.float() - target).abs() * mask[..., None]
        # Global element-normalized masked L1, even with different tail lengths per rank.
        denom = mask.sum().float() * target.shape[-1]
        if world > 1:
            dist.all_reduce(denom)
        loss = error.sum() * world / denom
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        assert all(p.grad is None for p in model.text_encoder.bert.parameters())
        gradients = {
            name: sum(
                float(p.grad.float().abs().sum()) for p in module.parameters() if p.grad is not None
            )
            for name, module in [("vision", model.vision_encoder), ("action", model.action_head)]
        }
        if not all(np.isfinite(x) and x > 0 for x in gradients.values()):
            raise RuntimeError(f"Invalid gradient paths: {gradients}")
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if rank == 0:
            with torch.no_grad():
                for k, p0 in model.named_parameters():
                    if k in ema:
                        ema[k].lerp_(p0.detach(), 0.01)
        metric = loss.detach()
        if world > 1:
            dist.all_reduce(metric)
            metric /= world
        torch.cuda.synchronize()
        if rank == 0:
            record = dict(
                step=step + 1,
                loss=metric.item(),
                gradient_norm=norm.item(),
                gradient_paths=gradients,
                seconds=time.monotonic() - tick,
                peak_memory_bytes=torch.cuda.max_memory_allocated(),
                lr=optimizer.param_groups[0]["lr"],
            )
            loss_log.append(record)
            with (a.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, rng_state())
            else:
                states[0] = rng_state()
            if rank == 0:
                saved = dict(
                    signature=signature,
                    metadata=metadata,
                    step=step + 1,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    ema=ema,
                    rng=states,
                )
                final = a.output / f"step-{step + 1:04d}.pt"
                partial = final.with_suffix(".partial")
                torch.save(saved, partial)
                partial.replace(final)
            if world > 1:
                dist.barrier()
    if rank == 0:
        model.eval()
        with torch.no_grad():
            texts, image, state, target, mask = validation.batch(0, 0, 2, fixed=True)
            pred = model(texts, image.cuda(), state.cuda()).float().cpu()
            val = ((pred - target).abs() * mask[..., None]).sum() / (mask.sum() * 14)
        summary = dict(
            steps=a.steps,
            start_step=start,
            validation_l1=val.item(),
            first_loss=loss_log[0]["loss"],
            last_loss=loss_log[-1]["loss"],
            purpose="random initialization pipeline/overfit diagnostic; not pretrained benchmark performance",
        )
        (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
