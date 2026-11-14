"""Complete trusted-local checkpoints, including per-rank RNG and EMA."""

from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def save_checkpoint(path, model, optimizer, scheduler, ema, step, signature):
    world = dist.get_world_size() if dist.is_initialized() else 1
    states = [None] * world
    if dist.is_initialized():
        dist.all_gather_object(states, rng_state())
    else:
        states[0] = rng_state()
    if not dist.is_initialized() or dist.get_rank() == 0:
        path = Path(path)
        partial = path.with_suffix(".partial")
        torch.save(
            dict(
                schema="td-vla-checkpoint-v1",
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                ema=ema,
                step=step,
                signature=signature,
                rng=states,
            ),
            partial,
        )
        partial.replace(path)
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(path, model, optimizer, scheduler, signature):
    # Only load trusted self-generated artifacts, never untrusted pickle checkpoints.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if (
        saved["schema"] != "td-vla-checkpoint-v1"
        or saved["signature"] != signature
        or len(saved["rng"]) != world
    ):
        raise ValueError("Checkpoint signature/world-size mismatch")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    restore_rng(saved["rng"][rank])
    device = next(model.parameters()).device
    ema = {k: v.to(device) for k, v in saved["ema"].items()} if rank == 0 else {}
    return saved["step"], ema
