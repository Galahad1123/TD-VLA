# ruff: noqa: E402 -- optional policy dependencies
"""Unequal valid counts must produce the same gradient as a combined global batch."""

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
import torch.multiprocessing as mp
from td_vla.training.losses import masked_regression


def _rank_loss(rank, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        weight = torch.tensor(0.3, requires_grad=True)
        pred = weight.expand(1, 1, 2, 2)
        valid = torch.full((1, 1, 2), rank == 1, dtype=torch.bool)
        target = torch.full_like(pred, 1.0 if rank == 1 else float("nan"))
        loss = masked_regression(pred, target, valid, smooth=True, distributed=True)
        loss.backward()
        averaged_gradient = weight.grad.clone()
        dist.all_reduce(averaged_gradient)
        averaged_gradient /= 2
        reference = torch.tensor(0.3, requires_grad=True)
        combined = masked_regression(
            reference.expand(2, 1, 2, 2),
            torch.tensor([float("nan"), 1.0]).reshape(2, 1, 1, 1).expand(2, 1, 2, 2),
            torch.tensor([False, True]).reshape(2, 1, 1).expand(2, 1, 2),
            smooth=True,
        )
        combined.backward()
        torch.testing.assert_close(averaged_gradient, reference.grad)
        assert torch.isfinite(loss)
    finally:
        dist.destroy_process_group()


def test_global_valid_normalization_with_empty_rank(tmp_path):
    mp.spawn(_rank_loss, args=(str(tmp_path / "gloo-init"),), nprocs=2, join=True)
