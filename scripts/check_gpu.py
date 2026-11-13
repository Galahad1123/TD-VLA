"""Real GPU forward/backward and optional two-process NCCL check."""
import json
import os

import torch
import torch.distributed as dist
from torch.nn.functional import scaled_dot_product_attention


def main():
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    devices = [rank] if distributed else list(range(torch.cuda.device_count()))
    assert devices, "No CUDA devices"
    results = []
    for device in devices:
        torch.cuda.set_device(device)
        q = torch.randn(2, 4, 32, 64, device=f"cuda:{device}", dtype=torch.bfloat16,
                        requires_grad=True)
        out = scaled_dot_product_attention(q, q, q)
        out.float().square().mean().backward()
        assert torch.isfinite(out).all() and torch.isfinite(q.grad).all()
        results.append({"device": device, "name": torch.cuda.get_device_name(device),
                        "capability": torch.cuda.get_device_capability(device), "sdpa": "pass"})
    if distributed:
        dist.init_process_group("nccl")
        x = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
        dist.all_reduce(x)
        world = dist.get_world_size()
        assert x.item() == world * (world + 1) / 2
        model = torch.nn.parallel.DistributedDataParallel(
            torch.nn.Linear(16, 8).cuda(rank), device_ids=[rank])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.module.weight.detach().clone()
        model(torch.randn(4, 16, device=f"cuda:{rank}")).square().mean().backward()
        optimizer.step()
        assert not torch.equal(before, model.module.weight)
        weights = [torch.empty_like(model.module.weight) for _ in range(world)]
        dist.all_gather(weights, model.module.weight.detach())
        assert all(torch.equal(weights[0], w) for w in weights[1:])
        dist.destroy_process_group()
    print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                      "devices": results, "ddp": distributed}))


if __name__ == "__main__":
    main()
