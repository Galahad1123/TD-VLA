import os
import torch
import torch.distributed as d
from datetime import timedelta

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
d.init_process_group("nccl", timeout=timedelta(seconds=90))
for size in (1024, 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024):
    print(rank, "broadcast", size, flush=True)
    x = torch.full((size,), rank + 1.0, device="cuda")
    d.broadcast(x, 0)
    torch.cuda.synchronize()
    assert torch.all(x == 1)
    print(rank, "reduce", size, flush=True)
    d.all_reduce(x)
    torch.cuda.synchronize()
    assert torch.all(x == 2)
    del x
print(rank, "PASS", flush=True)
d.destroy_process_group()
