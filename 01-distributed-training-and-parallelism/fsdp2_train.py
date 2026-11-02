# fsdp2_train.py  -- run with: torchrun --nproc_per_node=2 fsdp2_train.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
import logging

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

dist.init_process_group()                      # torchrun sets rank/world size
rank = int(os.environ["LOCAL_RANK"])
if torch.accelerator.is_available():
    torch.accelerator.set_device_index(rank)    # one GPU per process
    device = torch.accelerator.current_accelerator()
    logging.info(f"Using GPU {device}")
else:
    device = torch.device("cpu")
    logging.info("Using CPU")

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        return x + self.ff(x)

model = nn.Sequential(*[Block(256) for _ in range(4)]).to(device)
for block in model:
    fully_shard(block)        # each block's weights are split across GPUs
fully_shard(model)            # root module: shards whatever is left

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)   # create AFTER sharding
for step in range(5):
    x = torch.randn(8, 256, device=device)
    loss = model(x).pow(2).mean()
    loss.backward()           # gradients are reduce-scattered to their shards
    opt.step()
    opt.zero_grad()
    if rank == 0:
        print(f"step {step} loss {loss.item():.4f}")

dist.destroy_process_group()
