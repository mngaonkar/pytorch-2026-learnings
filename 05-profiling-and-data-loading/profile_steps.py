import torch
import torch.nn as nn
from torch.profiler import profile, schedule, tensorboard_trace_handler

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
model = nn.Sequential(nn.Linear(512, 2048), nn.ReLU(), nn.Linear(2048, 10)).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)

def train_step():
    x = torch.randn(64, 512, device=device)
    y = torch.randint(0, 10, (64,), device=device)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    opt.zero_grad()

# skip 1 step, warm up 1, record 3; repeat once
with profile(schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
             on_trace_ready=tensorboard_trace_handler("./log"),  # tensorboard --logdir ./log
             record_shapes=True) as prof:
    for _ in range(6):
        train_step()
        prof.step()                   # tell the profiler a step ended

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
