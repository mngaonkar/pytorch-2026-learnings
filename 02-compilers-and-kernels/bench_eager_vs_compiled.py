# bench_eager_vs_compiled.py
import time
import torch
import torch.nn as nn

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
model = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024)).to(device)
x = torch.randn(64, 1024, device=device)

def bench(fn, iters=50):
    for _ in range(5):                 # warmup: the first calls compile
        fn(x)
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()  # wait for queued GPU work
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

compiled = torch.compile(model)        # or mode="max-autotune" for longer tuning
with torch.no_grad():
    print(f"eager    {bench(model):.2f} ms")
    print(f"compiled {bench(compiled):.2f} ms")
    # check the outputs match before trusting the speedup
    print("max diff", (model(x) - compiled(x)).abs().max().item())
