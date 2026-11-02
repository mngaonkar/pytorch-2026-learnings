# rocm_check.py
import torch
import torch.distributed as dist

print("torch", torch.__version__)
print("HIP runtime", torch.version.hip)           # None means this is not a ROCm build
print("GPU available", torch.cuda.is_available())

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("device", props.name, props.gcnArchName)  # e.g. gfx1201 on an RX 9070 XT
    print("memory", round(props.total_memory / 2**30, 1), "GiB")
    a = torch.randn(1024, 1024, device="cuda")     # "cuda" is the HIP device on ROCm
    b = a @ a
    torch.cuda.synchronize()                        # wait for the GPU to finish
    print("matmul ok", tuple(b.shape))
    print("nccl backend (RCCL) available", dist.is_nccl_available())
