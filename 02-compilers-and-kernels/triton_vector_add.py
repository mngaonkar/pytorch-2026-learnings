# triton_vector_add.py
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                       # which block this program handles
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # element indices for this block
    mask = offs < n                              # guard the last partial block
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

device = torch.accelerator.current_accelerator()  # needs a GPU
x = torch.randn(1_000_000, device=device)
y = torch.randn_like(x)
out = torch.empty_like(x)
grid = (triton.cdiv(x.numel(), 1024),)          # one program per 1024 elements
add_kernel[grid](x, y, out, x.numel(), BLOCK=1024)
assert torch.allclose(out, x + y)
