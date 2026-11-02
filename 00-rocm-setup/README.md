# 0. ROCm setup

Two small checks to run right after `uv sync`, before anything else in this repo. They confirm that the environment holds a ROCm build of PyTorch, that it sees your AMD GPU, and that you can choose which GPUs a job uses.

Run both from this directory, inside the repo's uv environment (see the [root README](../README.md) for `uv sync`).

## `rocm_check.py`

Prints the torch version, the HIP runtime version, and whether a GPU is available. When it finds one, it prints the device name with its `gfx` target and memory, runs a 1024 x 1024 matmul on the GPU and waits for it, then reports whether the `nccl` distributed backend (RCCL on ROCm) is available.

**Run:**

```bash
uv run python rocm_check.py
```

**What you should see:** a `+rocm` torch version, a HIP version string, `GPU available True`, the device name with `gfx1201` (or your GPU's target), its memory, `matmul ok (1024, 1024)` and `True` for the nccl backend.

On ROCm, PyTorch reuses the `torch.cuda` API and the `"cuda"` device string for AMD GPUs, so `torch.cuda.is_available()` cannot tell a ROCm build from a CUDA build. `torch.version.hip` can: it is `None` on any build without ROCm, including the CPU-only wheel that `pip install torch` pulls from PyPI.

If you see `GPU available False` with a ROCm build, check that your user is in the `render` and `video` groups and that `uv run rocminfo` lists a `gfx` agent.

## `select_gpus.sh`

Lists the GPU architecture targets (`rocminfo`) and the GPUs with index, BDF and UUID (`amd-smi list`), runs `rocm_check.py` with `HIP_VISIBLE_DEVICES=0` so it sees only GPU 0, then starts `../01-distributed-training-and-parallelism/fsdp2_train.py` under `torchrun` on the first `NGPU` GPUs selected with `ROCR_VISIBLE_DEVICES`. `NGPU` defaults to 2.

**Run:**

```bash
# 2 or more GPUs
bash select_gpus.sh

# single-GPU machine
NGPU=1 bash select_gpus.sh
```

`HIP_VISIBLE_DEVICES` filters at the HIP level. `ROCR_VISIBLE_DEVICES` filters at the lower ROCm runtime level, applies to every ROCm program, and also accepts GPU UUIDs; AMD recommends it on Linux.

The script calls `rocminfo` and `amd-smi` through `uv run`, which uses the versions shipped in the environment's ROCm wheels. An older system `rocminfo` can fail on new GPUs with "The GPU node has an unrecognized id".

**Tested** on one Radeon RX 9070 XT (`gfx1201`, 16 GB) with `NGPU=1`, torch `2.14.0+rocm10.1.0`.
