# TorchTitan main on the ROCm nightly: context parallelism for packed sequences

The rest of `01-distributed-training-and-parallelism` uses TorchTitan v0.3.0 on stable torch `2.14.0+rocm10.1.0`. This folder is a separate uv project for TorchTitan `main`, which needs a newer PyTorch than any stable ROCm release. It installs AMD's ROCm nightly PyTorch and shows the newer context-parallel API:

```python
apply_transforms(config, [ContextParallelTransform(
    inner_attention_map={VarlenInnerAttention: AllGatherKVVarlenInnerAttention})])
```

**Source session:** "TorchTitan GraphTrainer: LLM Distributed Training with PyTorch's Compiler Toolkit" (Sanket Jayant Purandare, Sherlock Huang).

## Version requirement

- PyTorch: a ROCm nightly from **late October 2026 or later**. Tested with `torch[device-gfx1201]==2.16.0a0+rocm10.3.0a20261101` from `https://nightly.repo.amd.com/rocm/pytorch/whl-next/`.
- ROCm: the nightly pins its own ROCm and Triton, `rocm[libraries]==10.3.0a20261101` and `triton==3.9.0+gitb7ddcf86.rocm10.3.0a20261101`. They cannot share an environment with the stable `rocm==10.1.0` pin in the repo root, which is why this folder has its own `pyproject.toml`.
- TorchTitan: `main` at commit [`83abb2c`](https://github.com/pytorch/torchtitan/commit/83abb2c039ebeffdef27907ea7e33b6174061651) (2026-11-01).
- Python 3.12, Linux, the `amdgpu` kernel driver. For another GPU, change `device-gfx1201` to your target (for example `device-gfx942` for MI300X).

### Why older builds do not work

| Build | Result with TorchTitan `main` |
|---|---|
| Stable `2.14.0+rocm10.1.0` | Import fails: `cannot import name 'analyze_pipeline_activation_liveness' from 'torch.distributed.pipelining'`. Use TorchTitan v0.3.0 (the parent folder) with this build. |
| Nightly `2.15.0a0+rocm10.1.0a20260910` (newest 2.15/ROCm 10.1 nightly) | Same import error, and `PipelineStageInfo` is missing too. With both stubbed, training then fails in FSDP: `MixedPrecisionPolicy.__init__() got an unexpected keyword argument 'param_dtype_override_fn'`. These are real API gaps, so the build is too old. |
| TorchTitan v0.4.0 | Same import error on the 2.15 nightly, and v0.4.0 does not contain `AllGatherKVVarlenInnerAttention` (added to `main` on 2026-10-26). |
| Nightly `2.16.0a0+rocm10.3.0a20261101` | Works. Both pipelining APIs import natively, FSDP has `param_dtype_override_fn`, and both runs below complete. |

## Files

| File | What it does |
|---|---|
| `pyproject.toml` | Standalone uv project: nightly torch, ROCm 10.3 nightly wheels, TorchTitan from git at the tested commit. |
| `uv.lock` | Exact resolved versions from the tested setup. |
| `setup_torchtitan_nightly.sh` | Clones TorchTitan at the same commit into `./torchtitan` (for `run_train.sh`, the bundled tokenizer and the `c4_test` data), runs `uv sync`, then `check_nightly.py`. |
| `check_nightly.py` | Prints the torch/HIP version and GPU, and checks the APIs TorchTitan `main` needs. |
| `train_debugmodel_1gpu.sh` | Real training run of the Llama 3 debug model, `NGPU=1` by default. |
| `my_recipes.py` | `llama3_debugmodel_cp2`: varlen attention, `context_parallel_degree = 2`, `ContextParallelTransform` with `AllGatherKVVarlenInnerAttention`. |
| `dry_run_cp2_fake.sh` | Runs `llama3_debugmodel_cp2` on one GPU with fake process groups (`COMM_BACKEND=fake`, logical `NGPU=2`). |

## Setup

From this folder:

```bash
bash setup_torchtitan_nightly.sh
```

Expected output from `check_nightly.py`:

```
torch 2.16.0a0+rocm10.3.0a20261101 HIP 7.18.26434
GPU available True
device AMD Radeon RX 9070 XT gfx1201
pipelining liveness API ok
FSDP MixedPrecisionPolicy.param_dtype_override_fn ok
TorchTitan ContextParallelTransform + AllGatherKVVarlenInnerAttention ok
```

Notes on the `pyproject.toml`:

- Two indexes: torch, the `amd-torch-device-*` wheels and Triton are on `.../rocm/pytorch/whl-next/`; `rocm`, `rocm-sdk-*` and `rocm-bootstrap` are on `.../rocm/core/whl-next/`. `index-strategy = "unsafe-best-match"` lets uv pick each package from whichever index has the pinned version.
- `prerelease = "allow"` is needed because the ROCm SDK wheels are pre-releases pulled in transitively.
- `constraint-dependencies = ["tokenizers<1"]`: with pre-releases allowed, uv otherwise picks `tokenizers 1.0.0rc3`, which removed `AddedToken`, and TorchTitan fails at import.
- The `torchtitan/` checkout is still needed because `run_train.sh`, `tests/assets/tokenizer` and `tests/assets/c4_test` are not part of the installed package. The launch scripts `cd` into it and call `uv run --project ..`, so they use this folder's environment rather than building one from TorchTitan's own `pyproject.toml`.

## Run

Stop other GPU jobs on a 16 GB card first (`uv run amd-smi metric -m` shows used VRAM).

```bash
bash train_debugmodel_1gpu.sh      # real run, NGPU=1
bash dry_run_cp2_fake.sh           # CP=2 config on one GPU, fake communication
```

TorchTitan `main` changed its launcher compared with v0.3.0:

- `MODULE` is a full Python module path. The default is `torchtitan_recipes.tests.models.llama3` (v0.3.0 used `llama3`).
- The dry run uses `COMM_BACKEND=fake` (v0.3.0 used `COMM_MODE=fake_backend`). It runs `python3 -m torchtitan.train ... --comm-backend fake` in one process, without `torchrun`, and `NGPU` is the logical world size.
- Your own recipe module is found through `PYTHONPATH`; `dry_run_cp2_fake.sh` sets `PYTHONPATH=..` so `my_recipes.py` in this folder is importable from inside the checkout.

## Results (RX 9070 XT, gfx1201, 16 GB)

`train_debugmodel_1gpu.sh` (6.16M-parameter debug model, sequence length 2048, 16,384 tokens per step, varlen attention, 10 steps):

```
Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=1, cp=1, tp=1, ep=1
step:  1  loss:  7.98055  grad_norm:  1.3955  memory:  0.84GiB(5.30%)  tps: 3,954
step: 10  loss:  4.05408  grad_norm:  2.0161  memory:  0.90GiB(5.68%)  tps: 330,926  tflops: 23.69  mfu: 7.59%
```

`dry_run_cp2_fake.sh` (5 steps):

```
Running with fake process groups
Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=1, cp=2, tp=1, ep=1
step:  1  loss:  4.03406  grad_norm:  0.6301  memory:  0.63GiB(3.97%)
step:  5  loss:  3.26768  grad_norm:  1.1403  memory:  0.66GiB(4.12%)
```

The fake-backend run proves that the recipe builds, the attention swap is accepted, the CP=2 mesh is created and forward and backward run with half the tokens per rank. Its collectives do no real communication, so its loss, memory and throughput are not a measurement of context-parallel performance. A real CP=2 run needs two GPUs, for example on an AMD Instinct node: `PYTHONPATH=.. NGPU=2 MODULE=my_recipes CONFIG=llama3_debugmodel_cp2 uv run --project .. ./run_train.sh` from inside `torchtitan/`.
