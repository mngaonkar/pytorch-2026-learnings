# From Parallelism to Profiling on AMD GPUs

Code from Mahadev Gaonkar's Medium article **From Parallelism to Profiling on AMD GPUs: Five Things I Learned at PyTorch Conference 2026** (link coming when the article is published). Each folder matches a section of the article and has a README that names the conference session, the requirements, and the command to run. Script names match the file names printed in the article.

Everything here runs on AMD GPUs with ROCm. The repo is a [uv](https://docs.astral.sh/uv/) project that installs a ROCm build of PyTorch from AMD's package index.

## Setup (AMD GPU + ROCm with uv)

You need `uv` and the AMD GPU kernel driver (`amdgpu`) on the host. ROCm itself comes from Python wheels, so a system-wide ROCm install is not required.

```bash
git clone https://github.com/mngaonkar/pytorch-2026-learnings
cd pytorch-2026-learnings
uv sync
```

`uv sync` reads `.python-version` (3.12), creates `.venv`, and installs the pins from `pyproject.toml`:

- `[[tool.uv.index]]` adds AMD's index, `https://stable.repo.amd.com/rocm/whl-next/`, so `torch` and `rocm` resolve to ROCm builds instead of the CPU-only wheels on PyPI.
- `torch[device-gfx1201]==2.14.0+rocm10.1.0` is PyTorch 2.14.0 built against ROCm 10.1.0, with GPU code for `gfx1201` (Radeon RX 9070 XT). Resolving it also pulls in a ROCm build of Triton.
- `rocm[device-gfx1201,libraries]==10.1.0` installs the ROCm runtime, math libraries and tools into the environment.

**Other GPUs:** change `device-gfx1201` in both the `torch` and `rocm` lines to your GPU's target, for example `device-gfx942` (MI300X/MI325X), `device-gfx950` (MI350X/MI355X) or `device-gfx1100` (RX 7900 XTX), then run `uv sync` again. [TheRock release notes](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) list every target.

Check the install:

```bash
cd 00-rocm-setup
uv run python rocm_check.py
```

It should print a `+rocm` torch version, a HIP version, `GPU available True` and your GPU's `gfx` target. If `torch.version.hip` is `None`, the environment has a CPU-only torch.

### Running and adding packages

- `uv run <command>` runs a command inside the environment, for example `uv run python profile_steps.py` or `uv run torchrun --nproc_per_node=1 fsdp2_train.py`. No manual activation is needed.
- Add optional packages with `uv add`, which resolves them against the same ROCm pins: `uv add openai trl datasets tensorboard` covers every folder (`openai` for `measure_ttft.py`, `trl` and `datasets` for `grpo_exact_match.py`, `tensorboard` to view profiler traces).
- Do not `pip install torch` or install a package that pulls torch from PyPI. It replaces the ROCm build with the CPU-only one.
- vLLM is not installed into this environment. On AMD GPUs, run it with the official ROCm image: `03-inference-and-serving/vllm_serve_rocm_docker.sh`.

### ROCm tools

The environment bundles its own ROCm tools, matched to the runtime PyTorch loads. Call them through `uv run`:

```bash
uv run rocminfo
uv run amd-smi list
uv run rocprofv3 --sys-trace -- python profile_steps.py
```

A system ROCm in `/opt/rocm` from a different release may not recognize newer GPUs, and a mismatched `rocprofv3` can abort while tracing.

## Structure

```
00-rocm-setup/                             ROCm install check and GPU selection
01-distributed-training-and-parallelism/   FSDP2 script, TorchTitan context-parallel recipe and launch commands
02-compilers-and-kernels/                  eager vs compile, graph breaks, Triton add, fusion diagram
03-inference-and-serving/                  vLLM serve (pip and ROCm Docker), prefix-cache latency client, KV-cache diagram
04-training-agents/                        GRPO exact-match reward, agent-loop diagram
05-profiling-and-data-loading/             profiler schedule, TraceLens report, rocprofv3 trace, num_workers sweep, pinned copies
```

ASCII diagrams live in the folder README. Shell files are the article's commands, unchanged. Start with `00-rocm-setup`, then the README in the folder you want to run.

The new AMD scripts were tested on one Radeon RX 9070 XT (`gfx1201`, 16 GB) with torch `2.14.0+rocm10.1.0`. Multi-GPU commands, such as context parallelism across 2 GPUs, need a machine with that many GPUs, for example an AMD Instinct node.

| Article block | Where it lives |
|---|---|
| ROCm install check | [`00-rocm-setup/rocm_check.py`](00-rocm-setup/rocm_check.py) |
| GPU selection on ROCm | [`00-rocm-setup/select_gpus.sh`](00-rocm-setup/select_gpus.sh) |
| FSDP2 training script | [`01-distributed-training-and-parallelism/fsdp2_train.py`](01-distributed-training-and-parallelism/fsdp2_train.py) |
| TorchTitan context-parallel recipe | [`01-distributed-training-and-parallelism/my_recipes.py`](01-distributed-training-and-parallelism/my_recipes.py) |
| `NGPU=2` launch of `llama3_8b_cp2` | [`01-distributed-training-and-parallelism/launch_llama3_8b_cp2.sh`](01-distributed-training-and-parallelism/launch_llama3_8b_cp2.sh) |
| `run_train.sh` excerpt | [`01-distributed-training-and-parallelism/run_train.sh`](01-distributed-training-and-parallelism/run_train.sh) |
| `COMM_BACKEND=fake` dry run | [`01-distributed-training-and-parallelism/dry_run_fake_backend.sh`](01-distributed-training-and-parallelism/dry_run_fake_backend.sh) |
| Eager vs fused memory diagram | [`02-compilers-and-kernels/README.md`](02-compilers-and-kernels/README.md) |
| Eager vs `torch.compile` benchmark | [`02-compilers-and-kernels/bench_eager_vs_compiled.py`](02-compilers-and-kernels/bench_eager_vs_compiled.py) |
| Graph-break report | [`02-compilers-and-kernels/find_graph_breaks.py`](02-compilers-and-kernels/find_graph_breaks.py) |
| `TORCH_LOGS` commands | [`02-compilers-and-kernels/torch_logs.sh`](02-compilers-and-kernels/torch_logs.sh) |
| Triton vector add | [`02-compilers-and-kernels/triton_vector_add.py`](02-compilers-and-kernels/triton_vector_add.py) |
| Prefill / decode / KV-cache diagram | [`03-inference-and-serving/README.md`](03-inference-and-serving/README.md) |
| `vllm serve` | [`03-inference-and-serving/vllm_serve.sh`](03-inference-and-serving/vllm_serve.sh) |
| vLLM on ROCm (Docker) | [`03-inference-and-serving/vllm_serve_rocm_docker.sh`](03-inference-and-serving/vllm_serve_rocm_docker.sh) |
| TTFT and per-token client | [`03-inference-and-serving/measure_ttft.py`](03-inference-and-serving/measure_ttft.py) |
| Agent harness / reward diagram | [`04-training-agents/README.md`](04-training-agents/README.md) |
| GRPO exact-match trainer | [`04-training-agents/grpo_exact_match.py`](04-training-agents/grpo_exact_match.py) |
| Compute-bound vs input-bound diagram | [`05-profiling-and-data-loading/README.md`](05-profiling-and-data-loading/README.md) |
| Profiler schedule | [`05-profiling-and-data-loading/profile_steps.py`](05-profiling-and-data-loading/profile_steps.py) |
| TraceLens report | [`05-profiling-and-data-loading/tracelens_report.sh`](05-profiling-and-data-loading/tracelens_report.sh) |
| rocprofv3 system trace | [`05-profiling-and-data-loading/rocprofv3_trace.sh`](05-profiling-and-data-loading/rocprofv3_trace.sh) |
| `num_workers` sweep | [`05-profiling-and-data-loading/bench_num_workers.py`](05-profiling-and-data-loading/bench_num_workers.py) |
| Pinned host-to-device copies | [`05-profiling-and-data-loading/overlap_host_to_device.py`](05-profiling-and-data-loading/overlap_host_to_device.py) |
