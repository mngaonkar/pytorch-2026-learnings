# From Parallelism to Profiling

This repository holds the code from Mahadev Gaonkar's article [From Parallelism to Profiling: Five Things I Learned at PyTorch Conference 2026](TODO). It turns the article's five sections into folders of runnable snippets: distributed training and parallelism, compilers and kernels, inference and serving, training agents, and profiling and data loading. The snippets keep the logic and values from the article, and each folder README names the conference session, the requirements, and the command to run. Replace the `TODO` link target with the Medium URL when the article is published.

## Setup

```bash
pip install -r requirements.txt
```

`trl` and `vllm` are optional. Comments in `requirements.txt` say when to install them. Snippets that can run on CPU use `torch.accelerator` and take the CPU path when no accelerator is present. The Triton kernel and the vLLM server need a GPU. The code is hardware-vendor neutral.

## Structure

```
01-distributed-training-and-parallelism/   FSDP2 script, TorchTitan context-parallel recipe and launch commands
02-compilers-and-kernels/                  eager vs compile, graph breaks, Triton add, fusion diagram
03-inference-and-serving/                  vLLM serve, prefix-cache latency client, KV-cache diagram
04-training-agents/                        GRPO exact-match reward, agent-loop diagram
05-profiling-and-data-loading/             profiler schedule, num_workers sweep, pinned copies, input-pipeline diagram
```

ASCII diagrams live in the folder README. Shell files are the article's commands, unchanged. Start with the README in the folder you want to run.

| Article block | Where it lives |
|---|---|
| FSDP2 training script (`train.py` in the article) | [`01-distributed-training-and-parallelism/fsdp2_train.py`](01-distributed-training-and-parallelism/fsdp2_train.py) |
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
| TTFT and per-token client | [`03-inference-and-serving/measure_ttft.py`](03-inference-and-serving/measure_ttft.py) |
| Agent harness / reward diagram | [`04-training-agents/README.md`](04-training-agents/README.md) |
| GRPO exact-match trainer | [`04-training-agents/grpo_exact_match.py`](04-training-agents/grpo_exact_match.py) |
| Compute-bound vs input-bound diagram | [`05-profiling-and-data-loading/README.md`](05-profiling-and-data-loading/README.md) |
| Profiler schedule | [`05-profiling-and-data-loading/profile_steps.py`](05-profiling-and-data-loading/profile_steps.py) |
| `num_workers` sweep | [`05-profiling-and-data-loading/bench_num_workers.py`](05-profiling-and-data-loading/bench_num_workers.py) |
| Pinned host-to-device copies | [`05-profiling-and-data-loading/overlap_host_to_device.py`](05-profiling-and-data-loading/overlap_host_to_device.py) |
