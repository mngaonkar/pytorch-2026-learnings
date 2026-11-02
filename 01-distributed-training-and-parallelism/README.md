# 1. Distributed training and parallelism

Two ways to split a training step, copied from the article. `fsdp2_train.py` shards a small model with FSDP2 and runs on CPU or GPU. The TorchTitan files configure context parallelism for TorchTitan v0.3.0 and run from a TorchTitan checkout placed next to this repo.

## `fsdp2_train.py`

Builds a four-block residual MLP, calls `fully_shard` on each block and then on the root, creates AdamW after sharding, and trains for five steps on random inputs.

**Source session:** "Transformers.distributed: Bringing First-Class Distributed Training to 2.4M+ Hugging Face Models" (Ferdinand Mom).

**Requirements:** PyTorch (`torch.distributed.fsdp.fully_shard` and `torchrun`). No extra packages and no download. CPU is enough to exercise the sharding logic. When GPUs are present, one process is pinned to each GPU.

**Run** from this directory:

```bash
# CPU, or a single GPU
uv run torchrun --nproc_per_node=1 fsdp2_train.py

# match --nproc_per_node to the number of GPUs
uv run torchrun --nproc_per_node=2 fsdp2_train.py
```

Rank 0 prints five `step N loss ...` lines. `torchrun` sets `LOCAL_RANK`, the world size, and the rendezvous addresses that `init_process_group()` reads. Create the optimizer after `fully_shard`, which this script already does: sharding replaces parameters, and an optimizer built earlier would keep the unsharded tensors.

## TorchTitan context parallelism

`my_recipes.py`, `launch_llama3_8b_cp2.sh`, `run_train.sh`, and `dry_run_fake_backend.sh` are the article's TorchTitan snippets. They run from a checkout of [pytorch/torchtitan](https://github.com/pytorch/torchtitan), not from this folder.

**Version.** This repo pins TorchTitan **v0.3.0**, the release that targets stable torch 2.14, so it installs into this repo's ROCm 10.1 uv environment (`torch 2.14.0+rocm10.1.0`). TorchTitan `main` and v0.4.0 need a nightly torch and do not install against these pins.

### Setup

Put the TorchTitan checkout next to this repo, so `../torchtitan` from the repo root is the checkout:

```bash
# in the folder that contains pytorch-2026-learnings
git clone --branch v0.3.0 https://github.com/pytorch/torchtitan

cd pytorch-2026-learnings
uv sync --extra titan                                    # adds torchtitan==0.3.0 to .venv
cp 01-distributed-training-and-parallelism/my_recipes.py ../torchtitan/
```

`uv sync --extra titan` installs TorchTitan 0.3.0 next to the ROCm build of PyTorch, using the `override-dependencies` in `pyproject.toml` (`datasets>=5.1.0`, `spmd-types==0.2.3`). Plain `uv sync` leaves TorchTitan out. Do not `pip install torchtitan`: that fetches the latest release, which needs a nightly torch.

Launch every TorchTitan command from the checkout root and point `uv run` at this repo's environment:

```bash
cd ../torchtitan
NGPU=1 uv run --project ../pytorch-2026-learnings ./run_train.sh --training.local_batch_size 2 --training.seq_len 512
```

`--project` is needed because the TorchTitan checkout has its own `pyproject.toml`. A plain `uv run` there would create a second environment with the CPU-only torch from PyPI. Launching from the checkout root matters too: the debug model's tokenizer path (`./tests/assets/tokenizer`) is relative, and the trainer imports `my_recipes` from the working directory.

**16 GB GPUs.** The default debug run (no extra flags) peaks at about 13.7 GiB. Starting it while another GPU job held 14 GB froze the test machine. Stop other GPU jobs first (`uv run amd-smi metric -m` shows used VRAM), or keep `--training.local_batch_size 2 --training.seq_len 512`, which fits in about 0.43 GiB.

**Tested** on one Radeon RX 9070 XT (`gfx1201`, 16 GB) with `torch 2.14.0+rocm10.1.0` and TorchTitan v0.3.0:

| Command | Result |
|---|---|
| `NGPU=1 ./run_train.sh` (default debug model) | `Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=1, cp=1, tp=1, ep=1`; step 1 loss 8.03748, step 10 loss 4.12329, memory 13.74GiB (86.27%), 14,647 tps |
| same, with `--training.local_batch_size 2 --training.seq_len 512` | step 10 loss 4.65166, memory 0.43GiB |
| `dry_run_fake_backend.sh` (`llama3_debugmodel_cp2`) | mesh with `cp=2`, one step, memory 7.26GiB |

The 2-GPU launch (`launch_llama3_8b_cp2.sh`) was not run: it needs 2 GPUs and the Llama 3.1 8B tokenizer.

## `my_recipes.py`

Two recipe functions for TorchTitan v0.3.0. `llama3_8b_cp2` starts from `llama3_8b()` in `torchtitan.models.llama3.config_registry` and sets `context_parallel_degree = 2`. `llama3_debugmodel_cp2` does the same for the tiny debug model, so a context-parallel config can be checked on a small GPU with the tokenizer bundled in the checkout. In v0.3.0 the trainer swaps in the context-parallel attention itself when the degree is above 1, so the recipe only sets the degree.

**Source session:** "TorchTitan GraphTrainer: LLM Distributed Training with PyTorch's Compiler Toolkit" (Sanket Jayant Purandare, Sherlock Huang).

**Requirements:** the TorchTitan v0.3.0 checkout described above, with this file copied to its root. A real `llama3_8b_cp2` launch also needs the Llama 3.1 tokenizer at `./assets/hf/Llama-3.1-8B`, downloaded from the checkout root with:

```bash
uv run --project ../pytorch-2026-learnings python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
```

## `launch_llama3_8b_cp2.sh`

The article's launch line:

```bash
NGPU=2 MODULE=my_recipes CONFIG=llama3_8b_cp2 ./run_train.sh
```

**Source session:** "TorchTitan GraphTrainer: LLM Distributed Training with PyTorch's Compiler Toolkit" (Sanket Jayant Purandare, Sherlock Huang).

**Run** from the TorchTitan checkout root, after `my_recipes.py` is in that directory, on a machine with 2 GPUs:

```bash
uv run --project ../pytorch-2026-learnings bash ../pytorch-2026-learnings/01-distributed-training-and-parallelism/launch_llama3_8b_cp2.sh
```

`./run_train.sh` in that command is TorchTitan's script, resolved from the checkout's working directory. `NGPU` becomes `--nproc_per_node` and must equal the product of every parallelism degree; the upstream default is 8. You should see a device-mesh line with `cp=2`, then per-step loss, tokens per second, and peak memory.

## `run_train.sh`

The article's excerpt of TorchTitan v0.3.0's [`run_train.sh`](https://github.com/pytorch/torchtitan/blob/v0.3.0/run_train.sh), without the license header and `set -ex`: defaults for `NGPU`, `LOG_RANK`, `MODULE` (`llama3`), `CONFIG` (`llama3_debugmodel`) and `COMM_MODE`, then either a single `python3 -m torchtitan.train ... --comm.mode=${COMM_MODE} --training.steps 1` process when `COMM_MODE` is set, or `torchrun ... -m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@"` with `PYTORCH_ALLOC_CONF=expandable_segments:True`.

**Source session:** "TorchTitan GraphTrainer: LLM Distributed Training with PyTorch's Compiler Toolkit" (Sanket Jayant Purandare, Sherlock Huang).

This copy is for reading. Run the checkout's own `run_train.sh` from the checkout root, as shown in Setup. `LOG_RANK` selects which ranks print to the terminal. Port `0` in the rendezvous endpoint lets the OS pick a free port. Extra arguments after `./run_train.sh` pass through to the trainer.

## `dry_run_fake_backend.sh`

The article's config check:

```bash
NGPU=2 COMM_MODE=fake_backend MODULE=my_recipes CONFIG=llama3_debugmodel_cp2 ./run_train.sh
```

**Source session:** "TorchTitan GraphTrainer: LLM Distributed Training with PyTorch's Compiler Toolkit" (Sanket Jayant Purandare, Sherlock Huang).

**Run** from the TorchTitan checkout root, with `my_recipes.py` beside `run_train.sh`:

```bash
uv run --project ../pytorch-2026-learnings bash ../pytorch-2026-learnings/01-distributed-training-and-parallelism/dry_run_fake_backend.sh
```

`COMM_MODE=fake_backend` makes `run_train.sh` skip `torchrun` and start one process with fake process groups for exactly one step. `NGPU` is the logical world size, so TorchTitan builds the same device mesh as a 2-GPU job (`cp=2`). It catches degrees that do not multiply to `NGPU` and shapes that do not divide. Collectives do no real communication, so it says nothing about multi-rank speed or numerical agreement. `COMM_MODE=local_tensor` is the other option: it simulates all ranks on one GPU.

The dry run peaks at about 7.3 GiB on one GPU, so stop other GPU jobs on a 16 GB card first. To check the 8B recipe itself, change `CONFIG` to `llama3_8b_cp2`. That process still allocates one rank's weights, gradients, optimizer state, and activations, which for 8B over two ranks is tens of GB, so it needs a large GPU such as an MI300X.
