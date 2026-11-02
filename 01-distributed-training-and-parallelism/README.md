# 1. Distributed training and parallelism

Two ways to split a training step, copied from the article. `fsdp2_train.py` shards a small model with FSDP2 and runs on CPU or GPU. The TorchTitan files configure context parallelism and have to live in a TorchTitan checkout before they will run.

## `fsdp2_train.py`

Builds a four-block residual MLP, calls `fully_shard` on each block and then on the root, creates AdamW after sharding, and trains for five steps on random inputs. The file comment in the article names this script `train.py`.

**Source session:** "Transformers.distributed: Bringing First-Class Distributed Training to 2.4M+ Hugging Face Models" (Ferdinand Mom).

**Requirements:** PyTorch (`torch.distributed.fsdp.fully_shard` and `torchrun`). No extra packages and no download. CPU is enough to exercise the sharding logic. When GPUs are present, one process is pinned to each GPU.

**Run** from this directory:

```bash
# CPU, or a single GPU
torchrun --nproc_per_node=1 fsdp2_train.py

# match --nproc_per_node to the number of GPUs
torchrun --nproc_per_node=2 fsdp2_train.py
```

Rank 0 prints five `step N loss ...` lines. `torchrun` sets `LOCAL_RANK`, the world size, and the rendezvous addresses that `init_process_group()` reads. Create the optimizer after `fully_shard`, which this script already does: sharding replaces parameters, and an optimizer built earlier would keep the unsharded tensors.

## TorchTitan context parallelism

`my_recipes.py`, `launch_llama3_8b_cp2.sh`, `run_train.sh`, and `dry_run_fake_backend.sh` are the article's TorchTitan snippets. They need a checkout of [pytorch/torchtitan](https://github.com/pytorch/torchtitan) and that repo's own install. They do not run from this folder alone.

**Where `my_recipes.py` goes.** Copy it into the TorchTitan repository root, the directory that contains upstream `run_train.sh`. Launch commands set `MODULE=my_recipes`, and the trainer imports that name. Starting the trainer with `python -m` puts the working directory on the import path, so `my_recipes.py` has to sit in the directory you launch from.

`NGPU` must equal the number of GPUs you have, and it must equal the product of every parallelism degree in the recipe. Upstream `run_train.sh` defaults to `NGPU=8`. Set `NGPU` on the command line on any smaller machine.

`llama3_8b_cp2` sets `context_parallel_degree = 2` and leaves every other degree as `llama3_8b()` defined it. The real Llama 3.1 recipe expects its tokenizer at `./assets/hf/Llama-3.1-8B` inside the checkout. The article's download command is:

```bash
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
```

A single-GPU debug launch uses the script defaults (`MODULE=torchtitan_recipes.tests.models.llama3`, `CONFIG=llama3_debugmodel`), which ship a tiny model and a tokenizer inside the repo:

```bash
NGPU=1 ./run_train.sh
```

Leave `context_parallel_degree` at 1 for that debug path. `NGPU=1` debug was verified on a single 16 GB consumer GPU, where the loss went from 8.13 to 4.03 over 10 steps.

## `my_recipes.py`

Recipe function `llama3_8b_cp2`. It starts from `llama3_8b()`, sets `context_parallel_degree = 2`, and uses `ContextParallelTransform` to replace `VarlenInnerAttention` with `AllGatherKVVarlenInnerAttention`. TorchTitan requires that attention swap whenever the context-parallel degree is greater than 1. Each rank holds activations for part of the sequence, all-gathers keys and values, and runs attention.

**Source session:** "TorchTitan GraphTrainer: LLM Distributed Training with PyTorch's Compiler Toolkit" (Sanket Jayant Purandare, Sherlock Huang).

**Requirements:** the TorchTitan checkout described above, with this file copied to the repo root. A real 2-GPU launch also needs the Llama 3.1 tokenizer assets. No GPU is required to read the file.

## `launch_llama3_8b_cp2.sh`

The article's launch line:

```bash
NGPU=2 MODULE=my_recipes CONFIG=llama3_8b_cp2 ./run_train.sh
```

**Source session:** "TorchTitan GraphTrainer: LLM Distributed Training with PyTorch's Compiler Toolkit" (Sanket Jayant Purandare, Sherlock Huang).

**Run** from the TorchTitan checkout root, after `my_recipes.py` is in that directory, on a machine with 2 GPUs whose parallelism degrees multiply to 2:

```bash
bash /path/to/this/launch_llama3_8b_cp2.sh
```

`./run_train.sh` in that command is TorchTitan's script, resolved from the checkout's working directory. `NGPU` becomes `--nproc_per_node`. You should see a device-mesh line with `cp=2`, then per-step loss, tokens per second, and peak memory.

## `run_train.sh`

The article's excerpt of upstream [`run_train.sh`](https://github.com/pytorch/torchtitan/blob/main/run_train.sh): defaults for `NGPU`, `LOG_RANK`, `MODULE`, `CONFIG`, and `ENTRYPOINT`, then `torchrun` with `PYTORCH_ALLOC_CONF=expandable_segments:True`.

**Source session:** "TorchTitan GraphTrainer: LLM Distributed Training with PyTorch's Compiler Toolkit" (Sanket Jayant Purandare, Sherlock Huang).

Keep the checkout's own `run_train.sh` as the file you execute. This copy is the excerpt printed in the article. The upstream script also implements `COMM_BACKEND=fake`, which this excerpt leaves out. Run a real job from the checkout root so `./run_train.sh` is that upstream file. Defaults point at `llama3_debugmodel`, so `NGPU=1 ./run_train.sh` needs no checkpoint download. `LOG_RANK` selects which ranks print to the terminal. Port `0` in the rendezvous endpoint lets the OS pick a free port.

## `dry_run_fake_backend.sh`

The article's config check:

```bash
NGPU=2 COMM_BACKEND=fake MODULE=my_recipes CONFIG=llama3_8b_cp2 ./run_train.sh
```

**Source session:** "TorchTitan GraphTrainer: LLM Distributed Training with PyTorch's Compiler Toolkit" (Sanket Jayant Purandare, Sherlock Huang).

**Run** from the TorchTitan checkout root (upstream `run_train.sh` on `PATH`'s `./`), with `my_recipes.py` beside it. `COMM_BACKEND=fake` starts one process and builds the same device mesh as a 2-GPU job, with `NGPU` as the logical world size. Use it to catch degrees that do not multiply to `NGPU`, a missing context-parallel attention, or shapes that do not divide. Collectives do no real communication, so it says nothing about multi-rank speed or numerical agreement.

The fake process still allocates one rank's weights, gradients, optimizer state, and activations. An 8B model split over two ranks needs a large GPU for the dry run itself. On a smaller GPU, check recipe logic with `llama3_debugmodel` from `torchtitan_recipes.tests.models.llama3` first.
