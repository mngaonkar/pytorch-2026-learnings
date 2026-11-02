# 2. Compilers and kernels

Eager PyTorch launches one kernel per operation and writes every intermediate result back to memory. Fusion keeps those values on chip. `torch.compile` records a graph and Inductor generates fused kernels, often in Triton. A graph break falls back to eager for that piece.

## Eager vs fused

```
Eager:   [read x] add [write] -> [read] mul [write] -> [read] relu [write]
Fused:   [read x]  add -> mul -> relu  [write]      (one memory trip)
```

General background on operator fusion, not from a single session. The fused line is one memory round-trip for the same add, multiply, and ReLU.

## `bench_eager_vs_compiled.py`

Times a two-layer MLP in eager mode and under `torch.compile`, after five warmup calls, and prints the max absolute difference of the outputs.

**Source session:** "How PyTorch Powers Open Foundational Models — From Language to Vision" (Songhao Jia, Laura Gustafson, Elisa Cascardi).

**Requirements:** PyTorch. CPU or GPU. The script selects `torch.accelerator` when one is available and CPU otherwise. No download. The first compiled call traces and generates code, so the warmup runs before the clock starts. GPU work is asynchronous; the script synchronizes before reading the clock when an accelerator is present. On CPU, Inductor compiles the graph to C++, so the machine needs `g++` and Python development headers (`Python.h`).

**Run** from this directory:

```bash
python bench_eager_vs_compiled.py
```

Expect two times in milliseconds and a small max diff. On a model this small the gap can be minor. `mode="max-autotune"` is mentioned in the source comment when you want a longer kernel search. Keep input shapes stable across calls so the compiled graph is reused.

## `find_graph_breaks.py`

Runs `torch._dynamo.explain` on a function whose `print` cannot be traced, then prints the graph count, the break count, and each reason.

**Source session:** "Advancing torch.compile for Verifiable Precision & Dynamic Shapes" (Xiyuan Wang).

**Requirements:** PyTorch. CPU is enough. No GPU and no download.

**Run** from this directory:

```bash
python find_graph_breaks.py
```

Expect `graphs: 2 breaks: 1`, then a reason that mentions the builtin `print`. Each break is a place where two smaller graphs cannot be fused.

## `torch_logs.sh`

Runs the two compiler snippets in this folder under PyTorch's compile logs:

```bash
# torch_logs.sh
TORCH_LOGS=graph_breaks python find_graph_breaks.py         # where and why graphs split
TORCH_LOGS=recompiles python bench_eager_vs_compiled.py     # shape changes that trigger recompiles
```

**Source session:** "Advancing torch.compile for Verifiable Precision & Dynamic Shapes" (Xiyuan Wang).

**Requirements:** PyTorch. Both commands use the scripts next to this file: `find_graph_breaks.py` for graph breaks, and `bench_eager_vs_compiled.py` for recompiles. CPU is enough. The recompile command follows the same CPU C++ build requirement as the benchmark (`g++` and `Python.h`). `graph_breaks` logs the file, line, and reason of each break. `recompiles` logs each guard that failed, such as a tensor size change.

**Run** from this directory:

```bash
bash torch_logs.sh
```

## `triton_vector_add.py`

A Triton kernel: one program per 1024 elements adds two length-1,000,000 vectors, masks the tail, and checks `torch.allclose` against `x + y`.

**Source session:** "KernelAgent: Hardware-Guided GPU Kernel Optimization via Multi-Agent Orchestration" (Kaiming Cheng, Laura Wang).

**Requirements:** PyTorch, the `triton` package, and a GPU. The script calls `torch.accelerator.current_accelerator()` directly, so a CPU-only machine cannot run it. Install Triton by uncommenting `triton` in the top-level `requirements.txt`, or use a GPU PyTorch build that already includes it. No model download.

**Run** from this directory, on a GPU machine:

```bash
python triton_vector_add.py
```

Success is silent. The `assert` passing means the kernel matches `x + y`.
