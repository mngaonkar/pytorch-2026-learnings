# 5. Profiling and data loading

A GPU only helps when the next batch is ready. If file reads, decode, or the copy onto the device take longer than one compute step, the device waits. Profile a few steady-state steps before changing the pipeline, then time one loader knob at a time.

## Compute-bound vs input-bound

```
Compute-bound: data ready  ->  GPU busy GPU busy GPU busy
Input-bound:   GPU busy .... waiting for data .... GPU busy .... waiting ....
```

**Source session:** "Five Bottlenecks, Five Fixes: Systematic Performance Diagnosis for PyTorch Distributed Training" (Paulo Aragao, Aravind Neelakantan).

The gaps in the second line are the input pipeline. The three scripts below record operator time, sweep data-loader workers, and overlap the host-to-device copy.

## `profile_steps.py`

Trains a tiny classifier for six steps under a profiler schedule of wait 1, warmup 1, active 3, repeat 1. It writes a trace to `./log` and prints the top operators by self CPU time.

**Source session:** "What You Cannot Profile, You Cannot Optimize: Learning to Read PyTorch Traces" (Suvaditya Mukherjee).

**Requirements:** PyTorch. CPU or GPU. The script uses `torch.accelerator` when one exists and CPU otherwise. No download. `tensorboard` is optional and is only needed to view the trace (`tensorboard --logdir ./log`). The JSON trace can also be opened in Perfetto. `./log` is gitignored. `prof.step()` is what the schedule counts; the wait and warmup steps stay out of the table so startup allocation does not dominate.

**Run** from this directory:

```bash
python profile_steps.py
```

Read self-time (time inside the operator) and the call count. Large CPU self-time with little accelerator time points at Python or launch overhead.

## `bench_num_workers.py`

A dataset of 512 items that sleeps 2 ms in `__getitem__` to stand in for a file read and decode. It times one epoch at `num_workers` of 0, 2, 4, and 8, with `persistent_workers` and `prefetch_factor=2` whenever workers are used.

**Source session:** "Scaling PyTorch Dataloaders 10x: A Case Study" (Ian Stenbit, Christine Cheng, Dylan Doblar).

**Requirements:** PyTorch. CPU is enough; the sleep is the bottleneck on purpose. `pin_memory` turns on only when `torch.accelerator.is_available()` is true. No download. The `if __name__ == "__main__"` guard is required so spawned workers do not start the sweep again. The sweep uses up to 8 processes.

**Run** from this directory:

```bash
python bench_num_workers.py
```

Epoch time should fall as workers increase, then flatten. The article's CPU run went from 1.13 s to 0.19 s per epoch between 0 and 8 workers. Keep the smallest worker count that reaches that floor. `prefetch_factor` has to be `None` when `num_workers` is 0, which the script already does.

## `overlap_host_to_device.py`

Loads an in-memory tensor dataset. When an accelerator exists, the loader pins each batch and `.to(device, non_blocking=True)` returns while the copy is still queued, so the copy can overlap other device work. Later ops that read the batch wait for the copy.

**Source session:** "Scaling PyTorch Dataloaders 10x: A Case Study" (Ian Stenbit, Christine Cheng, Dylan Doblar).

**Requirements:** PyTorch. CPU or GPU. On CPU, `pin_memory` stays off and the copy target is CPU. On a GPU, pinning matters: `non_blocking=True` overlaps only when the source memory is page-locked. The loader pins one batch at a time. No download.

**Run** from this directory:

```bash
python overlap_host_to_device.py
```

The script prints `last loss` and a number. The overlap itself shows up on a profiler timeline as host-to-device copies sitting beside kernels.
