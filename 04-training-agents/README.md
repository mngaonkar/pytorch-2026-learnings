# 4. Training agents

An agent is a model in a loop: think, call a tool, read the result, repeat. The harness is the code around the model (which tools exist, how the prompt is built, when to stop). Reinforcement learning then favors tries that score higher on a reward you can check.

## Harness, environment, reward

```
Task -> model acts through the HARNESS -> ENVIRONMENT runs it -> REWARD (e.g. tests pass)
  ^                                                                  |
  +---------------- update the model: favor higher-reward tries -----+
```

**Source session:** "Train the Agent, Not Just the Model" (Sergio Paniego Blanco).

The same model can score very differently in two harnesses. The snippet below trains on a reward that is an exact match, which is the same idea as a unit test: pass or fail, with a known answer.

## `grpo_exact_match.py`

GRPO on 64 addition prompts (`1+1` through `8+8`). `exact_match` returns 1.0 only when the completion contains `<answer>N</answer>` and `N` is the true sum, otherwise 0.0. The trainer samples `num_generations=4` completions per prompt and scores them against the group. `max_steps=10`.

**Source session:** "Train the Agent, Not Just the Model" (Sergio Paniego Blanco).

**Requirements:** PyTorch, `datasets`, `transformers`, and `trl`. `trl` is optional in the top-level `requirements.txt`; uncomment it or run `pip install trl datasets`. The script loads `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face, so the first run downloads that model. The article's in-file comment shows the `GRPOConfig` flags for a CPU-only machine: `bf16=False, use_cpu=True`. For a short pipeline check, the article suggests swapping the model id for `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`. Default settings (`per_device_train_batch_size=4`, `num_generations=4`, `max_completion_length=32`) assume a GPU that can hold those generations. Checkpoints and logs are written to `grpo-add/`, which is gitignored.

**Run** from this directory when you are ready to download a model:

```bash
python grpo_exact_match.py
```

Per-step logs include `reward` and `reward_std`. Reward should rise over more steps. A `reward_std` of 0 means the four completions scored the same, so that step had no ranking signal. If the run runs out of memory, lower the batch size, `num_generations`, or `max_completion_length`.
