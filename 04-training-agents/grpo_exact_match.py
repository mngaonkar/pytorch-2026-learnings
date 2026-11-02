# grpo_exact_match.py  -- pip install trl datasets
import re
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# Tiny arithmetic task with a known answer for every prompt
rows = [{"prompt": f"What is {a} + {b}? Reply as <answer>N</answer>.", "answer": str(a + b)}
        for a in range(1, 9) for b in range(1, 9)]
dataset = Dataset.from_list(rows)

def exact_match(completions, answer, **kwargs):
    # extra dataset columns (here "answer") are passed in as keyword arguments
    rewards = []
    for completion, target in zip(completions, answer):
        m = re.search(r"<answer>\s*(-?\d+)\s*</answer>", completion)
        rewards.append(1.0 if m and m.group(1) == target else 0.0)  # verifiable pass/fail
    return rewards

args = GRPOConfig(
    output_dir="grpo-add",
    per_device_train_batch_size=4,
    num_generations=4,          # completions sampled per prompt, compared as a group
    max_completion_length=32,
    max_steps=10,
    logging_steps=1,
    # CPU-only test machine? add: bf16=False, use_cpu=True
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=exact_match,
    args=args,
    train_dataset=dataset,
)
trainer.train()
