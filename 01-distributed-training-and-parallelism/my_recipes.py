# my_recipes.py
from torchtitan.config.transform import apply_transforms, ContextParallelTransform
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.attention.cp_attention import AllGatherKVVarlenInnerAttention
from torchtitan_recipes.models.llama3 import llama3_8b

def llama3_8b_cp2():
    config = llama3_8b()
    config.parallelism.context_parallel_degree = 2   # split each sequence over 2 GPUs
    # replace the attention with its context-parallel version (all-gather K/V)
    return apply_transforms(config, [ContextParallelTransform(
        inner_attention_map={VarlenInnerAttention: AllGatherKVVarlenInnerAttention})])
