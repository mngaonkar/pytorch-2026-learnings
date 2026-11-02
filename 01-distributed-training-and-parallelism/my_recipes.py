# my_recipes.py
from torchtitan.models.llama3.config_registry import llama3_8b, llama3_debugmodel

def llama3_8b_cp2():
    config = llama3_8b()
    config.parallelism.context_parallel_degree = 2   # split each sequence over 2 GPUs
    return config

def llama3_debugmodel_cp2():
    config = llama3_debugmodel()                     # tiny model, for checking CP on small hardware
    config.parallelism.context_parallel_degree = 2
    return config
