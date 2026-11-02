# my_recipes.py  (TorchTitan main, ROCm nightly torch)
from torchtitan.components.data import GrainDataLoader
from torchtitan.config.transform import apply_transforms, ContextParallelTransform
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.attention.cp_attention import (
    AllGatherKVVarlenInnerAttention,
)
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests import _set_spmd_typechecking
from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel


def llama3_debugmodel_cp2(seq_len: int = 2048) -> Trainer.Config:
    """Llama 3 debug model, varlen attention, context parallel degree 2.

    Every rank holds seq_len / 2 query tokens. AllGatherKVVarlenInnerAttention
    all-gathers K and V across the CP group so each rank's queries can attend
    to the full (document-masked) sequence.
    """
    config = llama3_debugmodel(seq_len=seq_len, attn_backend="varlen")
    # Packed varlen metadata lacks SPMD annotations (same as upstream tests).
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.context_parallel_degree = 2
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    config.dataloader.max_num_documents = 64
    config.training.steps = 5
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    VarlenInnerAttention: AllGatherKVVarlenInnerAttention
                }
            )
        ],
    )
