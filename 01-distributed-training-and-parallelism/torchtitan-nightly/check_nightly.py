# check_nightly.py
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
import inspect

print("torch", torch.__version__, "HIP", torch.version.hip)
print("GPU available", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("device", props.name, props.gcnArchName)

# APIs that TorchTitan main uses and stable torch 2.14 / the 2.15 rocm10.1 nightly lack
from torch.distributed.pipelining import analyze_pipeline_activation_liveness, PipelineStageInfo
print("pipelining liveness API ok")
assert "param_dtype_override_fn" in inspect.signature(MixedPrecisionPolicy).parameters
print("FSDP MixedPrecisionPolicy.param_dtype_override_fn ok")

from torchtitan.config.transform import ContextParallelTransform
from torchtitan.models.common.attention.cp_attention import AllGatherKVVarlenInnerAttention
print("TorchTitan ContextParallelTransform + AllGatherKVVarlenInnerAttention ok")
