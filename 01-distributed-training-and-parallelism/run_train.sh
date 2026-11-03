#!/usr/bin/env bash
set -e

NGPU=${NGPU:-"1"}
export LOG_RANK=${LOG_RANK:-0}
ENTRYPOINT=${ENTRYPOINT:-"torchtitan.train"}

export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-"12.0.1"}
export PYTORCH_ALLOC_CONF="expandable_segments:True"

CONFIG=${CONFIG:-"llama3_8b"}
TOKENIZER_DIR="$(pwd)/assets/tokenizer"

if [[ -f "${CONFIG}" ]]; then
  CONFIG_ARGS=(--job.config-file "${CONFIG}")
else
  CONFIG_ARGS=(
    --model.name llama3
    --model.flavor 8b
    --model.hf-assets-path "${TOKENIZER_DIR}"
    --training.dataset c4_validation
    --training.steps 10
    --training.local-batch-size 1
    --training.seq-len 1024
    --parallelism.data-parallel-replicate-degree 1
    --parallelism.data-parallel-shard-degree 1
    --parallelism.tensor-parallel-degree 1
    --parallelism.pipeline-parallel-degree 1
  )
fi

torchrun --nproc_per_node="${NGPU}" \
  --rdzv_backend c10d \
  --rdzv_endpoint="localhost:0" \
  --local-ranks-filter "${LOG_RANK}" \
  --role rank \
  --tee 3 \
  -m "${ENTRYPOINT}" \
  "${CONFIG_ARGS[@]}" \
  "$@"

