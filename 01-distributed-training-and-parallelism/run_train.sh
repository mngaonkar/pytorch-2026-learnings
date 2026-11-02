NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"torchtitan_recipes.tests.models.llama3"}
CONFIG=${CONFIG:-"llama3_debugmodel"}
ENTRYPOINT=${ENTRYPOINT:-"torchtitan.train"}

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
  -m ${ENTRYPOINT} --module ${MODULE} --config ${CONFIG} "$@"
