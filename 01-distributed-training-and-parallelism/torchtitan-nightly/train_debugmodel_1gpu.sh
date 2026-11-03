# train_debugmodel_1gpu.sh
cd "$(dirname "$0")/torchtitan"                      # launch from the checkout root: tests/assets paths are relative
NGPU=${NGPU:-1} uv run --project .. ./run_train.sh "$@"
