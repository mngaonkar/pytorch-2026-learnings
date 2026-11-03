# dry_run_cp2_fake.sh
cd "$(dirname "$0")/torchtitan"                      # launch from the checkout root: tests/assets paths are relative
PYTHONPATH=.. NGPU=2 COMM_BACKEND=fake MODULE=my_recipes CONFIG=llama3_debugmodel_cp2 \
    uv run --project .. ./run_train.sh "$@"
