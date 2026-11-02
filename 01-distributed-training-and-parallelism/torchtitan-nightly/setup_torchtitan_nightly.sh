# setup_torchtitan_nightly.sh
set -e
TITAN_COMMIT=83abb2c039ebeffdef27907ea7e33b6174061651   # same commit as pyproject.toml
if [ ! -d torchtitan ]; then
    git clone https://github.com/pytorch/torchtitan torchtitan
fi
git -C torchtitan checkout --quiet "$TITAN_COMMIT"   # run_train.sh, tests/assets tokenizer and c4_test data
uv sync                                              # nightly torch, ROCm 10.3 wheels, TorchTitan at the same commit
uv run python check_nightly.py
