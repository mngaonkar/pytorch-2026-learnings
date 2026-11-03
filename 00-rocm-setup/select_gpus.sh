# select_gpus.sh
uv run rocminfo | grep gfx                           # GPU architecture targets on this machine
uv run amd-smi list                                  # GPU index, BDF and UUID
HIP_VISIBLE_DEVICES=0 uv run python rocm_check.py    # HIP programs see only GPU 0
NGPU=${NGPU:-1}                                      # GPUs for the FSDP2 run; NGPU=1 on a single-GPU machine
ROCR_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU - 1))) uv run torchrun --nproc_per_node=$NGPU ../01-distributed-training-and-parallelism/fsdp2_train.py
