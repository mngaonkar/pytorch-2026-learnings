TORCH_LOGS=graph_breaks python train.py   # where and why graphs split
TORCH_LOGS=recompiles python train.py     # shape changes that trigger recompiles
