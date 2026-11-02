# torch_logs.sh
TORCH_LOGS=graph_breaks python find_graph_breaks.py         # where and why graphs split
TORCH_LOGS=recompiles python bench_eager_vs_compiled.py     # shape changes that trigger recompiles
