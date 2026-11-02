# vllm_serve.sh
pip install vllm
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching
