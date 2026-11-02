# 3. Inference and serving

Serving a language model has two phases. Prefill reads the whole prompt and builds the KV cache. Decode then emits one token at a time and reads that cache on every step. vLLM manages the cache in blocks and can reuse a block when a new request starts with the same tokens.

## Prefill, decode, and the KV cache

```
Prompt --> [PREFILL: read everything, build KV cache] --> [DECODE: token, token, token ...]
                                                              ^ reads the KV cache each step
```

**Source session:** "State-of-the-Art KV Transfer for Disaggregated LLM Serving in vLLM" (Nicolò Lucchesi, Sunita Nadampalli, Zhanqiu Hu).

The cache grows with tokens, users, and layers. Prefix caching reuses blocks for any prompt that starts the same way, which is why the client below keeps the system prompt fixed and puts the user text last.

## `vllm_serve.sh`

Installs vLLM and serves `Qwen/Qwen2.5-0.5B-Instruct` with prefix caching turned on explicitly:

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching
```

**Source session:** "Lessons at Scale: Building a PyTorch Inference Stack with Open-Source Components" (Oscar Savolainen).

**Requirements:** a GPU with free memory for this 0.5B model, and network access so the first launch can download the weights from Hugging Face. `vllm` is an optional dependency in the top-level `requirements.txt`; this script installs it itself. The process is the server, so it stays in the foreground. Prefix caching is on by default in vLLM 0.32.0; the flag records the choice in the command.

**Run** from this directory when you intend to download the model and occupy a GPU:

```bash
bash vllm_serve.sh
```

Loading logs end with the server listening on port 8000. `curl localhost:8000/health` returns 200 when it is ready. If startup runs out of memory, the article points at lowering `--gpu-memory-utilization` and `--max-model-len`.

## `measure_ttft.py`

OpenAI-compatible client aimed at `http://localhost:8000/v1`. It asks two questions under one fixed system prompt, streams the reply, and prints time to first token, milliseconds per later chunk, and the chunk count.

**Source session:** "Making Enterprise Agentic Inference Production-Ready with PyTorch and vLLM" (Joseph Groenenboom, Tyler Michael Smith).

**Requirements:** the `openai` package (`pip install openai`, listed in `requirements.txt`). The vLLM server from `vllm_serve.sh` must already be listening. This process does not load a model; it only sends HTTP requests, so it is CPU-side. `api_key="EMPTY"` matches vLLM's default of not checking keys.

**Run** while the server is up:

```bash
python measure_ttft.py
```

Expect one line per question. The first request fills the cache; later requests that share the system prompt should start faster. The article records one example on a single 24 GB consumer GPU: TTFT was 54 ms on the first request and 13–26 ms after that. A timestamp or user id at the start of the system prompt changes the prefix and disables that reuse.
