# pip install openai   (server: vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching)
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# Stable prefix: identical on every request, so its KV cache can be reused
SYSTEM = "You are a support assistant for ExampleCo. Answer in two sentences."

def ask(question):
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},     # fixed part first
                  {"role": "user", "content": question}],    # changing part last
        max_tokens=128,                                     # cap output length
        stream=True,
    )
    ttft, n = None, 0
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0             # time to first token
            n += 1
    total = time.perf_counter() - t0
    per_tok = (total - ttft) / max(n - 1, 1)
    print(f"TTFT {ttft*1e3:.0f} ms, {per_tok*1e3:.1f} ms/token, {n} chunks")

for q in ["How do I reset my password?", "Where is my invoice?"]:
    ask(q)
