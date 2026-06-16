"""Test tokenizer loading for Qwen3.5-4B."""
from transformers import AutoTokenizer

for path in [
    "/data/models/Qwen3.5-4B/Qwen/Qwen3.5-4B",
    "/data/models/Qwen3.5-4B/Qwen/Qwen3___5-4B",
]:
    try:
        t = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        print(f"OK: {path} → {type(t).__name__}, vocab={len(t)}")
    except Exception as e:
        print(f"FAIL: {path} → {e}")
