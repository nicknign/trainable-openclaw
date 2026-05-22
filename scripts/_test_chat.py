#!/usr/bin/env python3
"""Quick test script for the veRL inference server."""
import urllib.request
import json

data = json.dumps({
    "model": "Qwen3-4B",
    "messages": [{"role": "user", "content": "Hello, introduce yourself in one sentence."}],
    "max_tokens": 100,
    "enable_thinking": False
}).encode()

req = urllib.request.Request(
    "http://localhost:8000/v1/chat/completions",
    data=data,
    headers={"Content-Type": "application/json"}
)
resp = urllib.request.urlopen(req, timeout=120)
result = json.loads(resp.read())
print(json.dumps(result, ensure_ascii=False, indent=2))
