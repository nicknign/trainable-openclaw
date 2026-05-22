#!/usr/bin/env python3
"""Diagnose chat output - save response and analyze token content."""
import urllib.request
import json
import os

# -------- Test 1: normal chat --------
print("=== Test 1: Normal chat ===")
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

# Save full response
with open("/tmp/chat_full.json", "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

content = result["choices"][0]["message"]["content"]
print(f"Content length: {len(content)} chars")
print(f"Content repr: {repr(content[:300])}")
print(f"Token count: {result['usage']}")

# Check for common indicators of normal output
if "Hello" in content or "I am" in content.lower() or "I'm" in content.lower():
    print("RESULT: OK - English response detected")
else:
    print("RESULT: GARBAGE - no recognizable English patterns found")

# Check character distribution
ascii_count = sum(1 for c in content if ord(c) < 128)
total = len(content)
print(f"ASCII ratio: {ascii_count}/{total} = {ascii_count/total*100:.1f}%")
