#!/usr/bin/env python3
"""Standalone vLLM test - bypass verl entirely."""
from vllm import LLM, SamplingParams

print("Loading Qwen3-4B with standalone vLLM...")
llm = LLM(
    model="/data/models/Qwen3-4B",
    gpu_memory_utilization=0.5,
    max_model_len=4096,
    trust_remote_code=True,
)

print("Generating...")
outputs = llm.generate(
    ["Hello, introduce yourself in one sentence."],
    SamplingParams(temperature=0.7, max_tokens=100)
)

text = outputs[0].outputs[0].text
print(f"Response length: {len(text)}")
print(f"First 200 chars: {repr(text[:200])}")

# Check coherence
coherent = any(w in text.lower() for w in ["hello", "assistant", "ai", "language", "model", "i am", "i'm", "qwen"])
print(f"Coherent: {coherent}")

# Save for analysis
with open("/tmp/vllm_standalone.txt", "w") as f:
    f.write(text)

print("Done!")
