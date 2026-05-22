#!/usr/bin/env python3
"""Test Qwen3-4B with pure transformers (no vLLM) to isolate the issue."""
import json, os, sys

# Fix libstdc++ for conda on older Linux
os.environ.setdefault("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = "/data/anaconda3/lib:" + os.environ["LD_LIBRARY_PATH"]

print("Loading model...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/data/models/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Model loaded. Generating...", flush=True)

messages = [{"role": "user", "content": "Hello, introduce yourself in one sentence."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)

response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

result = {
    "prompt": prompt,
    "response": response,
    "response_len": len(response),
    "is_coherent": any(word in response.lower() for word in ["hello", "assistant", "ai", "language", "model", "i am", "i'm"]),
}

with open("/tmp/xfmer_test.json", "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Response length: {len(response)}", flush=True)
print(f"Coherent: {result['is_coherent']}", flush=True)
print(f"First 100 chars repr: {repr(response[:100])}", flush=True)
print("Done. Saved to /tmp/xfmer_test.json", flush=True)
