#!/usr/bin/env python3
"""Head-to-head comparison: transformers vs vLLM(raw) vs vLLM(token_ids)"""
import json, os

os.environ.setdefault("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = "/data/anaconda3/lib:" + os.environ["LD_LIBRARY_PATH"]

from transformers import AutoTokenizer
import torch

MODEL_PATH = "/data/models/Qwen3-4B"
MSG = [{"role": "user", "content": "Hello, introduce yourself in one sentence."}]

# Load tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Format with chat template (same as verl server does)
formatted_prompt = tokenizer.apply_chat_template(MSG, tokenize=False, add_generation_prompt=True, enable_thinking=False)
prompt_ids = tokenizer.encode(formatted_prompt)

print("=== CHAT TEMPLATE OUTPUT ===")
print(formatted_prompt[:500])
print(f"\nPrompt token count: {len(prompt_ids)}")

# Test 1: transformers (gold standard)
print("\n=== TEST 1: Transformers ===")
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
text1 = tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(f"Response: {repr(text1[:200])}")
coherent1 = any(w in text1.lower() for w in ["hello", "assistant", "ai", "language", "model", "i am", "i'm", "qwen"])
print(f"Coherent: {coherent1}")
del model; torch.cuda.empty_cache()

# Test 2: vLLM with raw text (already confirmed working)
print("\n=== TEST 2: vLLM with raw text ===")
from vllm import LLM, SamplingParams
llm = LLM(model=MODEL_PATH, gpu_memory_utilization=0.5, max_model_len=4096, trust_remote_code=True)
out2 = llm.generate([formatted_prompt], SamplingParams(temperature=0.7, max_tokens=100))
text2 = out2[0].outputs[0].text
print(f"Response: {repr(text2[:200])}")
coherent2 = any(w in text2.lower() for w in ["hello", "assistant", "ai", "language", "model", "i am", "i'm", "qwen"])
print(f"Coherent: {coherent2}")
del llm; torch.cuda.empty_cache()

# Test 3: vLLM with explicit token IDs (simulating verl path)
print("\n=== TEST 3: vLLM with token IDs (verl simulation) ===")
llm3 = LLM(model=MODEL_PATH, gpu_memory_utilization=0.5, max_model_len=4096, trust_remote_code=True)
out3 = llm3.generate(
    [{"prompt_token_ids": prompt_ids}],
    SamplingParams(temperature=0.7, max_tokens=100)
)
token_ids_3 = out3[0].outputs[0].token_ids
text3 = tokenizer.decode(token_ids_3, skip_special_tokens=True)
print(f"Response: {repr(text3[:200])}")
coherent3 = any(w in text3.lower() for w in ["hello", "assistant", "ai", "language", "model", "i am", "i'm", "qwen"])
print(f"Coherent: {coherent3}")

# Summary
print("\n=== SUMMARY ===")
results = {"test1_transformers": coherent1, "test2_vllm_raw": coherent2, "test3_vllm_tokenids": coherent3}
print(json.dumps(results, indent=2))

with open("/tmp/diag_compare.json", "w") as f:
    json.dump({
        "results": results,
        "t1_text": text1,
        "t2_text": text2,
        "t3_text": text3,
        "t1_len": len(text1),
        "t2_len": len(text2),
        "t3_len": len(text3),
    }, f, ensure_ascii=False, indent=2)

print("Done! Results saved to /tmp/diag_compare.json")
