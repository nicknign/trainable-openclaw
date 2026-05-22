#!/usr/bin/env python3
"""Test: vLLM with explicit token IDs (simulating verl's communication pattern)."""
import os, json

os.environ.setdefault("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = "/data/anaconda3/lib:" + os.environ["LD_LIBRARY_PATH"]

print("Loading tokenizer...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/models/Qwen3-4B", trust_remote_code=True)

# Build prompt exactly as verl server does
MSG = [{"role": "user", "content": "Hello, introduce yourself in one sentence."}]

# Test with thinking disabled (same as our curl test)
for enable_thinking in [False, True]:
    print(f"\n{'='*60}")
    print(f"Testing enable_thinking={enable_thinking}")
    print(f"{'='*60}")

    formatted = tokenizer.apply_chat_template(
        MSG, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    prompt_ids = tokenizer.encode(formatted)

    print(f"Formatted prompt ({len(prompt_ids)} tokens):")
    print(repr(formatted[:300]))
    print(f"First 5 token IDs: {prompt_ids[:5]}")
    print(f"Last 5 token IDs: {prompt_ids[-5:]}")

    print("Loading vLLM...")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model="/data/models/Qwen3-4B",
        gpu_memory_utilization=0.5,
        max_model_len=4096,
        trust_remote_code=True,
    )

    print("Generating with token IDs...")
    outputs = llm.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(temperature=0.7, max_tokens=100)
    )

    token_ids = outputs[0].outputs[0].token_ids
    text = tokenizer.decode(token_ids, skip_special_tokens=True)

    print(f"Output token count: {len(token_ids)}")
    print(f"First 5 output IDs: {token_ids[:5]}")
    print(f"Decoded text: {repr(text[:200])}")

    coherent = any(w in text.lower() for w in ["hello", "assistant", "ai", "language", "model", "i am", "i'm", "qwen"])
    print(f"Coherent: {coherent}")

    # Save
    result = {
        "enable_thinking": enable_thinking,
        "prompt": formatted,
        "prompt_ids_len": len(prompt_ids),
        "output_ids_first5": token_ids[:5],
        "output_ids_len": len(token_ids),
        "decoded_text": text,
        "coherent": coherent,
    }

    with open(f"/tmp/vllm_tokenids_{enable_thinking}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    del llm
    import torch; torch.cuda.empty_cache()

print("\nDone!")
