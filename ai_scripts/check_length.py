"""Analyze required sequence lengths and GPU memory for GRPO training."""
# Model: Qwen3.5-4B, 4.57B params
model_gb = 4.57 * 2  # FP16

# Qwen3.5-4B: ~36 layers, GQA with ~8 kv_heads, head_dim ~128
kv_per_1k = 2 * 36 * 8 * 128 * 2 / (1024**3)  # GB per 1000 tokens

print("=" * 60)
print("GPU Memory Analysis — RTX 4090 48GB, Qwen3.5-4B LoRA rank=64")
print("=" * 60)

print("\n--- vLLM only (rollout phase) ---")
print(f"Model weights: {model_gb:.1f} GB")
for max_len in [6144, 8192, 12288, 16384, 24576]:
    kv_gb = kv_per_1k * max_len / 1000
    total = model_gb + kv_gb + 2.0
    fits = "YES" if total < 48 else "NO"
    print(f"  max_model_len={max_len:5d}: KV={kv_gb:.1f}GB, total~{total:.1f}GB, fits={fits}")

print("\n--- vLLM + Actor training (same GPU, LoRA) ---")
print("Actor: model + LoRA weights + optimizer states + activations")
for max_len in [6144, 8192, 12288, 16384]:
    # Actor: ~1.5x model (LoRA + optimizer + activations)
    actor_gb = model_gb * 1.3
    vllm_gb = model_gb * 0.9 + kv_per_1k * max_len / 1000  # vLLM slightly less (no optimizer)
    total = actor_gb + vllm_gb + 1.0
    fits = "YES" if total < 46 else "TIGHT" if total < 48 else "NO"
    print(f"  max_model_len={max_len:5d}: vllm~{vllm_gb:.1f}GB + actor~{actor_gb:.1f}GB = {total:.1f}GB, fits={fits}")

print("\n--- Sequence Length Requirements ---")
print("Error shows multi-turn responses: 10719-10809 tokens")
print(f"With prompt_length=1024: total sequence = 11743-11833")
print(f"With 2 assistant turns + tool responses: up to ~12000 tokens")
print(f"\nRecommended config:")
print(f"  max_model_len: 16384 (safe) or 12288 (tight)")
print(f"  response_length: 12288")
print(f"  prompt_length: 1024 (actual max ~284, but safe)")
print(f"  gpu_memory_utilization: 0.85")
print(f"\nHardware: 1x RTX 4090 48GB is SUFFICIENT")
print(f"No additional GPUs needed.")
