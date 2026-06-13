"""Convert extracted FSDP LoRA weights to PEFT adapter format for vLLM loading."""
import torch, os, json, sys

lora_pt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/global_step_10/actor/lora_adapter/lora_weights.pt"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/global_step_10/actor/lora_adapter"

print(f"Loading: {lora_pt}")
lora_weights = torch.load(lora_pt, map_location="cpu", weights_only=False)

# Remove base_model.model. prefix if present, PEFT/vLLM may or may not need it
# vLLM expects: model.layers.X.self_attn.q_proj.lora_A.weight (no base_model.model prefix)
# Let's keep both formats - save as-is for PEFT, and also check what vLLM expects
peft_weights = {}
for k, v in lora_weights.items():
    # Keep as-is: PEFT format has base_model.model. prefix
    peft_weights[k] = v

# Create adapter_config.json
# Infer settings from the weights
sample_keys = list(lora_weights.keys())
# Rank from lora_A shape: base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
r = None
alpha = None
for k in sample_keys:
    if "lora_A" in k:
        r = lora_weights[k].shape[0]
        break

target_modules = set()
for k in sample_keys:
    # Extract module name: self_attn.q_proj, mlp.gate_proj, etc.
    parts = k.split(".")
    for i, p in enumerate(parts):
        if p in ("self_attn", "mlp"):
            target_modules.add(p + "." + parts[i+1])
            break

adapter_config = {
    "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",  # approximate
    "peft_type": "LORA",
    "r": r or 16,
    "lora_alpha": alpha or 32,  # from training config
    "lora_dropout": 0.0,
    "target_modules": sorted(target_modules),
    "task_type": "CAUSAL_LM",
}

os.makedirs(out_dir, exist_ok=True)

# Save adapter in safetensors format for vLLM
try:
    from safetensors.torch import save_file
    # Convert keys for vLLM: remove "base_model.model." prefix and ".default" suffix
    vllm_weights = {}
    for k, v in peft_weights.items():
        new_k = k.replace("base_model.model.", "").replace(".default", "")
        vllm_weights[new_k] = v
    save_file(vllm_weights, os.path.join(out_dir, "adapter_model.safetensors"))
    print(f"Saved safetensors: {len(vllm_weights)} keys")
except ImportError:
    torch.save(peft_weights, os.path.join(out_dir, "adapter_model.bin"))
    print(f"Saved bin (safetensors not available): {len(peft_weights)} keys")

with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
    json.dump(adapter_config, f, indent=2, ensure_ascii=False)

print(f"adapter_config.json: r={r}, alpha={adapter_config['lora_alpha']}")
print(f"target_modules: {sorted(target_modules)}")
print(f"Done. Adapter saved to {out_dir}/")
