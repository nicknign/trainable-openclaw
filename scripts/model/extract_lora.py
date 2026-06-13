"""Extract LoRA weights from FSDP checkpoint and save as PEFT adapter."""
import torch
import os, sys, json
from collections import OrderedDict

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/global_step_10/actor/model_world_size_1_rank_0.pt"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/global_step_10/actor/lora_adapter"

print(f"Loading: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# Analyze
lora_weights = OrderedDict()
all_keys = list(ckpt.keys())

for k in all_keys:
    if "lora" in k.lower():
        lora_weights[k] = ckpt[k]

print(f"Total keys: {len(all_keys)}")
print(f"LoRA keys: {len(lora_weights)}")

# Print sample keys
for k in list(lora_weights.keys())[:10]:
    print(f"  {k}: shape={lora_weights[k].shape}")

# Save LoRA weights separately
os.makedirs(out_dir, exist_ok=True)
torch.save(lora_weights, os.path.join(out_dir, "lora_weights.pt"))
print(f"\nLoRA weights saved to {out_dir}/lora_weights.pt")
print(f"Done. Extracted {len(lora_weights)} LoRA parameters.")
