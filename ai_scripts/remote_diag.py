"""Run on remote to verify training setup."""
import json, os, sys

ok = lambda s: print(f"  [OK] {s}")
err = lambda s: (print(f"  [FAIL] {s}"), sys.exit(1))

# Test 1: Data files
for fname in ["data/tau_bench/train_agent_66.jsonl", "data/tau_bench/val_agent_18.jsonl"]:
    with open(fname) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    sample = lines[0]
    assert "agent_name" in sample, f"{fname}: missing agent_name"
    assert isinstance(sample["prompt"], list), f"{fname}: prompt not list"
    ok(f"{fname}: {len(lines)} records")

# Test 2: agent_tools
import trainable_openclaw.training.agent_tools  # noqa: F401 — registers @function_tool wrappers
ok("agent_tools module loaded")

# Test 3: grpo_reward
from trainable_openclaw.training.grpo_reward import compute_score
ok("grpo_reward.compute_score")

# Test 4: verl
import verl
ok(f"verl={verl.__version__}")

# Test 5: Model
model_path = "/data/models/Qwen3.5-4B/Qwen/Qwen3.5-4B"
assert os.path.exists(os.path.join(model_path, "config.json")), "Model config not found"
ok(f"Model: {model_path}")

# Test 6: GPU
import torch
ok(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    ok(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")

print("\nAll checks passed!")
