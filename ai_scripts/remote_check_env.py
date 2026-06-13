"""Upload to remote and run: checks vllm, torch, nanobot, openai, project imports."""
import sys

print("=== vllm ===")
try:
    import vllm
    print(f"vllm version: {vllm.__version__}")
except Exception as e:
    print(f"vllm ERROR: {e}")

print()
print("=== torch ===")
try:
    import torch
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"cuda memory: {mem_gb:.1f} GB")
except Exception as e:
    print(f"torch ERROR: {e}")

print()
print("=== nanobot (PYTHONPATH) ===")
try:
    sys.path.insert(0, "/data/wangye/trainable-openclaw/nanobot-0.2.1")
    import nanobot
    ver = getattr(nanobot, "__version__", "unknown")
    print(f"nanobot OK, version: {ver}")
except Exception as e:
    print(f"nanobot ERROR: {e}")

print()
print("=== openai ===")
try:
    import openai
    print(f"openai version: {openai.__version__}")
except Exception as e:
    print(f"openai ERROR: {e}")

print()
print("=== trainable_openclaw ===")
try:
    sys.path.insert(0, "/data/wangye/trainable-openclaw")
    from trainable_openclaw.evaluation import SimulatedUser, AgentRunner
    print("trainable_openclaw OK")
except Exception as e:
    print(f"trainable_openclaw ERROR: {e}")

print()
print("=== Qwen3-4B model ===")
import os
model_path = "/data/models/Qwen3-4B"
if os.path.isdir(model_path):
    files = os.listdir(model_path)
    print(f"Model dir exists, {len(files)} files")
    for f in sorted(files)[:8]:
        size_mb = os.path.getsize(os.path.join(model_path, f)) / 1e6
        print(f"  {f}: {size_mb:.0f} MB")
else:
    print("Model NOT FOUND at", model_path)

print()
print("All checks complete.")
