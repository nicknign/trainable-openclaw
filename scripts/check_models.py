"""Check available Qwen3 models on ModelScope and VRAM status."""
from modelscope import list_models

print("=== Available Qwen3 Instruct models ===")
models = list(list_models("qwen/Qwen3-", limit=30))
for m in models:
    mid = m["Id"]
    if "Qwen3-" in mid and ("Instruct" in mid or "0.6B" in mid or "1.7B" in mid or "4B" in mid or "8B" in mid):
        print(f"  {mid}")

print("\n=== VRAM ===")
import subprocess
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv"],
    capture_output=True, text=True
)
print(result.stdout)
