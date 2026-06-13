"""Download Qwen3 instruct model from ModelScope."""
import sys
from modelscope import snapshot_download

model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B-Instruct"
print(f"Trying: {model_id}")
try:
    path = snapshot_download(model_id)
    print(f"Downloaded to: {path}")
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)
