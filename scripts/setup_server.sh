#!/bin/bash
# ==============================================================================
# Server-side setup: vllm + Qwen3-0.6B + inference test
# Usage: bash setup_server.sh
# Log: written to /tmp/setup_<time>.log
# ==============================================================================
set -euo pipefail
LOG=/tmp/setup_$(date +%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "========================================="
echo "  Setup started: $(date)"
echo "  Log file: $LOG"
echo "========================================="

export TMPDIR=/root/autodl-tmp/tmp
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export HF_HOME=/root/autodl-tmp/models
export HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache

# ----------------------------------------------------------
# Step 1: Install vllm 0.12.0
# ----------------------------------------------------------
echo ""
echo "=== [1/3] Installing vllm 0.12.0 (compatible with CUDA 12.x) ==="
pip3 install vllm==0.12.0 2>&1 | tail -5

python3 -c "
import vllm
print('vLLM {} installed OK'.format(vllm.__version__))
"

# ----------------------------------------------------------
# Step 2: Download Qwen3-0.6B
# ----------------------------------------------------------
echo ""
echo "=== [2/3] Downloading Qwen3-0.6B from modelscope ==="

python3 -c "
from modelscope import snapshot_download
path = snapshot_download('Qwen/Qwen3-0.6B')
print('Downloaded to:', path)
"

# Move from modelscope cache to data disk
CACHE_DIR="$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
if [ -d "$CACHE_DIR" ]; then
    mkdir -p /root/autodl-tmp/models/Qwen3-0.6B
    cp -r "$CACHE_DIR"/* /root/autodl-tmp/models/Qwen3-0.6B/
    rm -rf "$CACHE_DIR"
    echo "Moved model to /root/autodl-tmp/models/Qwen3-0.6B"
fi

echo "Model disk usage:"
du -sh /root/autodl-tmp/models/Qwen3-0.6B 2>/dev/null || echo "(checking...)"

# ----------------------------------------------------------
# Step 3: Test vllm inference
# ----------------------------------------------------------
echo ""
echo "=== [3/3] Testing vllm + Qwen3-0.6B inference ==="

python3 -c "
from vllm import LLM, SamplingParams

print('Loading Qwen3-0.6B ...')
llm = LLM(
    model='/root/autodl-tmp/models/Qwen3-0.6B',
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)

print('Generating test output ...')
outputs = llm.generate(
    ['Hello, my name is'],
    SamplingParams(temperature=0.7, max_tokens=100)
)
for o in outputs:
    print('Prompt:', o.prompt)
    print('Output:', o.outputs[0].text)
print()
print('=== SUCCESS: Qwen3-0.6B works with vllm 0.12.0 ===')
"

echo ""
echo "========================================="
echo "  Setup complete: $(date)"
echo ""
echo "  Disk usage:"
df -h / /root/autodl-tmp 2>/dev/null
echo ""
echo "  GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv 2>/dev/null || true
echo "========================================="
