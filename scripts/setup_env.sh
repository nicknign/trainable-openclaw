#!/usr/bin/env bash
# ==============================================================================
# trainable-openclaw 3090 Linux 环境安装脚本
#
# 用法:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
# 自定义配置（环境变量覆盖）:
#   CUDA_VER=12.6  VENV_DIR=./venv  ./setup_env.sh
# ==============================================================================
set -euo pipefail

# ── 配置 ─────────────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VER="${CUDA_VER:-12.1}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/verl_train}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERL_DIR="$PROJECT_ROOT/verl-main-0516"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERR]${NC}   $*"; exit 1; }

# ── Step 0: 系统检查 ────────────────────────────────────────────────────────
log "Step 0: 检查系统环境..."

[[ "$(uname -s)" == "Linux" ]] || err "此脚本仅支持 Linux 环境"

# 检查 Python
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    PYTHON_BIN="python"
fi
PY_VER=$("$PYTHON_BIN" --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
log "系统 Python: $PY_VER"
if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 10 ]]; }; then
    err "需要 Python >= 3.10，当前: $PY_VER"
fi

# 检查 GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unknown")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "0")
    log "GPU: $GPU_NAME, 显存: ${GPU_MEM}MB"
    if [[ "$GPU_MEM" -lt 20000 ]]; then
        warn "显存 < 20GB，仅推荐 0.5B~1.5B 模型 + LoRA"
    fi
else
    err "未检测到 nvidia-smi，请先安装 NVIDIA 驱动"
fi

# ── Step 1: 创建 venv ────────────────────────────────────────────────────────
log "Step 1: 创建 Python venv: $VENV_DIR..."

mkdir -p "$(dirname "$VENV_DIR")"
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

# ── Step 2: 安装 PyTorch ────────────────────────────────────────────────────
log "Step 2: 安装 PyTorch (CUDA $CUDA_VER)..."

CUDA_TAG="cu${CUDA_VER//./}"
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.device_count()}')"

# ── Step 3: 安装 veRL + vllm ────────────────────────────────────────────────
log "Step 3: 安装 veRL (verl-main-0516) + vllm..."

pip install -e "$VERL_DIR"
pip install vllm

# ── Step 4: flash-attn（可选）────────────────────────────────────────────────
log "Step 4: flash-attn（编译需要几分钟，失败自动跳过）..."

pip install ninja packaging
if pip install flash-attn --no-build-isolation 2>/dev/null; then
    python -c "import flash_attn; print(f'flash-attn {flash_attn.__version__} OK')"
    log "flash-attn 安装成功"
else
    warn "flash-attn 编译失败，将使用 PyTorch SDPA（功能完整，性能略低）"
fi

# ── Step 5: 其他优化 ────────────────────────────────────────────────────────
log "Step 5: liger-kernel..."

pip install liger-kernel

# ── Step 6: 项目依赖 ────────────────────────────────────────────────────────
log "Step 6: 安装项目依赖..."

if [[ -s "$PROJECT_ROOT/requirements.txt" ]]; then
    pip install -r "$PROJECT_ROOT/requirements.txt"
fi

# ── Step 7: 验证 ────────────────────────────────────────────────────────────
log "Step 7: 验证安装..."

python -c "
results = []
try:
    import torch
    results.append(f'PyTorch {torch.__version__} | CUDA: {\"OK\" if torch.cuda.is_available() else \"MISSING\"} | GPU: {torch.cuda.device_count()}')
except Exception as e:
    results.append(f'PyTorch: FAILED ({e})')

try:
    import vllm; results.append(f'vLLM {vllm.__version__} OK')
except Exception as e:
    results.append(f'vLLM: FAILED ({e})')

try:
    import verl; results.append(f'veRL {verl.__version__} OK')
except Exception as e:
    results.append(f'veRL: FAILED ({e})')

try:
    import flash_attn; results.append(f'flash-attn {flash_attn.__version__} OK')
except Exception:
    results.append(f'flash-attn: skipped (SDPA fallback)')

try:
    import transformers, peft
    results.append(f'transformers {transformers.__version__} | peft {peft.__version__} OK')
except Exception as e:
    results.append(f'transformers/peft: FAILED ({e})')

try:
    import fastapi; results.append(f'FastAPI {fastapi.__version__} OK')
except Exception as e:
    results.append(f'FastAPI: FAILED ({e})')

print()
print('==========================================')
print('  安装完成')
print('==========================================')
for r in results:
    print(f'  {r}')
"

echo ""
echo "  激活环境:  source $VENV_DIR/bin/activate"
echo ""
echo "  下一步:"
echo "    1. 下载模型: huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct"
echo "    2. 启动服务: cd $PROJECT_ROOT && python -m trainable_openclaw.server.app"
echo ""
