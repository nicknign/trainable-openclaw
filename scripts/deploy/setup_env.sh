#!/usr/bin/env bash
# ==============================================================================
# trainable-openclaw 环境准备脚本
#
# 用法:
#   bash scripts/deploy/setup_env.sh
#
# 自定义:
#   CONDA_PYTHON=/data/anaconda3/bin/python  # conda Python 路径
#   PROJECT_DIR=/data/wangye/trainable-openclaw  # 项目根目录
#   SKIP_MODEL_CHECK=1  # 跳过模型文件检查
# ==============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# ---- config ---------------------------------------------------------------
PYTHON="${PYTHON:-python3}"
CONDA_PYTHON="${CONDA_PYTHON:-/data/anaconda3/bin/python}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
VERL_DIR="${VERL_DIR:-$PROJECT_DIR/verl-main-0516}"
NANOBOT_DIR="${NANOBOT_DIR:-$PROJECT_DIR/nanobot-0.2.1}"
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3-4B}"

echo "============================================"
echo "  trainable-openclaw Environment Setup"
echo "============================================"
echo "  Project:  $PROJECT_DIR"
echo "  Python:   $PYTHON"
echo "  verl:     $VERL_DIR"
echo "  nanobot:  $NANOBOT_DIR"
echo "  Model:    $MODEL_PATH"
echo ""

# ---- fix libstdc++ for conda ----------------------------------------------
if [ -d /data/anaconda3/lib ]; then
    export LD_LIBRARY_PATH=/data/anaconda3/lib:${LD_LIBRARY_PATH:-}
    ok "LD_LIBRARY_PATH set (conda libs)"
fi

# ---- 1. Python ------------------------------------------------------------
echo "--- [1/6] Python ---"
PY_VER=$($PYTHON --version 2>&1) || fail "python3 not found"
ok "$PY_VER"

# ---- 2. pip packages ------------------------------------------------------
echo "--- [2/6] pip packages ---"
MISSING=""
for pkg in torch vllm ray requests transformers; do
    if $PYTHON -c "import $pkg" 2>/dev/null; then
        ok "$pkg"
    else
        warn "$pkg MISSING"
        MISSING="$MISSING $pkg"
    fi
done

if [ -n "$MISSING" ]; then
    echo ""
    echo "Missing packages:$MISSING"
    echo "Install with: pip install$MISSING"
    echo ""
    read -p "Install now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $PYTHON -m pip install $MISSING
    fi
fi

# ---- 3. verl --------------------------------------------------------------
echo "--- [3/6] verl ---"
if [ -d "$VERL_DIR" ]; then
    if $PYTHON -c "import verl" 2>/dev/null; then
        VERL_VER=$($PYTHON -c "import verl; print(getattr(verl, '__version__', 'unknown'))" 2>/dev/null || echo "ok")
        ok "verl imported ($VERL_VER)"
    else
        warn "verl not installed, installing from $VERL_DIR..."
        cd "$VERL_DIR" && $PYTHON -m pip install -e . -q
        $PYTHON -c "import verl" && ok "verl installed" || fail "verl install failed"
    fi
else
    warn "verl dir not found: $VERL_DIR (set VERL_DIR=...)"
fi

# ---- 4. trainable_openclaw ------------------------------------------------
echo "--- [4/6] trainable_openclaw ---"
cd "$PROJECT_DIR"
$PYTHON -m pip install -e . -q
$PYTHON -c "import trainable_openclaw" && ok "trainable_openclaw imported" || fail "trainable_openclaw import failed"

# verify key training modules
for mod in training.grpo_reward training.rollout_env training.rubric_rules; do
    $PYTHON -c "import trainable_openclaw.$mod" 2>/dev/null \
        && ok "  $mod" \
        || warn "  $mod MISSING"
done

# ---- 5. nanobot -----------------------------------------------------------
echo "--- [5/6] nanobot ---"
if [ -d "$NANOBOT_DIR" ]; then
    ok "nanobot source: $NANOBOT_DIR"
    if PYTHONPATH="$NANOBOT_DIR:$PYTHONPATH" $PYTHON -c "import nanobot" 2>/dev/null; then
        ok "nanobot importable"
    else
        warn "nanobot import failed (check nanobot source)"
    fi
else
    warn "nanobot not found at $NANOBOT_DIR (set NANOBOT_DIR=...)"
    warn "Training in REWARD_MODE=direct doesn't need nanobot"
fi

# ---- 6. GPU + model -------------------------------------------------------
echo "--- [6/6] GPU + model ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | while read -r line; do
        ok "GPU: $line"
    done
else
    warn "nvidia-smi not found"
fi

if [ "${SKIP_MODEL_CHECK:-0}" = "1" ]; then
    warn "model check skipped (SKIP_MODEL_CHECK=1)"
else
    if [ -d "$MODEL_PATH" ]; then
        if [ -f "$MODEL_PATH/config.json" ]; then
            ok "Model: $MODEL_PATH"
        else
            warn "Model dir exists but no config.json: $MODEL_PATH"
        fi
    else
        warn "Model not found: $MODEL_PATH"
        warn "Download: python scripts/model/download_model.py"
    fi
fi

# ---- data ----------------------------------------------------------------
echo ""
echo "--- Data files ---"
DATA_DIR="$PROJECT_DIR/data/tau_bench"
REQUIRED_FILES=(
    "train_split_66.jsonl"
    "val_split_18.jsonl"
)
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$DATA_DIR/$f" ]; then
        ok "$f ($(wc -l < "$DATA_DIR/$f") lines)"
    else
        warn "$f MISSING — run: python scripts/data/filter_split_tau_bench.py"
    fi
done

echo ""
echo "============================================"
echo "  Setup complete"
echo "============================================"
echo ""
echo "  Next: start training"
echo "    Direct mode:  bash scripts/train/run_grpo_direct.sh"
echo "    Nanobot mode: bash scripts/train/run_grpo_nanobot.sh"
echo ""
