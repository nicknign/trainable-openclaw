#!/usr/bin/env bash
# ==============================================================================
# GRPO Training — vllm Direct Mode (REWARD_MODE=direct)
#
# verl rollout → compute_score scores output directly (single-turn rubric).
# No nanobot dependency. Works on any hardware.
#
# Usage:
#   bash scripts/train/run_grpo_direct.sh
#
# Config: scripts/train/grpo_retail.yaml
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

# ---- env ------------------------------------------------------------------
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}/data/anaconda3/lib:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/verl-main-0516:${PYTHONPATH}"
export REWARD_MODE="direct"
export NANOBOT_URL="${NANOBOT_URL:-http://localhost:8900}"

echo "============================================"
echo "  GRPO Training — vllm Direct Mode"
echo "============================================"
echo "  Project:     $PROJECT_DIR"
echo "  REWARD_MODE: $REWARD_MODE"
echo "  GPU:         $CUDA_VISIBLE_DEVICES"
echo "  Config:      scripts/train/grpo_retail.yaml"
echo "  Data:        data/tau_bench/train_split_66.jsonl"
echo "============================================"
echo ""

# ---- verify preconditions -------------------------------------------------
$PYTHON -c "import trainable_openclaw.training.grpo_reward" \
    || { echo "ERROR: trainable_openclaw not installed. Run: bash scripts/deploy/setup_env.sh"; exit 1; }

for f in data/tau_bench/train_split_66.jsonl data/tau_bench/val_split_18.jsonl; do
    [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
done

echo "Launching GRPO training..."
echo "  Log: /tmp/grpo_direct.log"
echo ""

# ---- launch ---------------------------------------------------------------
python -m verl.trainer.main_ppo \
    --config-path scripts/train \
    --config-name grpo_retail \
    2>&1 | tee /tmp/grpo_direct.log
