#!/usr/bin/env bash
# ==============================================================================
# GRPO Training — Nanobot Mode (REWARD_MODE=nanobot)
#
# verl rollout → compute_score calls nanobot for multi-turn agent loop.
# Requires: nanobot + vllm running on a SEPARATE GPU (not the training GPU).
#
# On single GPU: nanobot will be unreachable during training (verl sleeps vllm),
#   so reward falls back to single-turn scoring — same as direct mode.
# On multi-GPU: nanobot's vllm stays awake on a separate GPU → full multi-turn.
#
# Usage:
#   # 1. Start nanobot services (if not already running)
#   bash scripts/deploy/start_experience.sh
#
#   # 2. Launch training
#   bash scripts/train/run_grpo_nanobot.sh
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
export REWARD_MODE="nanobot"
export NANOBOT_URL="${NANOBOT_URL:-http://localhost:8900}"
export NANOBOT_TIMEOUT="${NANOBOT_TIMEOUT:-180}"

echo "============================================"
echo "  GRPO Training — Nanobot Mode"
echo "============================================"
echo "  Project:      $PROJECT_DIR"
echo "  REWARD_MODE:  $REWARD_MODE"
echo "  GPU:          $CUDA_VISIBLE_DEVICES"
echo "  NANOBOT_URL:  $NANOBOT_URL"
echo "  Config:       scripts/train/grpo_retail.yaml"
echo "  Data:         data/tau_bench/train_split_66.jsonl"
echo "============================================"
echo ""

# ---- verify preconditions -------------------------------------------------
for f in data/tau_bench/train_split_66.jsonl data/tau_bench/val_split_18.jsonl; do
    [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
done

$PYTHON -c "import trainable_openclaw.training.grpo_reward" \
    || { echo "ERROR: trainable_openclaw not installed. Run: bash scripts/deploy/setup_env.sh"; exit 1; }

# ---- check nanobot --------------------------------------------------------
echo -n "Checking nanobot at $NANOBOT_URL ... "
if curl -sf "${NANOBOT_URL}/health" > /dev/null 2>&1; then
    echo "OK"
elif curl -sf "${NANOBOT_URL}/v1/chat/completions" -m 5 > /dev/null 2>&1; then
    echo "OK (API)"
else
    echo "UNREACHABLE"
    echo ""
    echo "WARNING: nanobot is not running. Training will proceed but rewards"
    echo "will fall back to single-turn scoring (same as direct mode)."
    echo ""
    echo "To start nanobot for multi-turn rewards:"
    echo "  bash scripts/deploy/start_experience.sh"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

echo ""
echo "Launching GRPO training..."
echo "  Log: /tmp/grpo_nanobot.log"
echo ""

# ---- launch ---------------------------------------------------------------
python -m verl.trainer.main_ppo \
    --config-path scripts/train \
    --config-name grpo_retail \
    2>&1 | tee /tmp/grpo_nanobot.log
