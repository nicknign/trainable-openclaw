#!/usr/bin/env bash
# ==============================================================================
# GRPO Training — Agent Loop Mode (REWARD_MODE=agent)
#
# verl Agent Loop handles multi-turn tool calling DURING rollout (vllm awake).
# Tools execute locally — zero API cost. No nanobot dependency.
#
# Usage:
#   bash scripts/train/run_grpo_agent.sh
#
# Prerequisites:
#   - Qwen3.5-4B model at /data/models/Qwen3.5-4B
#   - Preprocessed data: train_agent_66.jsonl, val_agent_18.jsonl
#     (run: python scripts/data/add_agent_name.py)
#   - verl installed: pip install -e verl-main-0516/
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
export REWARD_MODE="agent"

echo "============================================"
echo "  GRPO Training — Agent Loop Mode"
echo "============================================"
echo "  Project:      $PROJECT_DIR"
echo "  REWARD_MODE:  $REWARD_MODE"
echo "  GPU:          $CUDA_VISIBLE_DEVICES"
echo "  Model:        /data/models/Qwen3.5-4B"
echo "  Config:       scripts/train/grpo_retail.yaml"
echo "  Data:         data/tau_bench/train_agent_66.jsonl"
echo "  Tools:        trainable_openclaw/training/agent_tools.py"
echo "============================================"
echo ""

# ---- verify preconditions -------------------------------------------------
PYTHON="${PYTHON:-python}"

$PYTHON -c "import trainable_openclaw.training.grpo_reward" \
    || { echo "ERROR: trainable_openclaw not installed. Run: bash scripts/deploy/setup_env.sh"; exit 1; }

for f in data/tau_bench/train_agent_66.jsonl data/tau_bench/val_agent_18.jsonl; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found"
        echo "  Run: python scripts/data/add_agent_name.py data/tau_bench/train_split_66.jsonl data/tau_bench/train_agent_66.jsonl"
        echo "  Run: python scripts/data/add_agent_name.py data/tau_bench/val_split_18.jsonl data/tau_bench/val_agent_18.jsonl"
        exit 1
    fi
    echo "  OK: $f ($(wc -l < "$f") tasks)"
done

for f in trainable_openclaw/training/agent_tools.py; do
    [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
    echo "  OK: $f"
done

if [ ! -d /data/models/Qwen3.5-4B ]; then
    echo "ERROR: Qwen3.5-4B not found at /data/models/Qwen3.5-4B"
    exit 1
fi
echo "  OK: Qwen3.5-4B model"

echo ""
echo "Launching GRPO training (Agent Loop)..."
echo "  Log: /tmp/grpo_agent.log"
echo ""

# ---- launch ---------------------------------------------------------------
$PYTHON -m verl.trainer.main_ppo \
    --config-path scripts/train \
    --config-name grpo_retail \
    2>&1 | tee /tmp/grpo_agent.log
