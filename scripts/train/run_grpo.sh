#!/bin/bash
# Launch GRPO training for tau-bench retail on single RTX 4090
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/verl-main-0516:${PYTHONPATH}"
export TAU_BENCH_TRAIN_PROMPTS="${PROJECT_DIR}/data/tau_bench/train_prompts_augmented.jsonl"

echo "============================================"
echo "GRPO Training: Qwen3.5-4B + LoRA on tau-bench retail"
echo "Project dir:  ${PROJECT_DIR}"
echo "Config:       scripts/train/grpo_retail.yaml"
echo "Data:         data/tau_bench/train_prompts_augmented.jsonl"
echo "============================================"

python -m verl.trainer.main_ppo \
    --config-path scripts/train \
    --config-name grpo_retail
