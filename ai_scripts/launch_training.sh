#!/bin/bash
# Launch GRPO training on remote machine
set -euo pipefail

export LD_LIBRARY_PATH="/data/anaconda3/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/data/wangye/trainable-openclaw:/data/wangye/trainable-openclaw/verl-main-0516:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export REWARD_MODE=agent

cd /data/wangye/trainable-openclaw

echo "=== Starting GRPO training $(date) ==="
echo "Config: grpo_retail.yaml | Model: Qwen3.5-4B | GPU: RTX 4090 48GB"
echo "Log: /tmp/grpo_agent.log"

nohup /data/anaconda3/bin/python -m verl.trainer.main_ppo \
    --config-name grpo_retail \
    > /tmp/grpo_agent.log 2>&1 &

PID=$!
echo "PID=$PID"
echo "=== Launched ==="
