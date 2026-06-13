#!/bin/bash
# Auto-generated start script
set -euo pipefail

# Ensure conda and its libs are in PATH (fix: libstdc++ and python3 resolution)
export PATH=/data/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/data/anaconda3/lib:${LD_LIBRARY_PATH:-}

MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3-4B}"
SERVE_PORT="${SERVE_PORT:-8000}"
LOG_FILE="${LOG_FILE:-/tmp/serve_ppo.log}"

cd /data/wangye/trainable-openclaw

echo "=== Starting serve_ppo ==="
echo "Model:      $MODEL_PATH"
echo "Port:       $SERVE_PORT"
echo "Log:        $LOG_FILE"
echo "Python:     $(which python3)"

nohup python3 -m verl.trainer.serve_ppo \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    +trainer.serve_port="$SERVE_PORT" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    > "$LOG_FILE" 2>&1 &
echo "PID: $!"
