#!/bin/bash
# ==============================================================================
# serve_ppo 启动脚本 — 推理服务 + 空闲检测训练触发
# 经过单卡 RTX 4090 24G + Qwen3-0.6B + vllm 0.12.0 验证通过
#
# 前置条件:
#   1. vllm 0.12.0 + PyTorch 2.9.0+cu128
#   2. 模型: /root/autodl-tmp/models/<model_name>
#   3. verl-main-0516 已安装: pip install -e verl-main-0516
#   4. flash_attn 非必须 (已改用 SDPA 默认值)
#
# 用法:
#   chmod +x scripts/run_serve.sh
#   bash scripts/run_serve.sh
#
# 查看日志:
#   tail -f /tmp/serve_ppo.log
#
# 测试:
#   curl http://localhost:8000/v1/health
#   curl http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"qwen3","messages":[{"role":"user","content":"hello"}],"max_tokens":50}'
# ==============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-0.6B}"
SERVE_PORT="${SERVE_PORT:-8000}"
LOG_FILE="${LOG_FILE:-/tmp/serve_ppo.log}"

cd "$(dirname "$0")/.."

echo "Starting serve_ppo ..."
echo "  Model:      $MODEL_PATH"
echo "  Port:       $SERVE_PORT"
echo "  Log:        $LOG_FILE"
echo ""

python3 -m verl.trainer.serve_ppo \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    +trainer.serve_port="$SERVE_PORT" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    2>&1 | tee "$LOG_FILE"
