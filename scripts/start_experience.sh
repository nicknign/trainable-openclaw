#!/bin/bash
# Quick-start: serve_ppo + nanobot for interactive experimentation
# Pure inference mode — no training, no external API needed
set -e

export LD_LIBRARY_PATH=/data/anaconda3/lib:$LD_LIBRARY_PATH
cd /data/wangye/trainable-openclaw

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3-4B}"
SERVE_PORT="${SERVE_PORT:-8000}"
NANOBOT_PORT="${NANOBOT_PORT:-18790}"
GPU_MEM="${GPU_MEM:-0.4}"          # 0.4 = leave room for FSDP overhead (HYBRID mode)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

echo "============================================"
echo "  nanobot Experience Mode"
echo "============================================"
echo "  Model:      $MODEL_PATH"
echo "  serve_ppo:  http://localhost:$SERVE_PORT/v1"
echo "  nanobot:    http://localhost:$NANOBOT_PORT"
echo "  GPU mem:    $GPU_MEM"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Kill old processes
# ---------------------------------------------------------------------------
echo "[1/4] Cleaning up old processes..."
pkill -f "serve_ppo" 2>/dev/null || true
pkill -f "nanobot gateway" 2>/dev/null || true
sleep 2
echo "  Done"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Start serve_ppo (pure inference)
# ---------------------------------------------------------------------------
echo "[2/4] Starting serve_ppo (pure inference, idle_timeout=999999)..."
nohup /data/anaconda3/bin/python -m verl.trainer.serve_ppo \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.rollout.enable_sleep_mode=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.logger='[console]' \
    +trainer.serve_port="$SERVE_PORT" \
    +trainer.idle_timeout=999999 \
    +trainer.min_samples=999999 \
    +trainer.max_train_rounds=0 \
    > /tmp/serve_ppo_experience.log 2>&1 &

SERVE_PID=$!
echo "  PID: $SERVE_PID (log: /tmp/serve_ppo_experience.log)"

# Wait for serve_ppo to be ready
echo -n "  Waiting for serve_ppo..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:$SERVE_PORT/v1/health" > /dev/null 2>&1; then
        echo " OK (${i}s)"
        break
    fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo ""
        echo "  ERROR: serve_ppo died. Check log:"
        tail -20 /tmp/serve_ppo_experience.log
        exit 1
    fi
    echo -n "."
    sleep 1
done
echo ""

# ---------------------------------------------------------------------------
# Step 3: Generate nanobot config
# ---------------------------------------------------------------------------
echo "[3/4] Generating nanobot config..."
CONFIG_DIR="$HOME/.nanobot"
CONFIG_FILE="$CONFIG_DIR/config.json"
mkdir -p "$CONFIG_DIR"
mkdir -p "$HOME/.nanobot/workspace"

cat > "$CONFIG_FILE" << EOFCFG
{
  "agents": {
    "defaults": {
      "workspace": "$HOME/.nanobot/workspace",
      "model": "qwen3-4b",
      "provider": "custom",
      "maxTokens": 4096,
      "contextWindowTokens": 8192,
      "temperature": 0.7,
      "maxToolIterations": 50,
      "maxConcurrentSubagents": 4,
      "timezone": "Asia/Shanghai",
      "botName": "trainable-claw",
      "sessionTtlMinutes": 60,
      "disabledSkills": [
        "image-generation",
        "long-goal",
        "cron"
      ]
    }
  },
  "providers": {
    "custom": {
      "apiBase": "http://localhost:${SERVE_PORT}/v1",
      "apiKey": "no-key"
    }
  },
  "gateway": {
    "host": "0.0.0.0",
    "port": ${NANOBOT_PORT}
  },
  "api": {
    "host": "127.0.0.1",
    "port": 8900
  },
  "tools": {
    "exec": {
      "sandbox": "none",
      "allowLocalhost": true
    }
  }
}
EOFCFG
echo "  Config: $CONFIG_FILE"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Start nanobot gateway
# ---------------------------------------------------------------------------
echo "[4/4] Starting nanobot gateway..."
nohup /data/anaconda3/bin/python -m nanobot gateway \
    --config "$CONFIG_FILE" \
    > /tmp/nanobot_gateway.log 2>&1 &

NANOBOT_PID=$!
echo "  PID: $NANOBOT_PID (log: /tmp/nanobot_gateway.log)"

sleep 3
if kill -0 "$NANOBOT_PID" 2>/dev/null; then
    echo ""
    echo "============================================"
    echo "  READY"
    echo "============================================"
    echo ""
    echo "  WebChat:   http://localhost:${NANOBOT_PORT}/webui/"
    echo "  API:       http://localhost:${NANOBOT_PORT}/api/v1/chat/completions"
    echo ""
    echo "  Logs:"
    echo "    serve_ppo:  tail -f /tmp/serve_ppo_experience.log"
    echo "    nanobot:    tail -f /tmp/nanobot_gateway.log"
    echo ""
    echo "  Stop:  pkill -f 'serve_ppo'; pkill -f 'nanobot gateway'"
    echo "============================================"
else
    echo "  ERROR: nanobot gateway failed to start"
    echo "  tail -20 /tmp/nanobot_gateway.log"
    tail -20 /tmp/nanobot_gateway.log
    exit 1
fi
