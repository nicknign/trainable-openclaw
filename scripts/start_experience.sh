#!/bin/bash
# Quick-start: serve_ppo + nanobot (API server + gateway) for experimentation
# Pure inference mode — no training, no external API needed
set -e

export LD_LIBRARY_PATH=/data/anaconda3/lib:$LD_LIBRARY_PATH
cd /data/wangye/trainable-openclaw

# -- Config ---------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3-4B}"
SERVE_PORT="${SERVE_PORT:-8000}"
NANOBOT_API_PORT="${NANOBOT_API_PORT:-8900}"
NANOBOT_GW_PORT="${NANOBOT_GW_PORT:-18790}"
GPU_MEM="${GPU_MEM:-0.4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
PYTHON=/data/anaconda3/bin/python
NANOBOT_SRC=/data/wangye/trainable-openclaw/nanobot-0.2.1

echo "============================================"
echo "  nanobot Experience Mode"
echo "============================================"
echo "  Model:        $MODEL_PATH"
echo "  serve_ppo:    http://localhost:$SERVE_PORT/v1"
echo "  nanobot API:  http://localhost:$NANOBOT_API_PORT/v1/chat/completions"
echo "  nanobot GW:   http://localhost:$NANOBOT_GW_PORT"
echo "============================================"
echo ""

# -- Step 1: Kill old processes -------------------------------------------
echo "[1/5] Cleaning up old processes..."
pkill -f "serve_ppo" 2>/dev/null || true
pkill -f "nanobot gateway" 2>/dev/null || true
pkill -f "nanobot serve" 2>/dev/null || true
sleep 2
echo "  Done"
echo ""

# -- Step 2: Install deps if needed ---------------------------------------
echo "[2/5] Checking dependencies..."
$PYTHON -c "import aiohttp" 2>/dev/null || $PYTHON -m pip install aiohttp -q
$PYTHON -c "import dulwich" 2>/dev/null || $PYTHON -m pip install dulwich -q
echo "  Done"
echo ""

# -- Step 3: Start serve_ppo (pure inference) ------------------------------
echo "[3/5] Starting serve_ppo (pure inference)..."
nohup $PYTHON -m verl.trainer.serve_ppo \
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
echo "  PID: $SERVE_PID"

echo -n "  Waiting for serve_ppo..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:$SERVE_PORT/v1/health" > /dev/null 2>&1; then
        echo " OK (${i}s)"
        break
    fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo " ERROR: died"
        tail -20 /tmp/serve_ppo_experience.log
        exit 1
    fi
    echo -n "."
    sleep 1
done
echo ""

# -- Step 4: Generate nanobot config --------------------------------------
echo "[4/5] Generating nanobot config..."
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
      "contextWindowTokens": 32768,
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
    "host": "127.0.0.1",
    "port": ${NANOBOT_GW_PORT}
  },
  "api": {
    "host": "127.0.0.1",
    "port": ${NANOBOT_API_PORT}
  },
  "tools": {
    "exec": {
      "sandbox": "none",
      "allowLocalhost": true
    }
  },
  "channels": {
    "websocket": {
      "enabled": true,
      "host": "127.0.0.1",
      "port": 18791,
      "path": "/",
      "allow_from": ["*"],
      "streaming": true
    }
  }
}
EOFCFG
echo "  Config: $CONFIG_FILE"
echo ""

# -- Step 5: Start nanobot services ---------------------------------------
echo "[5/5] Starting nanobot..."

# API server (OpenAI-compatible, port 8900)
PYTHONPATH="$NANOBOT_SRC:$PYTHONPATH" nohup $PYTHON -m nanobot serve \
    --config "$CONFIG_FILE" \
    --host 127.0.0.1 --port $NANOBOT_API_PORT \
    > /tmp/nanobot_api.log 2>&1 &
API_PID=$!
echo "  API PID: $API_PID (log: /tmp/nanobot_api.log)"

# Gateway (health + websocket, port 18790)
PYTHONPATH="$NANOBOT_SRC:$PYTHONPATH" nohup $PYTHON -m nanobot gateway \
    --config "$CONFIG_FILE" \
    --port $NANOBOT_GW_PORT \
    > /tmp/nanobot_gateway.log 2>&1 &
GW_PID=$!
echo "  GW PID:  $GW_PID (log: /tmp/nanobot_gateway.log)"

sleep 4

# Verify
API_OK=0
GW_OK=0
curl -sf http://localhost:$NANOBOT_API_PORT/health > /dev/null 2>&1 && API_OK=1
curl -sf http://localhost:$NANOBOT_GW_PORT/health > /dev/null 2>&1 && GW_OK=1

echo ""
echo "============================================"
echo "  READY"
echo "============================================"
echo ""
echo "  serve_ppo:  http://localhost:${SERVE_PORT}/v1  [$( [[ $SERVE_PID ]] && echo OK || echo FAIL )]"
echo "  nanobot API: http://localhost:${NANOBOT_API_PORT}   [$( [[ $API_OK -eq 1 ]] && echo OK || echo FAIL )]"
echo "  nanobot GW:  http://localhost:${NANOBOT_GW_PORT}  [$( [[ $GW_OK -eq 1 ]] && echo OK || echo FAIL )]"
echo ""
echo "  Test API:"
echo "    curl http://localhost:${NANOBOT_API_PORT}/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\":\"qwen3-4b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":100}'"
echo ""
echo "  CLI chat (interactive):"
echo "    cd /data/wangye/trainable-openclaw"
echo "    PYTHONPATH=$NANOBOT_SRC:\$PYTHONPATH $PYTHON -m nanobot agent --config $CONFIG_FILE"
echo ""
echo "  Logs:"
echo "    serve_ppo:  tail -f /tmp/serve_ppo_experience.log"
echo "    nanobot API: tail -f /tmp/nanobot_api.log"
echo "    nanobot GW:  tail -f /tmp/nanobot_gateway.log"
echo ""
echo "  Stop:  pkill -f 'serve_ppo'; pkill -f 'nanobot'"
echo "============================================"
