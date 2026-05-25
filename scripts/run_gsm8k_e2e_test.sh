#!/usr/bin/env bash
# ============================================================================
# End-to-end GSM8K GRPO training test — 20+ steps, batch size 64
#
# Usage (on remote Linux GPU server):
#   bash scripts/run_gsm8k_e2e_test.sh
#
# Config (override via env vars):
#   STEPS=20          Training steps per cycle
#   PROMPTS_PER=16    Prompts per step (× rollout.n=4 → batch=64)
#   GSM8K_N=320       GSM8K prompts loaded (STEPS × PROMPTS_PER)
#   GPU_MEM=0.35      vLLM GPU memory fraction
#   RESP_LEN=4096     Max generation tokens
#   IDLE_TO=10        Idle timeout before training triggers
#   LR=3e-6           Learning rate
# ============================================================================
set -euo pipefail

# ---- Defaults (override via env) ----
STEPS=${STEPS:-20}
PROMPTS_PER=${PROMPTS_PER:-16}
GSM8K_N=${GSM8K_N:-320}
GPU_MEM=${GPU_MEM:-0.35}
RESP_LEN=${RESP_LEN:-4096}
IDLE_TO=${IDLE_TO:-10}
LR=${LR:-3e-6}
ROLLOUT_N=4
MINI_BATCH=16
PORT=8000
LOG_DIR=/tmp
SERVER_LOG="${LOG_DIR}/serve_ppo_gsm8k.log"
TRAIN_LOG="${LOG_DIR}/serve_ppo_train.log"

# ---- Environment ----
export PATH=/data/anaconda3/bin:/usr/bin:/bin:$PATH
export LD_LIBRARY_PATH=/data/anaconda3/lib:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "============================================================================"
echo "  GSM8K GRPO Training — ${STEPS} steps, batch=$((PROMPTS_PER * ROLLOUT_N))"
echo "============================================================================"

# ---- Step 1: Cleanup ----
echo "=== [1/4] Cleaning up previous processes ==="

pkill -f "serve_ppo" 2>/dev/null || true
sleep 2
ray stop 2>/dev/null || true
sleep 3

GPU_PIDS=$(nvidia-smi 2>/dev/null | grep -oP '\d+(?=\s+C\s)' || true)
for pid in $GPU_PIDS; do kill -9 "$pid" 2>/dev/null || true; done
sleep 2

GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | grep -oP '\d+' | head -1 || echo "0")
echo "  GPU memory used: ${GPU_USED} MiB"

:> "$TRAIN_LOG" 2>/dev/null || true

# ---- Step 2: Start server ----
echo "=== [2/4] Starting serve_ppo (${STEPS} steps, ${GSM8K_N} prompts) ==="

/data/anaconda3/bin/python3 -m verl.trainer.serve_ppo \
    actor_rollout_ref.model.path=/data/models/Qwen3-4B \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.rollout.enable_sleep_mode=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.logger='["console"]' \
    +trainer.serve_port=${PORT} \
    ++trainer.idle_timeout=${IDLE_TO} \
    ++trainer.min_samples=1 \
    ++trainer.gsm8k.enabled=true \
    ++trainer.gsm8k.num_prompts=${GSM8K_N} \
    ++trainer.train_steps_per_cycle=${STEPS} \
    ++trainer.prompts_per_step=${PROMPTS_PER} \
    ++actor_rollout_ref.rollout.response_length=${RESP_LEN} \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# ---- Step 3: Wait for server ready + monitor training ----
echo "=== [3/4] Waiting for server ready + training ==="

MAX_WAIT=180
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s --max-time 3 "http://localhost:${PORT}/v1/health" > /dev/null 2>&1; then
        HEALTH=$(curl -s "http://localhost:${PORT}/v1/health")
        echo "  Server ready: $HEALTH"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "  ERROR: Server did not start within ${MAX_WAIT}s"
    tail -20 "$SERVER_LOG"
    exit 1
fi

echo "  Waiting for training trigger..."
sleep $((IDLE_TO + 5))

# Monitor training progress
PREV_LINES=0
STEP=0
while true; do
    HEALTH=$(curl -s --max-time 3 "http://localhost:${PORT}/v1/health" 2>/dev/null || echo '{"mode":"error"}')
    MODE=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('mode','error'))" 2>/dev/null || echo "error")

    if [ -f "$TRAIN_LOG" ]; then
        CUR_LINES=$(wc -l < "$TRAIN_LOG" 2>/dev/null || echo 0)
        if [ "$CUR_LINES" -gt "$PREV_LINES" ]; then
            NEW=$(tail -n $((CUR_LINES - PREV_LINES)) "$TRAIN_LOG")
            echo "$NEW" | grep -E "Rewards:|train_step completed|Actor update|Gen done" | while read -r line; do
                TIMESTAMP=$(date +%H:%M:%S)
                echo "  [$TIMESTAMP] $line"
            done
            PREV_LINES=$CUR_LINES
        fi
    fi

    STEPS_DONE=$(grep -c "train_step completed" "$TRAIN_LOG" 2>/dev/null || echo 0)
    if [ "$STEPS_DONE" -ge "$STEPS" ] && echo "$HEALTH" | grep -q '"mode":"serving"'; then
        echo "  All ${STEPS} steps done, server back to serving!"
        break
    fi

    sleep 15
done

# ---- Step 4: Summary ----
echo ""
echo "============================================================================"
echo "  Summary — GSM8K GRPO Training (${STEPS} steps, batch=$((PROMPTS_PER * ROLLOUT_N)))"
echo "============================================================================"

echo ""
echo "  Rewards trend:"
grep "Rewards:" "$TRAIN_LOG" | nl -w2 -s'. ' || echo "  (no data)"

echo ""
echo "  Loss trend:"
grep "loss=" "$TRAIN_LOG" | grep "train_step completed" | nl -w2 -s'. ' || echo "  (no data)"

echo ""
echo "  Final health:"
curl -s "http://localhost:${PORT}/v1/health" | python3 -m json.tool 2>/dev/null || echo "  (unreachable)"

echo ""
echo "  Server log: $SERVER_LOG"
echo "  Train log:  $TRAIN_LOG"
echo "  Stop:       kill $SERVER_PID"
echo "============================================================================"
