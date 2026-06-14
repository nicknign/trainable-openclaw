#!/bin/bash
# Test checkpoint saving — run 1 training step and verify checkpoint files
set -e

export LD_LIBRARY_PATH=/data/anaconda3/lib:$LD_LIBRARY_PATH
cd /data/wangye/trainable-openclaw

# Kill any existing serve_ppo
pkill -9 -f "serve_ppo" 2>/dev/null || true
sleep 2

TEST_LOG=/data/wangye/trainable-openclaw/ckpt_test.log
CKPT_BASE=/data/wangye/trainable-openclaw/checkpoints

# Clean previous test checkpoints and logs
rm -rf "$CKPT_BASE"
rm -rf /tmp/ckpt_test 2>/dev/null || true
rm -f "$TEST_LOG" 2>/dev/null || true

echo "=== Checkpoint Test ==="
echo "Config: 1 step, save_ckpt_interval=1, checkpoint_dir=$CKPT_BASE"
echo "Log: $TEST_LOG"
echo ""

nohup /data/anaconda3/bin/python -m verl.trainer.serve_ppo \
    actor_rollout_ref.model.path=/data/models/Qwen3-4B \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.rollout.enable_sleep_mode=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
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
    +trainer.serve_port=8001 \
    +trainer.idle_timeout=5 \
    +trainer.min_samples=0 \
    +trainer.max_train_rounds=1 \
    +trainer.train_steps_per_cycle=1 \
    +trainer.prompts_per_step=2 \
    +trainer.trajectory.enabled=true \
    +trainer.trajectory.data_path=data/phase3_datasets/train_prompts.jsonl \
    +trainer.trajectory.rubrics_path=data/rubrics_dynamic.json \
    +trainer.trajectory.api_key="${DEEPSEEK_API_KEY:?}" \
    +trainer.trajectory.max_rubrics=2 \
    +trainer.trajectory.reward_mode=mean \
    +trainer.save_ckpt_interval=1 \
    +trainer.checkpoint_dir=$CKPT_BASE \
    > $TEST_LOG 2>&1 &

SERVE_PID=$!
echo "serve_ppo PID: $SERVE_PID"
echo "Waiting for server startup (~60s)..."

# Wait for server to be ready
for i in $(seq 1 120); do
    if curl -s http://localhost:8001/v1/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

# Verify health
HEALTH=$(curl -s http://localhost:8001/v1/health)
echo "Health: $HEALTH"

echo ""
echo "Waiting for training to trigger (idle_timeout=5s) and complete..."
echo "Monitor: tail -f $TEST_LOG"

# Wait for checkpoint directory to appear (poll up to 10 minutes)
FOUND=0
for i in $(seq 1 600); do
    if ls $CKPT_BASE/global_step_*/actor/model_world_size_*_rank_0.pt 2>/dev/null | head -1 | grep -q .; then
        FOUND=1
        echo ""
        echo "=== Checkpoint files found after ${i}s ==="
        break
    fi
    if grep -q "Training failed" $TEST_LOG 2>/dev/null; then
        echo ""
        echo "Training failed — check log:"
        tail -30 $TEST_LOG
        exit 1
    fi
    sleep 1
done

if [ $FOUND -eq 0 ]; then
    echo ""
    echo "=== TIMEOUT: No checkpoint found after 10 minutes ==="
    echo "Last 30 lines of log:"
    tail -30 $TEST_LOG
    exit 1
fi

# Verify checkpoint contents
echo ""
echo "=== Checkpoint Contents ==="
CKPT_DIR=$(ls -d $CKPT_BASE/global_step_*/actor 2>/dev/null | head -1)
if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint directory found"
    exit 1
fi

echo "Directory: $CKPT_DIR"
echo ""
echo "Files:"
find "$CKPT_DIR" -type f | sort
echo ""

# Check for key files
ERRORS=0
check_file() {
    local pattern="$1"
    local desc="$2"
    if ls "$CKPT_DIR"/$pattern 2>/dev/null | head -1 | grep -q .; then
        echo "  [PASS] $desc"
    else
        echo "  [FAIL] $desc — pattern: $pattern"
        ERRORS=$((ERRORS + 1))
    fi
}

# Verify model checkpoint exists and has reasonable size
MODEL_FILE=$(ls "$CKPT_DIR"/model_world_size_*_rank_0.pt 2>/dev/null | head -1)
if [ -n "$MODEL_FILE" ]; then
    SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 1024 ]; then
        echo "  [PASS] Model state dict ($SIZE bytes)"
    else
        echo "  [FAIL] Model state dict too small: $SIZE bytes"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "  [FAIL] Model state dict missing"
    ERRORS=$((ERRORS + 1))
fi

# HF config: optional but nice to have
if [ -d "$CKPT_DIR/huggingface" ]; then
    echo "  [PASS] HuggingFace config/tokenizer directory"
else
    echo "  [WARN] HuggingFace config/tokenizer missing (non-critical)"
fi

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "=== ALL CHECKS PASSED ==="
    echo "Checkpoint saved successfully at: $CKPT_DIR"
else
    echo "=== $ERRORS CHECK(S) FAILED ==="
    exit 1
fi

# Cleanup
echo ""
echo "Stopping test server (PID $SERVE_PID)..."
kill $SERVE_PID 2>/dev/null || true
sleep 2
pkill -9 -f "serve_ppo" 2>/dev/null || true

echo "Test complete. Checkpoint files preserved at: $CKPT_DIR"
