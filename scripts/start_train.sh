#!/bin/bash
# Phase 3: 初期训练闭环 — start serve_ppo with rubric-based GRPO training
set -e

export LD_LIBRARY_PATH=/data/anaconda3/lib:$LD_LIBRARY_PATH
cd /data/wangye/trainable-openclaw

# Kill any existing serve_ppo
pkill -f "serve_ppo" 2>/dev/null || true
sleep 2

echo "=== Starting Phase 3 Training Server ==="
echo "Model: Qwen3-4B + LoRA rank=16"
echo "Training data: data/training_pairs.jsonl (48 prompts, 33 unique)"
echo "Rubrics: data/rubrics_v2.json (5 high-quality rubrics)"
echo "Reward: rubric-based via DeepSeek-v4-flash judge (thinking mode)"
echo "Config: idle=30s, 10 steps/cycle, 8 prompts/step, rollout_n=8, lr=5e-6, max_rounds=10"
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
    actor_rollout_ref.rollout.n=8 \
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
    +trainer.serve_port=8000 \
    +trainer.idle_timeout=30 \
    +trainer.min_samples=0 \
    +trainer.max_train_rounds=10 \
    +trainer.train_steps_per_cycle=10 \
    +trainer.prompts_per_step=8 \
    +trainer.trajectory.enabled=true \
    +trainer.trajectory.data_path=data/training_pairs.jsonl \
    +trainer.trajectory.rubrics_path=data/rubrics_v2.json \
    +trainer.trajectory.api_key=sk-906ad0dc48354e7aba594ef6d9aa5be6 \
    +trainer.trajectory.max_rubrics=8 \
    +trainer.trajectory.reward_mode=mean \
    > /tmp/phase3_train.log 2>&1 &

echo "Server PID: $!"
echo "Log: /tmp/phase3_train.log"
echo "Train step log: /tmp/serve_ppo_train.log"
echo ""
echo "Monitor with: tail -f /tmp/phase3_train.log"
echo "Train steps:  tail -f /tmp/serve_ppo_train.log"
