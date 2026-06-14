#!/bin/bash
# Phase 3: 初期训练闭环 — start serve_ppo with rubric-based GRPO training
set -e

export LD_LIBRARY_PATH=/data/anaconda3/lib:$LD_LIBRARY_PATH
cd /data/wangye/trainable-openclaw

# Kill any existing serve_ppo
pkill -f "serve_ppo" 2>/dev/null || true
sleep 2

echo "=== Starting Coding-Only Training Server ==="
echo "Model: Qwen3-4B + LoRA rank=16"
echo "Training data: data/coding/train_all.jsonl (80 coding prompts)"
echo "Rubrics: data/rubrics_coding_v4.json (4 evolved coding rubrics)"
echo "Reward: rubric-based via DeepSeek-v4-flash judge (individual mode, no thinking, max_tokens=2048)"
echo "Config: idle=30s, 20 steps/cycle, 16 prompts/step (from 80), rollout_n=4, lr=1e-5, max_rounds=5"
echo "Tokens: prompt_length=2048, response_length=4096, max_model_len=8192"
echo "System prompt: code generator mode, temperature=0.6"
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
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.rollout.enable_sleep_mode=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
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
    +trainer.serve_port=8000 \
    +trainer.idle_timeout=30 \
    +trainer.min_samples=0 \
    +trainer.max_train_rounds=5 \
    +trainer.train_steps_per_cycle=20 \
    +trainer.prompts_per_step=16 \
    +trainer.trajectory.enabled=true \
    +trainer.trajectory.data_path=data/coding/train_all.jsonl \
    +trainer.trajectory.rubrics_path=data/rubrics_coding_v4.json \
    +trainer.trajectory.api_key="${DEEPSEEK_API_KEY:?}" \
    +trainer.trajectory.max_rubrics=4 \
    +trainer.trajectory.reward_mode=mean \
    +trainer.save_ckpt_interval=10 \
    +trainer.checkpoint_dir=/data/wangye/trainable-openclaw/checkpoints \
    > /tmp/phase3_train.log 2>&1 &

echo "Server PID: $!"
echo "Log: /tmp/phase3_train.log"
echo "Train step log: /tmp/serve_ppo_train.log"
echo ""
echo "Monitor with: tail -f /tmp/phase3_train.log"
echo "Train steps:  tail -f /tmp/serve_ppo_train.log"
