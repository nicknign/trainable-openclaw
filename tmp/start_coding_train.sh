#!/bin/bash
# Coding-focused training experiment — 80 train / 20 test, coding-specific rubrics
# Every 10 training steps → test set evaluation
set -e

export LD_LIBRARY_PATH=/data/anaconda3/lib:$LD_LIBRARY_PATH
cd /data/wangye/trainable-openclaw

# Kill any existing serve_ppo
pkill -f "serve_ppo" 2>/dev/null || true
sleep 2

echo "============================================"
echo "  Coding-Focused Training Experiment"
echo "============================================"
echo "Model: Qwen3-4B + LoRA rank=16"
echo "Train: 80 coding prompts (22 with correction pairs)"
echo "Test:  20 coding prompts"
echo "Rubrics: 6 coding-specific (auto-generated)"
echo "Config: idle=5s, 10 steps/cycle, 32 prompts/step, rollout_n=4, lr=1e-5"
echo "Eval: every 10 steps on test set, judge=deepseek-v4-flash"
echo "Checkpoint: disabled (focus on online eval)"
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
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_model_len=4096 \
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
    +trainer.idle_timeout=5 \
    +trainer.min_samples=0 \
    +trainer.max_train_rounds=10 \
    +trainer.train_steps_per_cycle=10 \
    +trainer.prompts_per_step=32 \
    +trainer.save_ckpt_interval=0 \
    +trainer.trajectory.enabled=true \
    +trainer.trajectory.data_path=data/coding/train_all.jsonl \
    +trainer.trajectory.rubrics_path=data/rubrics_coding.json \
    +trainer.trajectory.api_key=sk-906ad0dc48354e7aba594ef6d9aa5be6 \
    +trainer.trajectory.max_rubrics=6 \
    +trainer.trajectory.reward_mode=mean \
    +trainer.eval.enabled=true \
    +trainer.eval.interval=10 \
    +trainer.eval.test_data_path=data/coding/test.jsonl \
    > /tmp/coding_train.log 2>&1 &

echo "Server PID: $!"
echo "Log: /tmp/coding_train.log"
echo "Train step log: /tmp/serve_ppo_train.log"
echo ""
echo "Monitor: tail -f /tmp/coding_train.log"
echo "Steps:   tail -f /tmp/serve_ppo_train.log"
