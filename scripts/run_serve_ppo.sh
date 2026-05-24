#!/usr/bin/env bash
# serve_ppo startup script — inference + idle detection + GSM8K GRPO training
# Runs on single GPU (RTX 3090 48GB), Qwen3-4B + LoRA (rank=16)
set -xeuo pipefail

export PATH=/data/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/data/anaconda3/lib:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

/usr/bin/python3 -m verl.trainer.serve_ppo \
    actor_rollout_ref.model.path=/data/models/Qwen3-4B \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.rollout.enable_sleep_mode=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
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
    +trainer.serve_port=8000 \
    +trainer.idle_timeout=30 \
    +trainer.min_samples=1 \
    +trainer.gsm8k.enabled=true \
    +trainer.gsm8k.num_prompts=20 \
    +trainer.train_steps_per_cycle=5 \
    +trainer.prompts_per_step=4 \
    "$@"
