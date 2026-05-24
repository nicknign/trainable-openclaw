#!/usr/bin/env bash
# Native veRL GRPO + LoRA training test | Qwen3-4B | Single GPU
# Purpose: Verify veRL's native GRPO training works on GSM8K before integrating with serve_ppo
set -xeuo pipefail

export PATH=/data/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/data/anaconda3/lib:${LD_LIBRARY_PATH:-}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---- config ----
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3-4B}"
TRAIN_FILE="$HOME/data/gsm8k/train.parquet"
TEST_FILE="$HOME/data/gsm8k/test.parquet"

NGPUS_PER_NODE=1
NNODES=1

# Small batch sizes for single 48GB GPU with LoRA co-located vLLM+FSDP
TRAIN_BATCH_SIZE=8
PPO_MINI_BATCH_SIZE=4
PPO_MICRO_BATCH_SIZE_PER_GPU=1
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
PPO_MAX_TOKEN_LEN_PER_GPU=3072

ACTOR_LR=3e-6
KL_LOSS_COEF=0.001
ENTROPY_COEFF=0

LORA_RANK=16
LORA_ALPHA=32

ROLLOUT_TP=1
ROLLOUT_GPU_MEM_UTIL=0.35
ROLLOUT_N=4

TOTAL_TRAINING_STEPS=5
SAVE_FREQ=-1
TEST_FREQ=-1

echo "=== Native GRPO+LoRA Training ==="
echo "Model:      $MODEL_PATH"
echo "Train data: $TRAIN_FILE"
echo "LoRA rank:  $LORA_RANK"
echo "Steps:      $TOTAL_TRAINING_STEPS"
echo "Python:     $(which python3)"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    ++actor_rollout_ref.rollout.enable_sleep_mode=false \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.balance_batch=True \
    trainer.logger='["console"]' \
    trainer.project_name=verl_native_grpo_lora_test \
    trainer.experiment_name=qwen3_4b_lora_gsm8k_test \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.val_before_train=False \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    "$@"

echo "=== Training completed with exit code: $? ==="
