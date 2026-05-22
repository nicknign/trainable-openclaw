# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Trainable OpenClaw Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Serve-only mode entry point for veRL.

Takes the veRL hybrid engine (ActorRolloutRefWorker) and runs it as a persistent
inference API server.  No training loop — the rollout engine stays awake and serves
requests via an OpenAI-compatible chat completions endpoint.

Usage:
    python -m verl.trainer.serve_ppo \\
        config=ppo_trainer \\
        actor_rollout_ref.model.path=/path/to/model \\
        actor_rollout_ref.rollout.name=vllm \\
        trainer.serve_port=8000

Architecture:
    Driver process (serve_ppo.py)
        ├── Ray init
        ├── ServeRunner (Ray actor) — manages veRL workers + LLMServerManager
        └── FastAPI — receives HTTP requests, calls LLMServerClient.generate()
"""

import logging
import os
import socket
import time

import numpy as np
import hydra
import ray
import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import TaskRunner
from trainable_openclaw.server.api import (  # standalone — no verl deps
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    HealthResponse,
    UsageInfo,
    _app_state,
    create_app,
    logger,
)
from verl.utils.device import auto_set_device


# ---------------------------------------------------------------------------
# ServeRunner — Ray actor that manages the veRL serving engine
# ---------------------------------------------------------------------------


class ServeRunner(TaskRunner):
    """Ray remote class for serving-only mode.

    Inherits from TaskRunner to reuse ``add_actor_rollout_worker()``,
    ``init_resource_pool_mgr()``, and the role-mapping infrastructure.
    """

    def run(self, config) -> dict:
        """Initialize veRL workers in serving-only mode and return client info.

        This method sets up the actor-rollout hybrid worker, launches the
        LLM server replicas, creates the client, and keeps the engine awake
        (no sleep/train cycle).

        Returns:
            dict with tokenizer path and rollout config for the driver.
        """
        from pprint import pprint

        from verl.trainer.ppo.ray_trainer import Role
        from verl.utils.fs import copy_to_local

        print(f"ServeRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # ---- Step 1: Worker class registration (actor/rollout only) ----
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        # No critic, no reward model, no ref policy, no teacher

        from verl.utils.config import validate_config
        from verl.trainer.ppo.utils import need_critic, need_reference_policy

        validate_config(config, use_reference_policy=False, use_critic=False)

        # ---- Step 2: Tokenizer ----
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_tokenizer

        tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.get("trust_remote_code", False))

        # ---- Step 3: Resource pool ----
        resource_pool_manager = self.init_resource_pool_mgr(config)
        resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        actor_rollout_resource_pool = resource_pool_manager.get_resource_pool(actor_role)

        from verl.single_controller.ray import RayClassWithInitArgs, create_colocated_worker_cls

        actor_rollout_init = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=config.actor_rollout_ref,
            role=str(actor_role),
        )
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_init

        # ---- Step 4: Create worker group ----
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg = ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            spawn_wg = wg.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        # ---- Step 5: LLMServerManager (launches vLLM/SGLang replicas) ----
        from verl.workers.rollout.llm_server import LLMServerManager

        self.llm_server_manager = LLMServerManager.create(
            config=config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
        )

        llm_client = self.llm_server_manager.get_client()
        print(f"LLMServerManager ready. Replicas: {len(self.llm_server_manager.server_addresses)}")

        # ---- Step 5b: CheckpointEngineManager (sleep/wake/sync orchestrator) ----
        from verl.utils.config import omega_conf_to_dataclass

        checkpoint_engine_config = omega_conf_to_dataclass(
            config.actor_rollout_ref.rollout.checkpoint_engine
        )
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )
        print(f"CheckpointEngineManager ready. Backend: {checkpoint_engine_config.backend}")

        # ---- Step 6: Keep engine awake — NO sleep_replicas ----
        # The engine is already awake after init_model.  In the training flow
        # checkpoint_manager.sleep_replicas() is called to free GPU memory,
        # but for serving we keep everything hot.

        gpu_count = config.trainer.n_gpus_per_node * config.trainer.nnodes

        self._local_path = local_path
        self._gpu_count = gpu_count
        self._rollout_config = config.actor_rollout_ref.rollout

        return {
            "tokenizer_path": local_path,
            "rollout_config": OmegaConf.to_container(config.actor_rollout_ref.rollout, resolve=True),
            "gpu_count": gpu_count,
        }

    def get_llm_client(self, _config=None):
        """Return the LLMServerClient to the driver.

        The returned object is serialized by Ray.  ActorHandles inside it
        survive serialization, so the driver can call ``generate()`` which
        internally does Ray RPC to the LLM server replicas.
        """
        return self.llm_server_manager.get_client()

    # ------------------------------------------------------------------
    # A2: Sleep / Wake / Train — called by TrainingOrchestrator
    # ------------------------------------------------------------------

    def sleep_replicas(self):
        """Put all rollout replicas to sleep to free GPU memory for training."""
        self.checkpoint_manager.sleep_replicas()
        logger.info("All rollout replicas asleep")

    def wake_replicas(self):
        """Wake all rollout replicas after training to resume inference."""
        self.checkpoint_manager.wake_up_replicas()
        logger.info("All rollout replicas awake")

    def train_step(self, training_data: dict) -> None:
        """Execute one GRPO training step using veRLʼs hybrid engine.

        Args:
            training_data: dict with:
                - prompts: list[list[int]] — replicated prompt token IDs
                - responses: list[list[int]] — generated response token IDs
                - rewards: list[float] — per-sample rewards
                - n_prompts: int — number of unique prompts (for GRPO grouping)

        Follows the RayPPOTrainer.fit() single-step pattern:
        1. Sleep replicas via CheckpointEngineManager (frees GPU memory)
        2. Build DataProto batch from prompts + responses + rewards
        3. Compute old_log_probs via actor forward
        4. Compute GRPO advantages (group-norm per prompt)
        5. Update actor (FSDP forward/backward/optimizer step)
        6. Update weights to rollout via CheckpointEngineManager (naive/IPC)
        7. Wake replicas via CheckpointEngineManager (restore for serving)
        """
        import time

        prompts = training_data["prompts"]
        responses = training_data["responses"]
        rewards = training_data["rewards"]
        n_prompts = training_data.get("n_prompts", len(prompts))

        logger.info(
            "train_step: %d samples (%d prompts x %d responses)",
            len(responses), n_prompts, len(responses) // max(n_prompts, 1),
        )
        t_start = time.time()

        # ---- 1. Sleep replicas (free GPU memory for training) ----
        logger.info("Sleeping rollout replicas via CheckpointEngineManager...")
        self.checkpoint_manager.sleep_replicas()
        logger.info("Replicas asleep — building training batch")

        # ---- 2. Build DataProto batch ----
        rollout_n = len(responses) // n_prompts
        batch = self._build_training_batch(prompts, responses, rewards, rollout_n)
        logger.info("Training batch built: %s", {k: v.shape for k, v in batch.batch.items()})

        # ---- 3. Compute old_log_probs ----
        from verl.trainer.ppo.ray_trainer import compute_response_mask
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding

        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)

        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)

        old_log_prob_output = self.actor_rollout_wg.compute_log_prob(batch_td)
        logger.info("old_log_probs computed")

        # Merge old_log_probs back
        batch = batch.union(DataProto.from_tensordict(old_log_prob_output))
        if "old_log_probs" not in batch.batch:
            # compute_log_prob returns log_probs, rename
            batch.batch["old_log_probs"] = old_log_prob_output["log_probs"]

        # ---- 4. Compute GRPO advantages ----
        from verl.trainer.ppo.ray_trainer import compute_advantage
        from verl.trainer.ppo.core_algos import AdvantageEstimator

        # Create uid for group-based advantage: same uid = same prompt group
        batch.non_tensor_batch["uid"] = np.array(
            [f"p{i // rollout_n}" for i in range(len(responses))], dtype=object
        )
        batch = compute_advantage(
            batch,
            adv_estimator=AdvantageEstimator.GRPO,
            gamma=1.0,
            lam=1.0,
            num_repeat=rollout_n,
            norm_adv_by_std_in_grpo=True,
        )
        logger.info("GRPO advantages computed")

        # ---- 5. Update actor ----
        from verl.utils.tensordict_utils import assign_non_tensor

        batch.meta_info["multi_turn"] = False
        batch.meta_info["temperature"] = self._rollout_config.get("temperature", 1.0)
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)

        ppo_mini_batch_size = len(responses)
        assign_non_tensor(
            batch_td,
            calculate_entropy=True,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=1,
            seed=42,
            dataloader_kwargs={"shuffle": False},
        )

        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        logger.info("Actor update completed")

        # ---- 6. Update weights (sync actor → rollout via naive IPC) ----
        global_steps = getattr(self, "_global_steps", 0) + 1
        self._global_steps = global_steps
        logger.info("Syncing weights to rollout (global_steps=%d)", global_steps)
        t_sync = time.time()
        self.checkpoint_manager.update_weights(global_steps)
        logger.info("Weight sync done in %.1fs", time.time() - t_sync)

        # ---- 7. Wake replicas (restore vLLM engine for serving) ----
        logger.info("Waking rollout replicas via CheckpointEngineManager...")
        self.checkpoint_manager.wake_up_replicas()
        logger.info("Replicas awake — resuming serving")

        logger.info("train_step completed in %.1fs", time.time() - t_start)

    # ------------------------------------------------------------------
    # Batch construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_training_batch(
        prompts: list[list[int]],
        responses: list[list[int]],
        rewards: list[float],
        rollout_n: int,
    ) -> DataProto:
        """Build a DataProto training batch from raw token sequences.

        Constructs input_ids, attention_mask, position_ids, response_mask,
        and token_level_scores from prompt/response token lists.
        """
        import torch

        n = len(prompts)

        # Pad prompts (left-padding to match veRL convention)
        max_prompt_len = max(len(p) for p in prompts)
        padded_prompts = []
        for p in prompts:
            pad_len = max_prompt_len - len(p)
            padded_prompts.append([0] * pad_len + p)

        # Pad responses (right-padding)
        max_resp_len = max(len(r) for r in responses)
        padded_responses = []
        for r in responses:
            pad_len = max_resp_len - len(r)
            padded_responses.append(r + [0] * pad_len)

        total_len = max_prompt_len + max_resp_len
        input_ids = torch.zeros(n, total_len, dtype=torch.long)
        attention_mask = torch.zeros(n, total_len, dtype=torch.long)
        position_ids = torch.zeros(n, total_len, dtype=torch.long)

        prompt_tensor = torch.tensor(padded_prompts, dtype=torch.long)
        response_tensor = torch.tensor(padded_responses, dtype=torch.long)

        for i in range(n):
            # Prompt: left-padded
            prompt = prompt_tensor[i]
            input_ids[i, :max_prompt_len] = prompt
            prompt_mask = prompt != 0
            attention_mask[i, :max_prompt_len] = prompt_mask
            position_ids[i, :max_prompt_len] = torch.cumsum(prompt_mask, dim=0) - prompt_mask.long()

            # Response: after prompt
            resp = response_tensor[i]
            input_ids[i, max_prompt_len:] = resp
            resp_mask = resp != 0
            attention_mask[i, max_prompt_len:] = resp_mask
            position_ids[i, max_prompt_len:] = position_ids[i, max_prompt_len - 1] + torch.cumsum(resp_mask, dim=0) + (1 - resp_mask.long())

        # Token-level scores: broadcast reward to each response token
        token_level_scores = torch.zeros(n, max_resp_len)
        for i, (r, reward) in enumerate(zip(responses, rewards)):
            resp_len = len(r)
            token_level_scores[i, :resp_len] = reward

        # UIDs for group advantage
        uid = np.array([f"p{i // rollout_n}_r{i}" for i in range(n)], dtype=object)

        batch_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_tensor,
            "response_mask": response_tensor != 0,
            "token_level_scores": token_level_scores,
        }
        non_tensor_data = {
            "uid": uid,
            "data_source": np.array(["gsm8k"] * n, dtype=object),
        }

        return DataProto.from_dict(tensors=batch_data, non_tensors=non_tensor_data)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Serve-only entry point.

    Reads the same ``ppo_trainer`` Hydra config as ``main_ppo``, but instead of
    launching a full training loop it starts a persistent FastAPI inference server.
    """
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_serve(config)


def _extract_gsm8k_answer(text: str) -> float | None:
    """Extract the final numeric answer from a GSM8K-style response.

    Looks for ``#### <number>`` at the end of the text and returns the
    number as a float, or None if no answer delimiter is found.
    """
    if "####" not in text:
        return None
    answer_part = text.split("####")[-1].strip()
    import re

    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer_part)
    if not numbers:
        return None
    return float(numbers[-1].replace(",", ""))


def _load_gsm8k_data(config, tokenizer) -> list[dict]:
    """Load a batch of GSM8K prompts for GRPO training validation.

    Returns a list of dicts with ``prompt_ids`` and ``ground_truth``.
    Returns an empty list if the dataset cannot be loaded.
    """
    try:
        from datasets import load_dataset
        from verl.utils import hf_tokenizer
    except Exception:
        logger.warning("Cannot load GSM8K data — datasets library unavailable")
        return []

    gsm8k_config = config.trainer.get("gsm8k", {})
    if not gsm8k_config.get("enabled", False):
        logger.info("GSM8K training data loading disabled (trainer.gsm8k.enabled=false)")
        return []

    num_prompts = gsm8k_config.get("num_prompts", 8)
    split = gsm8k_config.get("split", "train")
    prompt_template = gsm8k_config.get(
        "prompt_template",
        "{question}\nLet's think step by step.\n",
    )

    try:
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        if num_prompts < len(dataset):
            dataset = dataset.select(range(num_prompts))
    except Exception as e:
        logger.warning("Failed to load GSM8K dataset: %s", e)
        return []

    data = []
    for row in dataset:
        question = row["question"]
        # Extract ground truth number from answer (e.g., "#### 42")
        answer_text = row["answer"]
        gt = _extract_gsm8k_answer(answer_text)

        prompt_text = prompt_template.format(question=question)
        prompt_ids = tokenizer.encode(prompt_text)

        data.append({"prompt_ids": prompt_ids, "ground_truth": gt, "question": question})

    logger.info("Loaded %d GSM8K prompts for training (split=%s)", len(data), split)
    return data


def run_serve(config) -> None:
    """Initialize Ray, create ServeRunner actor, and start FastAPI on the driver."""
    from verl.utils.device import is_cuda_available

    # In serve-only mode, load_format must NOT be "dummy" — there is no
    # training engine to sync real weights, so vLLM would generate garbage.
    load_format = OmegaConf.select(config, "actor_rollout_ref.rollout.load_format")
    if load_format is None or load_format == "dummy":
        config.actor_rollout_ref.rollout.load_format = "auto"
        print(f"[serve_ppo] load_format={load_format or 'default'} → auto (serve-only mode)")

    # Ensure naive checkpoint engine backend (direct IPC, no NCCL needed)
    ckpt_backend = OmegaConf.select(config, "actor_rollout_ref.rollout.checkpoint_engine.backend")
    if ckpt_backend is None or ckpt_backend != "naive":
        config.actor_rollout_ref.rollout.checkpoint_engine.backend = "naive"
        print(f"[serve_ppo] checkpoint_engine.backend={ckpt_backend or 'default'} → naive")

    # ---- Ray init ----
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # ---- Create ServeRunner and init ----
    runner = ray.remote(num_cpus=1)(ServeRunner).remote()
    info = ray.get(runner.run.remote(config))

    # ---- Load tokenizer on driver (needed for API endpoints) ----
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(info["tokenizer_path"], trust_remote_code=config.data.get("trust_remote_code", False))

    # ---- Get LLM client from the Ray actor ----
    llm_client = ray.get(runner.get_llm_client.remote(config))

    # ---- Store state for FastAPI endpoints ----
    _app_state["llm_client"] = llm_client
    _app_state["tokenizer"] = tokenizer
    _app_state["rollout_config"] = info["rollout_config"]
    _app_state["gpu_count"] = info["gpu_count"]
    _app_state["active_requests"] = 0
    _app_state["runner"] = runner

    # ---- A2: Training orchestrator (idle detection + training trigger) ----
    from trainable_openclaw.training.orchestrator import TrainingOrchestrator

    orchestrator = TrainingOrchestrator(
        idle_timeout=config.trainer.get("idle_timeout", 60.0),
        min_samples=config.trainer.get("min_samples", 16),
    )

    # ---- A3: Pre-load GSM8K training data (for GRPO training validation) ----
    gsm8k_data = _load_gsm8k_data(config, tokenizer)

    # Wire external-data check so GSM8K prompts bypass the min_samples gate
    orchestrator.set_external_data_check(lambda: bool(gsm8k_data))

    def _train_bridge(samples):
        """Bridge: orchestrator thread → generate responses → train on Ray actor.

        Runs in the orchestrator monitor thread (not the uvicorn event loop),
        so asyncio.run() works for calling the async LLM client.
        """
        import asyncio
        from uuid import uuid4

        rollout_config = info["rollout_config"]
        rollout_n = rollout_config.get("n", 4)

        # Build prompt replicas
        all_prompts = []
        all_ground_truths = []

        if gsm8k_data:
            # Use pre-loaded GSM8K data for training
            for item in gsm8k_data:
                for _ in range(rollout_n):
                    all_prompts.append(item["prompt_ids"])
                    all_ground_truths.append(item["ground_truth"])
            n_prompts = len(gsm8k_data)
        elif samples:
            # Fall back to accumulated API samples
            for s in samples:
                for _ in range(rollout_n):
                    all_prompts.append(s.prompt_ids)
                    all_ground_truths.append(s.metadata.get("ground_truth"))
            n_prompts = len(samples)
        else:
            logger.warning("No training data available — skipping training")
            return

        logger.info(
            "Generating %d responses (%d prompts x %d)...",
            len(all_prompts), n_prompts, rollout_n,
        )

        # Generate N responses per prompt using vLLM servers
        sampling_params = {
            "temperature": rollout_config.get("temperature", 1.0),
            "top_p": rollout_config.get("top_p", 1.0),
            "max_tokens": rollout_config.get("response_length", 2048),
        }

        async def _generate_all():
            return await asyncio.gather(*[
                llm_client.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )
                for prompt_ids in all_prompts
            ])

        outputs = asyncio.run(_generate_all())
        all_responses = [list(o.token_ids) for o in outputs]

        # Compute rewards (GSM8K answer extraction)
        rewards = []
        for response_ids, gt in zip(all_responses, all_ground_truths):
            if gt is not None:
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                extracted = _extract_gsm8k_answer(response_text)
                reward = 1.0 if extracted is not None and abs(extracted - gt) < 1e-6 else 0.0
            else:
                reward = 0.0
            rewards.append(reward)

        logger.info(
            "Rewards: mean=%.3f, %d/%d correct",
            sum(rewards) / len(rewards), sum(r > 0.5 for r in rewards), len(rewards),
        )

        training_data = {
            "prompts": all_prompts,
            "responses": all_responses,
            "rewards": rewards,
            "n_prompts": n_prompts,
        }
        ray.get(runner.train_step.remote(training_data))

    orchestrator.set_train_fn(_train_bridge)
    orchestrator.start_monitoring()
    _app_state["orchestrator"] = orchestrator

    # ---- Start FastAPI ----
    app = create_app()
    serve_port = config.trainer.get("serve_port", 8000)
    print(f"\n{'='*60}")
    print(f"  veRL Inference Server starting on http://0.0.0.0:{serve_port}")
    print(f"  Health check: http://localhost:{serve_port}/v1/health")
    print(f"  Chat API:     http://localhost:{serve_port}/v1/chat/completions")
    print(f"{'='*60}\n")

    uvicorn.run(app, host="0.0.0.0", port=serve_port, log_level="info")


if __name__ == "__main__":
    main()
