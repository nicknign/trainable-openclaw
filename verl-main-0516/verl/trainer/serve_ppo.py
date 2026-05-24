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
import sys
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
        self._tokenizer = tokenizer

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
            "server_addresses": self.llm_server_manager.server_addresses,
            "model_path": local_path,
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

    def train_step(self, training_data: dict) -> dict:
        """Execute one GRPO training step — generation, reward, and training.

        All generation and reward computation happens inside the Ray actor
        (not on the driver), avoiding the uvicorn event-loop deadlock.

        Args:
            training_data: dict with:
                - prompts: list[list[int]] — prompt token IDs (one per unique prompt)
                - ground_truths: list[float|None] — correct answer per prompt
                - rollout_n: int — responses per prompt
                - sampling_params: dict — temperature, max_tokens, etc.

        Returns:
            dict with reward_mean, reward_correct, reward_total, actor_loss,
            grad_norm, step_time_seconds.
        """
        import json
        import time
        import urllib.request

        # ---- Logging: write directly to file (Ray actor stdout/stderr are buffered) ----
        TRAIN_LOG = "/tmp/serve_ppo_train.log"
        def _log(msg: str) -> None:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(TRAIN_LOG, "a") as lf:
                lf.write(f"[{ts}] [TRAIN] {msg}\n")

        prompts = training_data["prompts"]
        ground_truths = training_data["ground_truths"]
        rollout_n = training_data["rollout_n"]
        sampling_params = training_data["sampling_params"]
        n_prompts = len(prompts)
        total_samples = n_prompts * rollout_n

        _log(f"train_step start: {n_prompts} prompts x {rollout_n} = {total_samples} total")
        logger.info(
            "train_step: %d prompts x %d responses = %d total",
            n_prompts, rollout_n, total_samples,
        )
        t_start = time.time()

        # ---- 1. Generate responses via vLLM HTTP API ----
        vllm_addr = self.llm_server_manager.server_addresses[0]
        vllm_url = f"http://{vllm_addr}/v1/chat/completions"
        model_path = self._local_path

        all_prompt_ids = []
        all_responses_text = []

        for i, prompt_ids in enumerate(prompts):
            prompt_text = self._tokenizer.decode(prompt_ids, skip_special_tokens=False)
            for j in range(rollout_n):
                req_data = json.dumps({
                    "model": model_path,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": sampling_params.get("max_tokens", 1024),
                    "temperature": sampling_params.get("temperature", 1.0),
                    "top_p": sampling_params.get("top_p", 1.0),
                }).encode()
                req = urllib.request.Request(
                    vllm_url, data=req_data,
                    headers={"Content-Type": "application/json"},
                )
                resp = urllib.request.urlopen(req, timeout=180)
                if resp.status != 200:
                    raise RuntimeError(f"vLLM HTTP {resp.status}: {resp.read()[:500]}")
                data = json.loads(resp.read())
                response_text = data["choices"][0]["message"]["content"]
                all_prompt_ids.append(prompt_ids)
                all_responses_text.append(response_text)
                _log(f"Gen {i * rollout_n + j + 1}/{total_samples}: prompt={len(prompt_ids)} resp={len(response_text)}")
                logger.info(
                    "Generated %d/%d: prompt_len=%d resp_len=%d",
                    i * rollout_n + j + 1, total_samples,
                    len(prompt_ids), len(response_text),
                )

        t_gen = time.time()
        _log(f"Generation done in {t_gen - t_start:.1f}s")
        logger.info("Generation done in %.1fs", t_gen - t_start)

        # ---- 2. Compute rewards (GSM8K answer matching) ----
        responses = []
        rewards = []
        all_ground_truths = [gt for gt in ground_truths for _ in range(rollout_n)]
        for prompt_ids, response_text, gt in zip(all_prompt_ids, all_responses_text, all_ground_truths):
            response_ids = self._tokenizer.encode(response_text, add_special_tokens=False)
            responses.append(response_ids)
            if gt is not None:
                extracted = _extract_gsm8k_answer(response_text)
                reward = 1.0 if extracted is not None and abs(extracted - gt) < 1e-6 else 0.0
            else:
                reward = 0.0
            rewards.append(reward)

        n_correct = sum(r > 0.5 for r in rewards)
        reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
        _log(f"Rewards: mean={reward_mean:.3f}, {n_correct}/{len(rewards)} correct")
        logger.info("Rewards: mean=%.3f, %d/%d correct", reward_mean, n_correct, len(rewards))

        # ---- 3. Sleep replicas (free GPU memory for training) ----
        _log("Sleeping replicas via CheckpointEngineManager...")
        self.checkpoint_manager.sleep_replicas()
        _log("Replicas asleep")

        # ---- 4. Build DataProto batch ----
        batch = self._build_training_batch(all_prompt_ids, responses, rewards, rollout_n)
        batch.meta_info["temperature"] = sampling_params.get("temperature", 1.0)
        batch.meta_info["multi_turn"] = False
        _log(f"Training batch built: { {k: v.shape for k, v in batch.batch.items()} }")
        logger.info("Training batch built: %s", {k: v.shape for k, v in batch.batch.items()})

        # ---- 5. Compute old_log_probs ----
        from verl.trainer.ppo.ray_trainer import compute_response_mask
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding

        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)

        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)

        old_log_prob_output = self.actor_rollout_wg.compute_log_prob(batch_td)
        _log("old_log_probs computed")
        logger.info("old_log_probs computed")

        from verl.workers.utils.padding import no_padding_2_padding

        log_probs = old_log_prob_output["log_probs"]
        old_log_probs = no_padding_2_padding(log_probs, batch_td)
        batch.batch["old_log_probs"] = old_log_probs.float()

        # ---- 6. Compute GRPO advantages ----
        from verl.trainer.ppo.ray_trainer import compute_advantage
        from verl.trainer.ppo.core_algos import AdvantageEstimator

        batch.non_tensor_batch["uid"] = np.array(
            [f"p{i // rollout_n}" for i in range(total_samples)], dtype=object
        )
        batch = compute_advantage(
            batch,
            adv_estimator=AdvantageEstimator.GRPO,
            gamma=1.0,
            lam=1.0,
            num_repeat=rollout_n,
            norm_adv_by_std_in_grpo=True,
        )
        _log("GRPO advantages computed")
        logger.info("GRPO advantages computed")

        # ---- 7. Update actor ----
        from verl.utils.tensordict_utils import assign_non_tensor

        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)

        assign_non_tensor(
            batch_td,
            calculate_entropy=True,
            global_batch_size=total_samples,
            mini_batch_size=total_samples,
            epochs=1,
            seed=42,
            dataloader_kwargs={"shuffle": False},
        )

        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_loss = float(actor_output.get("loss", 0.0)) if actor_output is not None else 0.0
        _log(f"Actor update completed — loss={actor_loss:.4f}")
        logger.info("Actor update completed — loss=%.4f", actor_loss)

        # ---- 8. Update weights (sync actor → rollout via naive IPC) ----
        global_steps = getattr(self, "_global_steps", 0) + 1
        self._global_steps = global_steps
        t_sync = time.time()
        self.checkpoint_manager.update_weights(global_steps)
        _log(f"Weight sync done in {time.time() - t_sync:.1f}s (wake handled internally)")
        logger.info("Weight sync done in %.1fs", time.time() - t_sync)

        total_time = time.time() - t_start
        _log(f"train_step completed in {total_time:.1f}s — reward={reward_mean:.3f} ({n_correct}/{len(rewards)}), loss={actor_loss:.4f}")
        logger.info("train_step completed in %.1fs", total_time)

        return {
            "reward_mean": reward_mean,
            "reward_correct": n_correct,
            "reward_total": len(rewards),
            "actor_loss": actor_loss,
            "step_time_seconds": total_time,
        }

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
            "prompts": prompt_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_tensor,
            "response_mask": response_tensor != 0,
            "token_level_scores": token_level_scores,
            "token_level_rewards": token_level_scores,
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

    # ---- A2: Async training orchestrator (runs on uvicorn event loop) ----
    # We avoid threads entirely because Ray actor calls (llm_client.generate,
    # server.acquire, etc.) hang when invoked from background threads.
    # Instead, the orchestrator state runs inline on the uvicorn event loop.
    import asyncio
    import time as _time

    # ---- A3: Pre-load GSM8K training data (for GRPO training validation) ----
    gsm8k_data = _load_gsm8k_data(config, tokenizer)

    # Orchestrator state (no threads)
    _orch_state = {
        "mode": "serving",
        "last_request_time": _time.time(),
        "samples": [],
        "training_in_progress": False,
        "idle_timeout": config.trainer.get("idle_timeout", 60.0),
        "min_samples": config.trainer.get("min_samples", 16),
        "poll_interval": 1.0,
    }
    _app_state["orch_state"] = _orch_state

    async def _record_request_async(prompt_ids, response_ids, metadata=None):
        """Non-blocking sample recording (called from API handlers on event loop)."""
        s = _app_state["active_requests"]
        _orch_state["last_request_time"] = _time.time()
        _orch_state["samples"].append({
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "metadata": metadata or {},
        })
        # Cap buffer
        if len(_orch_state["samples"]) > 10000:
            _orch_state["samples"] = _orch_state["samples"][-5000:]

    async def _async_monitor_loop():
        """Monitor idle state and trigger training — runs on uvicorn event loop."""

        logger.info(
            "Async training monitor started — idle_timeout=%ds, min_samples=%d",
            _orch_state["idle_timeout"], _orch_state["min_samples"],
        )

        while True:
            await asyncio.sleep(_orch_state["poll_interval"])

            if _orch_state["training_in_progress"]:
                continue

            # Check idle
            elapsed = _time.time() - _orch_state["last_request_time"]
            if elapsed < _orch_state["idle_timeout"]:
                continue

            # Check sample threshold
            # GSM8K data can bypass min_samples only on first use;
            # after that, require API samples to prevent infinite re-trigger
            n_samples = len(_orch_state["samples"])
            has_gsm8k = gsm8k_data and not _orch_state.get("_gsm8k_exhausted", False)
            if not has_gsm8k and n_samples < _orch_state["min_samples"]:
                continue

            # ---- Trigger training ----
            _orch_state["training_in_progress"] = True
            _orch_state["mode"] = "training"
            t_start = _time.time()

            # Drain samples
            samples = list(_orch_state["samples"])
            _orch_state["samples"].clear()

            logger.info("Training triggered — %d samples, idle=%.1fs", len(samples), elapsed)
            print(f"[TRAINING] Training triggered — {len(samples)} samples, idle={elapsed:.1f}s", file=sys.stderr, flush=True)

            try:
                rollout_config = info["rollout_config"]
                rollout_n = rollout_config.get("n", 4)
                all_prompts = []
                all_ground_truths = []

                if gsm8k_data:
                    for item in gsm8k_data:
                        all_prompts.append(item["prompt_ids"])
                        all_ground_truths.append(item["ground_truth"])
                elif samples:
                    for s in samples:
                        all_prompts.append(s["prompt_ids"])
                        all_ground_truths.append(s["metadata"].get("ground_truth"))
                else:
                    logger.warning("No training data — skipping")
                    continue

                sampling_params = {
                    "temperature": rollout_config.get("temperature", 1.0),
                    "top_p": rollout_config.get("top_p", 1.0),
                    "max_tokens": rollout_config.get("response_length", 2048),
                }

                training_data = {
                    "prompts": all_prompts,
                    "ground_truths": all_ground_truths,
                    "rollout_n": rollout_n,
                    "sampling_params": sampling_params,
                }

                # Generation, reward, and training all happen inside the actor
                metrics = ray.get(runner.train_step.remote(training_data))

                elapsed_train = _time.time() - t_start
                logger.info(
                    "Training complete in %.1fs — reward=%.3f (%d/%d), loss=%.4f",
                    elapsed_train,
                    metrics.get("reward_mean", 0),
                    metrics.get("reward_correct", 0),
                    metrics.get("reward_total", 1),
                    metrics.get("actor_loss", 0),
                )
                print(
                    f"[TRAINING] Training complete in {elapsed_train:.1f}s — "
                    f"reward={metrics.get('reward_mean', 0):.3f} "
                    f"({metrics.get('reward_correct', 0)}/{metrics.get('reward_total', 1)}), "
                    f"loss={metrics.get('actor_loss', 0):.4f}",
                    file=sys.stderr, flush=True,
                )

            except Exception:
                logger.exception("Training failed — resuming serving with old weights")
            finally:
                _orch_state["mode"] = "serving"
                _orch_state["training_in_progress"] = False
                _orch_state["last_request_time"] = _time.time()
                # Mark GSM8K exhausted if it was used without new API samples,
                # preventing repeated training on the same data with no new requests
                if gsm8k_data and len(samples) == 0:
                    _orch_state["_gsm8k_exhausted"] = True
                    logger.info("GSM8K data exhausted; future training requires API samples")
                    print("[TRAINING] GSM8K data exhausted — future training requires API samples",
                          file=sys.stderr, flush=True)

    _app_state["record_request"] = _record_request_async
    _app_state["orch_state"] = _orch_state
    _app_state["_monitor_coro"] = _async_monitor_loop
    _app_state["vllm_server_address"] = info["server_addresses"][0] if info.get("server_addresses") else None
    _app_state["model_path"] = info.get("model_path", "")

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
