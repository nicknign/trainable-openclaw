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

        # ---- Init veRL Tracking logger (console + tensorboard) ----
        from verl.utils.tracking import Tracking

        self.config = config
        self.tracker = Tracking(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            default_backend=config.trainer.logger,
            config=OmegaConf.to_container(config, resolve=True),
        )

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

    def save_checkpoint(self, local_path: str, global_step: int = 0, max_ckpt_to_keep: int = None):
        """Save FSDP training checkpoint via actor_rollout_wg.

        Delegates to veRL's FSDPCheckpointManager which saves per-rank sharded
        model/optimizer/lr_scheduler/RNG state + HF config/tokenizer on rank 0.
        """
        import os as _os
        _os.makedirs(local_path, exist_ok=True)
        self.actor_rollout_wg.save_checkpoint(local_path, None, global_step, max_ckpt_to_keep)
        logger.info("Checkpoint saved to %s (step %d)", local_path, global_step)

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
        import torch
        import urllib.request
        from concurrent.futures import ThreadPoolExecutor, as_completed

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

        # ---- 1. Generate responses via vLLM HTTP API (concurrent) ----
        vllm_addr = self.llm_server_manager.server_addresses[0]
        vllm_url = f"http://{vllm_addr}/v1/chat/completions"
        model_path = self._local_path

        # Build all generation requests
        gen_requests = []
        for i, prompt_ids in enumerate(prompts):
            prompt_text = self._tokenizer.decode(prompt_ids, skip_special_tokens=False)
            for j in range(rollout_n):
                gen_requests.append({
                    "url": vllm_url,
                    "model": model_path,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": sampling_params.get("max_tokens", 2048),
                    "temperature": sampling_params.get("temperature", 1.0),
                    "top_p": sampling_params.get("top_p", 1.0),
                    "idx": i * rollout_n + j + 1,
                    "prompt_ids": prompt_ids,
                })

        def _do_generate(req):
            data = json.dumps({
                "model": req["model"],
                "messages": req["messages"],
                "max_tokens": req["max_tokens"],
                "temperature": req["temperature"],
                "top_p": req["top_p"],
            }).encode()
            r = urllib.request.Request(req["url"], data=data, headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(r, timeout=300)
            if resp.status != 200:
                raise RuntimeError(f"vLLM HTTP {resp.status}: {resp.read()[:500]}")
            body = json.loads(resp.read())
            return req["idx"], req["prompt_ids"], body["choices"][0]["message"]["content"]

        all_prompt_ids = [None] * len(gen_requests)
        all_responses_text = [None] * len(gen_requests)

        _log(f"Generating {len(gen_requests)} responses ({n_prompts} prompts x {rollout_n}) with {min(16, len(gen_requests))} concurrent workers")
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(_do_generate, r): r for r in gen_requests}
            for future in as_completed(futures):
                idx, prompt_ids, response_text = future.result()
                slot = idx - 1
                all_prompt_ids[slot] = prompt_ids
                all_responses_text[slot] = response_text
                _log(f"Gen {idx}/{total_samples}: prompt={len(prompt_ids)} resp={len(response_text)}")
                logger.info(
                    "Generated %d/%d: prompt_len=%d resp_len=%d",
                    idx, total_samples, len(prompt_ids), len(response_text),
                )

        t_gen = time.time()
        _log(f"Generation done in {t_gen - t_start:.1f}s")
        logger.info("Generation done in %.1fs", t_gen - t_start)

        # ---- 2. Compute rewards (Rubric-based via B3 Judge, or GSM8K fallback) ----
        rubrics_path = training_data.get("rubrics_path", "")
        api_key = training_data.get("api_key", "")
        use_rubric = bool(rubrics_path and api_key)

        responses = []
        rewards = []

        if use_rubric:
            from trainable_openclaw.training.reward_bridge import RewardBridge

            bridge = RewardBridge(
                rubrics_path=rubrics_path,
                api_key=api_key,
                base_url=training_data.get("base_url", ""),
                model=training_data.get("judge_model", ""),
                max_rubrics=training_data.get("max_rubrics", 0),
                reward_mode=training_data.get("reward_mode", "mean"),
                rubric_weights=training_data.get("rubric_weights"),
            )

            if not bridge.rubrics:
                _log("WARNING: No active rubrics found — falling back to zero rewards")
                for i, (prompt_ids, response_text) in enumerate(
                    zip(all_prompt_ids, all_responses_text)
                ):
                    response_ids = self._tokenizer.encode(response_text, add_special_tokens=False)
                    responses.append(response_ids)
                    rewards.append(0.0)
            else:
                # Group responses by prompt for per-prompt rubric scoring
                # prompt_texts: decode from prompt_ids if not provided
                prompt_texts = training_data.get("prompt_texts", None)
                if prompt_texts is None:
                    # Decode unique prompts (n_prompts, not total_samples)
                    unique_prompt_ids = [prompts[i] for i in range(n_prompts)]
                    prompt_texts = [
                        self._tokenizer.decode(ids, skip_special_tokens=False)
                        for ids in unique_prompt_ids
                    ]

                # Score each prompt's rollout_n responses
                for p_idx in range(n_prompts):
                    base = p_idx * rollout_n
                    group_responses = all_responses_text[base : base + rollout_n]

                    try:
                        group_rewards = bridge.score_responses(
                            prompt_texts[p_idx],
                            group_responses,
                        )
                    except Exception as e:
                        _log(f"Rubric scoring failed for prompt {p_idx}: {e} — zero rewards")
                        group_rewards = [0.0] * rollout_n

                    for j, (response_text, reward) in enumerate(
                        zip(group_responses, group_rewards)
                    ):
                        response_ids = self._tokenizer.encode(response_text, add_special_tokens=False)
                        responses.append(response_ids)
                        rewards.append(reward)

                    _log(
                        f"Prompt {p_idx + 1}/{n_prompts}: rewards={[round(r, 3) for r in group_rewards]}"
                    )
        else:
            # Legacy GSM8K reward path
            from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score

            all_ground_truths = [gt for gt in ground_truths for _ in range(rollout_n)]
            for i, (prompt_ids, response_text, gt) in enumerate(zip(all_prompt_ids, all_responses_text, all_ground_truths)):
                response_ids = self._tokenizer.encode(response_text, add_special_tokens=False)
                responses.append(response_ids)
                if gt is not None:
                    reward = gsm8k_compute_score(response_text, str(gt))
                else:
                    reward = 0.0
                rewards.append(reward)

        n_correct = sum(r > 0.5 for r in rewards)
        reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
        _log(f"Rewards: mean={reward_mean:.3f}, {n_correct}/{len(rewards)} >0.5 (mode={'rubric' if use_rubric else 'gsm8k'})")
        logger.info("Rewards: mean=%.3f, %d/%d >0.5", reward_mean, n_correct, len(rewards))

        # ---- 3. Sleep replicas (free GPU memory for training) ----
        _log("Sleeping replicas via CheckpointEngineManager...")
        self.checkpoint_manager.sleep_replicas()
        t_sleep = time.time()
        _log("Replicas asleep")

        # ---- 4. Build DataProto batch (structural only) ----
        batch = self._build_training_batch(all_prompt_ids, responses)
        batch.meta_info["temperature"] = sampling_params.get("temperature", 1.0)
        batch.meta_info["multi_turn"] = False
        _log(f"Training batch built: { {k: v.shape for k, v in batch.batch.items()} }")
        logger.info("Training batch built: %s", {k: v.shape for k, v in batch.batch.items()})

        # ---- 5. Compute rm_scores (last-token-only, matches veRL native) ----
        batch = self._compute_rm_scores(batch, rewards)
        _log("rm_scores computed (last-token placement)")
        logger.info("rm_scores computed (last-token placement)")

        # ---- 6. Compute response_mask (veRL native) ----
        from verl.trainer.ppo.ray_trainer import compute_response_mask

        batch.batch["response_mask"] = compute_response_mask(batch)
        _log("response_mask computed")
        logger.info("response_mask computed")

        # ---- 7. Compute old_log_probs ----
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding

        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)

        old_log_prob_output = self.actor_rollout_wg.compute_log_prob(batch_td)
        t_olp = time.time()
        _log("old_log_probs computed")
        logger.info("old_log_probs computed")

        from verl.workers.utils.padding import no_padding_2_padding

        log_probs = old_log_prob_output["log_probs"]
        old_log_probs = no_padding_2_padding(log_probs, batch_td)
        batch.batch["old_log_probs"] = old_log_probs.float()

        # ---- 8. Compute GRPO advantages ----
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
        t_adv = time.time()

        # ---- 9. Update actor (config-driven mini-batch) ----
        from verl.utils.tensordict_utils import assign_non_tensor

        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)

        actor_config = self.config.actor_rollout_ref.actor
        rollout_n_cfg = self.config.actor_rollout_ref.rollout.n
        # Cap mini_batch_size to actual batch size — serve_ppo batches are
        # small (n_prompts × rollout_n), unlike native veRL's large dataloader.
        effective_batch_size = min(
            actor_config.ppo_mini_batch_size * rollout_n_cfg,
            total_samples,
        )
        assign_non_tensor(
            batch_td,
            calculate_entropy=actor_config.calculate_entropy or (actor_config.entropy_coeff != 0.0),
            global_batch_size=effective_batch_size,
            mini_batch_size=effective_batch_size,
            epochs=actor_config.ppo_epochs,
            seed=actor_config.data_loader_seed,
            dataloader_kwargs={"shuffle": actor_config.shuffle},
            compute_loss=True,
        )

        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output_metrics = tu.get(actor_output, "metrics")
        from verl.utils.py_functional import rename_dict
        actor_metrics = rename_dict(actor_output_metrics, "actor/")
        t_actor = time.time()
        loss_val = actor_output_metrics.get("loss", 0.0) if actor_output_metrics else 0.0
        if isinstance(loss_val, list):
            loss_val = sum(loss_val) / len(loss_val) if loss_val else 0.0
        actor_loss = float(loss_val)
        _log(f"Actor update completed — loss={actor_loss:.4f}")
        logger.info("Actor update completed — loss=%.4f", actor_loss)

        # ---- 10. Update weights (sync actor → rollout via naive IPC) ----
        global_steps = getattr(self, "_global_steps", 0) + 1
        self._global_steps = global_steps
        t_sync = time.time()
        self.checkpoint_manager.update_weights(global_steps)
        _log(f"Weight sync done in {time.time() - t_sync:.1f}s (wake handled internally)")
        logger.info("Weight sync done in %.1fs", time.time() - t_sync)

        # ---- 11. Compute native metrics (data/timing/throughput) + tracker.log ----
        from verl.trainer.ppo.metric_utils import (
            compute_data_metrics,
            compute_timing_metrics,
            compute_throughout_metrics,
        )

        total_time = time.time() - t_start
        timing_raw = {
            "gen": t_gen - t_start,
            "old_log_prob": t_olp - t_sleep,
            "adv": t_adv - t_olp,
            "update_actor": t_actor - t_adv,
            "step": total_time,
        }

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        step_metrics = {}
        step_metrics.update(actor_metrics)
        step_metrics.update(compute_data_metrics(batch=batch, use_critic=False))
        step_metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        step_metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=self._gpu_count))
        step_metrics["training/global_step"] = global_steps

        def _s(key, default=0.0):
            """Safely extract a scalar from step_metrics (lists get averaged)."""
            v = step_metrics.get(key, default)
            if isinstance(v, list):
                return sum(v) / len(v) if v else default
            return v

        try:
            self.tracker.log(data=step_metrics, step=global_steps)
        except Exception:
            logger.warning("tracker.log failed (non-fatal)", exc_info=True)
        _log(
            f"train_step completed in {total_time:.1f}s — "
            f"reward={reward_mean:.3f} ({n_correct}/{len(rewards)}), "
            f"loss={actor_loss:.4f}, grad_norm={_s('actor/grad_norm'):.3f} | "
            f"score={_s('critic/score/mean'):.3f}, "
            f"rewards={_s('critic/rewards/mean'):.3f}, "
            f"adv={_s('critic/advantages/mean'):.3f} | "
            f"rlen={_s('response_length/mean'):.0f}, "
            f"plen={_s('prompt_length/mean'):.0f}, "
            f"abort={_s('response/aborted_ratio'):.3f} | "
            f"tps={_s('perf/throughput'):.1f} tok/s/gpu, "
            f"gen={_s('timing_per_token_ms/gen'):.2f}ms/tok, "
            f"actor={_s('timing_per_token_ms/update_actor'):.2f}ms/tok"
        )
        logger.info("train_step completed in %.1fs", total_time)

        return {
            "reward_mean": reward_mean,
            "reward_correct": n_correct,
            "reward_total": len(rewards),
            "actor_loss": actor_loss,
            "step_time_seconds": total_time,
            "step_metrics": step_metrics,
        }

    # ------------------------------------------------------------------
    # Batch construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_training_batch(
        prompts: list[list[int]],
        responses: list[list[int]],
    ) -> DataProto:
        """Build a DataProto training batch from raw token sequences.

        Constructs input_ids, attention_mask, position_ids, and responses.
        Does NOT set token_level_scores/token_level_rewards/response_mask —
        those are computed by _compute_rm_scores and compute_response_mask
        respectively, matching veRL's native flow.
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

        batch_data = {
            "prompts": prompt_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_tensor,
        }
        non_tensor_data = {
            "data_source": np.array(["gsm8k"] * n, dtype=object),
        }

        return DataProto.from_dict(tensors=batch_data, non_tensors=non_tensor_data)

    @staticmethod
    def _compute_rm_scores(batch: DataProto, rewards: list[float]) -> DataProto:
        """Place reward only on the last valid response token.

        Matches veRL's native ``reward_manager/base.py`` behavior:
        ``rm_scores[..., valid_response_length - 1] = scores``

        This is critical for GRPO: ``compute_grpo_outcome_advantage`` uses
        ``token_level_rewards.sum(dim=-1)`` which must equal the scalar
        reward, not ``reward * response_length``.
        """
        import torch

        n = len(rewards)
        response_mask = batch.batch["responses"] != 0
        max_resp_len = response_mask.size(1)

        # valid_response_length: 1-indexed position of last real token
        valid_response_length = response_mask.sum(dim=-1)

        rm_scores = torch.zeros(n, max_resp_len, dtype=torch.float32)
        for i in range(n):
            vlen = valid_response_length[i].item()
            if vlen > 0:
                rm_scores[i, vlen - 1] = rewards[i]

        batch.batch["token_level_scores"] = rm_scores
        batch.batch["token_level_rewards"] = rm_scores
        return batch


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
        "{question}\nLet's think step by step. Your final line MUST be exactly \"#### X\" where X is only the numerical answer (e.g., \"#### 72\"). Do not add extra text after the number.\n",
    )

    try:
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        if num_prompts < len(dataset):
            dataset = dataset.select(range(num_prompts))
    except Exception as e:
        logger.warning("Failed to load GSM8K dataset: %s", e)
        return []

    from verl.utils.reward_score.gsm8k import extract_solution

    data = []
    for row in dataset:
        question = row["question"]
        answer_text = row["answer"]
        gt = extract_solution(answer_text)

        prompt_text = prompt_template.format(question=question)
        prompt_ids = tokenizer.encode(prompt_text)

        data.append({"prompt_ids": prompt_ids, "ground_truth": gt, "question": question})

    logger.info("Loaded %d GSM8K prompts for training (split=%s)", len(data), split)
    return data


def _load_trajectory_data(config, tokenizer) -> list[dict]:
    """Load training prompts from trajectory training pairs (Phase 3).

    Reads data/training_pairs.jsonl and extracts unique seed prompts.
    Returns a list of dicts with ``prompt_ids`` and ``prompt_text``.
    """
    import json as _json

    traj_config = config.trainer.get("trajectory", {})
    if not traj_config.get("enabled", False):
        logger.info("Trajectory training data loading disabled (trainer.trajectory.enabled=false)")
        return []

    data_path = traj_config.get("data_path", "data/training_pairs.jsonl")
    num_prompts = traj_config.get("num_prompts", 0)

    try:
        pairs = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(_json.loads(line))
    except FileNotFoundError:
        logger.warning("Trajectory data not found: %s", data_path)
        return []

    # Extract unique seed prompts (preserve order)
    seen = set()
    unique_prompts = []
    for p in pairs:
        seed = (p.get("种子提示词", "") or p.get("prompt", "")).strip()
        if seed and seed not in seen:
            seen.add(seed)
            unique_prompts.append({
                "prompt_text": seed,
                "类别": p.get("类别", "") or p.get("category", ""),
            })

    if num_prompts and num_prompts < len(unique_prompts):
        unique_prompts = unique_prompts[:num_prompts]

    data = []
    for item in unique_prompts:
        prompt_ids = tokenizer.encode(item["prompt_text"])
        data.append({
            "prompt_ids": prompt_ids,
            "prompt_text": item["prompt_text"],
            "类别": item.get("类别", ""),
        })

    logger.info("Loaded %d unique seed prompts from %d training pairs (path=%s)",
                len(data), len(pairs), data_path)
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

    # ---- A3: Pre-load training data into unified pool ----
    gsm8k_data = _load_gsm8k_data(config, tokenizer)
    trajectory_data = _load_trajectory_data(config, tokenizer)

    # ---- Hyperparameters ----
    train_steps_per_cycle = config.trainer.get("train_steps_per_cycle", 10)
    prompts_per_step = config.trainer.get("prompts_per_step", 4)
    max_train_rounds = config.trainer.get("max_train_rounds", 10)
    idle_timeout = config.trainer.get("idle_timeout", 30.0)
    min_samples = config.trainer.get("min_samples", 16)
    save_ckpt_interval = config.trainer.get("save_ckpt_interval", 10)
    checkpoint_dir = config.trainer.get("checkpoint_dir", "checkpoints")

    # Unified training pool: each item tracks train_count
    _training_pool: list[dict] = []
    for item in (trajectory_data or []):
        _training_pool.append({
            "prompt_ids": item["prompt_ids"],
            "prompt_text": item.get("prompt_text", ""),
            "ground_truth": None,
            "train_count": 0,
            "source": "trajectory",
        })
    for item in (gsm8k_data or []):
        _training_pool.append({
            "prompt_ids": item["prompt_ids"],
            "prompt_text": item.get("question", ""),
            "ground_truth": item.get("ground_truth"),
            "train_count": 0,
            "source": "gsm8k",
        })

    logger.info("Training pool: %d total (%d trajectory + %d gsm8k), max_rounds=%d",
                len(_training_pool),
                len(trajectory_data or []),
                len(gsm8k_data or []),
                max_train_rounds)

    # Rubric config for trajectory-based training
    traj_config = config.trainer.get("trajectory", {})
    rubrics_path = traj_config.get("rubrics_path", "data/rubrics.json")
    api_key = traj_config.get("api_key", "") or os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = traj_config.get("base_url", "") or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    judge_model = traj_config.get("judge_model", "deepseek-v4-flash")

    # Orchestrator state (no threads)
    _orch_state = {
        "mode": "serving",
        "last_request_time": _time.time(),
        "samples": [],
        "training_in_progress": False,
        "idle_timeout": idle_timeout,
        "min_samples": min_samples,
        "poll_interval": 1.0,
    }
    _app_state["orch_state"] = _orch_state

    async def _record_request_async(prompt_ids, response_ids, metadata=None):
        """Non-blocking sample recording + add to training pool."""
        _orch_state["last_request_time"] = _time.time()
        _orch_state["samples"].append({
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "metadata": metadata or {},
        })
        if len(_orch_state["samples"]) > 10000:
            _orch_state["samples"] = _orch_state["samples"][-5000:]

        # Add to unified training pool as new data
        if metadata and metadata.get("prompt_text"):
            prompt_text = metadata["prompt_text"]
        else:
            try:
                prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            except Exception:
                prompt_text = ""
        _training_pool.append({
            "prompt_ids": list(prompt_ids),
            "prompt_text": prompt_text,
            "ground_truth": None,
            "train_count": 0,
            "source": "api",
        })

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
            # Get fresh prompts: those not yet trained max_rounds times
            fresh_prompts = [p for p in _training_pool if p["train_count"] < max_train_rounds]
            active_count = len(fresh_prompts)
            n_samples = len(_orch_state["samples"])

            if active_count == 0 and n_samples < min_samples:
                continue

            # ---- Trigger training ----
            _orch_state["training_in_progress"] = True
            _orch_state["mode"] = "training"
            t_start = _time.time()

            # Drain API samples
            samples = list(_orch_state["samples"])
            _orch_state["samples"].clear()

            # Active pool for this cycle: fresh prompts + API samples
            active_pool = fresh_prompts if fresh_prompts else samples
            active_source = fresh_prompts[0]["source"] if fresh_prompts else "api"

            logger.info("Training triggered — pool=%d fresh, %d samples, idle=%.1fs, steps=%d, source=%s",
                        active_count, len(samples), elapsed, train_steps_per_cycle, active_source)
            print(f"[TRAINING] Training triggered — {active_count} fresh prompts, {len(samples)} samples, "
                  f"idle={elapsed:.1f}s, steps={train_steps_per_cycle}, max_rounds={max_train_rounds}, "
                  f"source={active_source}",
                  file=sys.stderr, flush=True)

            if not active_pool:
                logger.warning("No training data — skipping")
                _orch_state["training_in_progress"] = False
                _orch_state["mode"] = "serving"
                continue

            all_step_metrics = []
            try:
                rollout_config = info["rollout_config"]
                rollout_n = rollout_config.get("n", 4)
                sampling_params = {
                    "temperature": rollout_config.get("temperature", 1.0),
                    "top_p": rollout_config.get("top_p", 1.0),
                    "max_tokens": rollout_config.get("response_length", 8192),
                }

                for step in range(train_steps_per_cycle):
                    t_step = _time.time()

                    # Re-filter: only fresh prompts for this step
                    step_candidates = [p for p in _training_pool if p["train_count"] < max_train_rounds]
                    if not step_candidates:
                        # Check if API samples arrived during training
                        step_candidates = _orch_state["samples"]
                        if not step_candidates:
                            logger.warning("No fresh prompts + no API samples — skipping remaining steps")
                            break

                    start_idx = (step * prompts_per_step) % max(len(step_candidates), 1)
                    step_items = []
                    step_prompts = []
                    step_ground_truths = []
                    step_prompt_texts = []
                    for k in range(prompts_per_step):
                        idx = (start_idx + k) % max(len(step_candidates), 1)
                        item = step_candidates[idx]
                        step_items.append(item)
                        step_prompts.append(item.get("prompt_ids") if isinstance(item, dict) else item["prompt_ids"])
                        step_ground_truths.append(item.get("ground_truth") if isinstance(item, dict) else item.get("metadata", {}).get("ground_truth"))
                        step_prompt_texts.append(item.get("prompt_text", "") if isinstance(item, dict) else item.get("metadata", {}).get("prompt_text", ""))

                    training_data = {
                        "prompts": step_prompts,
                        "ground_truths": step_ground_truths,
                        "rollout_n": rollout_n,
                        "sampling_params": sampling_params,
                    }

                    # Determine rubric mode from pool items
                    step_source = step_items[0].get("source", "api") if step_items and isinstance(step_items[0], dict) else "api"
                    use_rubric = step_source in ("trajectory", "api") and rubrics_path and api_key

                    if use_rubric:
                        training_data["rubrics_path"] = rubrics_path
                        training_data["api_key"] = api_key
                        training_data["base_url"] = base_url
                        training_data["judge_model"] = judge_model
                        training_data["prompt_texts"] = step_prompt_texts
                        training_data["max_rubrics"] = traj_config.get("max_rubrics", 0)
                        training_data["reward_mode"] = traj_config.get("reward_mode", "mean")
                        _rw = traj_config.get("rubric_weights")
                        if _rw is not None:
                            training_data["rubric_weights"] = _rw

                    # Use asyncio.to_thread to avoid blocking the uvicorn event loop
                    metrics = await asyncio.to_thread(
                        ray.get, runner.train_step.remote(training_data)
                    )

                    step_time = _time.time() - t_step
                    all_step_metrics.append(metrics)

                    # Increment train_count for items in _training_pool
                    trained_count = 0
                    for item in step_items:
                        if isinstance(item, dict) and "train_count" in item:
                            item["train_count"] += 1
                            trained_count += 1

                    sm = metrics.get("step_metrics", {})
                    def _ss(key, default=0.0):
                        """Safely extract a scalar from step_metrics (lists get averaged)."""
                        v = sm.get(key, default)
                        if isinstance(v, list):
                            return sum(v) / len(v) if v else default
                        return v

                    logger.info(
                        "Step %2d/%d done in %5.1fs — reward=%.3f (%d/%d) | "
                        "actor/loss=%.4f grad_norm=%.3f lr=%.2e | "
                        "resp_len=%.0f, score=%.3f rewards=%.3f adv=%.3f | "
                        "tps=%.1f tok/s/gpu, gen=%.2fms/tok actor=%.2fms/tok | "
                        "trained=%d",
                        step + 1, train_steps_per_cycle, step_time,
                        metrics.get("reward_mean", 0),
                        metrics.get("reward_correct", 0),
                        metrics.get("reward_total", 1),
                        _ss("actor/loss", metrics.get("actor_loss", 0)),
                        _ss("actor/grad_norm"),
                        _ss("actor/lr"),
                        _ss("response_length/mean"),
                        _ss("critic/score/mean"),
                        _ss("critic/rewards/mean"),
                        _ss("critic/advantages/mean"),
                        _ss("perf/throughput"),
                        _ss("timing_per_token_ms/gen"),
                        _ss("timing_per_token_ms/update_actor"),
                        trained_count,
                    )
                    print(f"[TRAIN STEP] {step + 1}/{train_steps_per_cycle} done in {step_time:.0f}s — "
                          f"reward={metrics.get('reward_mean', 0):.3f} "
                          f"({metrics.get('reward_correct', 0)}/{metrics.get('reward_total', 1)}) | "
                          f"loss={_ss('actor/loss', metrics.get('actor_loss', 0)):.4f} "
                          f"grad={_ss('actor/grad_norm'):.3f} "
                          f"lr={_ss('actor/lr'):.2e} | "
                          f"rlen={_ss('response_length/mean'):.0f} "
                          f"rew={_ss('critic/rewards/mean'):.3f} "
                          f"adv={_ss('critic/advantages/mean'):.3f} | "
                          f"tps={_ss('perf/throughput'):.1f} "
                          f"gen={_ss('timing_per_token_ms/gen'):.2f}ms/tok "
                          f"actor={_ss('timing_per_token_ms/update_actor'):.2f}ms/tok | "
                          f"trained={trained_count}",
                          file=sys.stderr, flush=True)

                    # ---- Save checkpoint every save_ckpt_interval steps ----
                    global_step = sm.get("training/global_step", step + 1)
                    if save_ckpt_interval > 0 and global_step % save_ckpt_interval == 0:
                        ckpt_path = os.path.join(checkpoint_dir, f"global_step_{global_step}", "actor")
                        print(f"[CHECKPOINT] Saving checkpoint to {ckpt_path} (step {global_step}) ...",
                              file=sys.stderr, flush=True)
                        try:
                            ray.get(runner.save_checkpoint.remote(ckpt_path, global_step, max_ckpt_to_keep=3))
                            print(f"[CHECKPOINT] Saved checkpoint to {ckpt_path}",
                                  file=sys.stderr, flush=True)
                        except Exception:
                            logger.exception("Checkpoint save failed (non-fatal)")

                elapsed_train = _time.time() - t_start
                reward_means = [m.get("reward_mean", 0) for m in all_step_metrics]
                reward_corrects = [m.get("reward_correct", 0) for m in all_step_metrics]
                reward_totals = [m.get("reward_total", 1) for m in all_step_metrics]
                actor_losses = [m.get("actor_loss", 0) for m in all_step_metrics]

                # Aggregate veRL-native step_metrics across steps
                def _avg(key):
                    vals = []
                    for m in all_step_metrics:
                        v = m.get("step_metrics", {}).get(key, 0)
                        if isinstance(v, list):
                            v = sum(v) / len(v) if v else 0
                        vals.append(v)
                    return sum(vals) / len(vals) if vals else 0

                # Pool stats after cycle
                fresh_after = sum(1 for p in _training_pool if p["train_count"] < max_train_rounds)
                total_pool = len(_training_pool)
                logger.info(
                    "Training complete in %.1fs — %d steps, pool=%d/%d fresh | "
                    "reward %.3f→%.3f | loss %.4f→%.4f | "
                    "tps=%.1f tok/s/gpu, thrpt=%.0f tok, "
                    "rlen=%.0f, score=%.3f, adv=%.3f",
                    elapsed_train, len(all_step_metrics),
                    fresh_after, total_pool,
                    reward_means[0] if reward_means else 0,
                    reward_means[-1] if reward_means else 0,
                    actor_losses[0] if actor_losses else 0,
                    actor_losses[-1] if actor_losses else 0,
                    _avg("perf/throughput"),
                    _avg("perf/total_num_tokens"),
                    _avg("response_length/mean"),
                    _avg("critic/score/mean"),
                    _avg("critic/advantages/mean"),
                )
                print(
                    f"[TRAINING] Complete in {elapsed_train:.1f}s — {len(all_step_metrics)} steps, "
                    f"pool={fresh_after}/{total_pool} fresh (max_rounds={max_train_rounds}) | "
                    f"reward {reward_means[0]:.3f}→{reward_means[-1]:.3f} | "
                    f"loss {actor_losses[0]:.4f}→{actor_losses[-1]:.4f} | "
                    f"tps={_avg('perf/throughput'):.1f} tok/s/gpu, rlen={_avg('response_length/mean'):.0f}, "
                    f"score={_avg('critic/score/mean'):.3f}, adv={_avg('critic/advantages/mean'):.3f}",
                    file=sys.stderr, flush=True,
                )

            except Exception:
                logger.exception("Training failed — resuming serving with old weights")
            finally:
                _orch_state["mode"] = "serving"
                _orch_state["training_in_progress"] = False
                _orch_state["last_request_time"] = _time.time()

    _app_state["record_request"] = _record_request_async
    _app_state["orch_state"] = _orch_state
    _app_state["_monitor_coro"] = _async_monitor_loop
    _app_state["vllm_server_address"] = info["server_addresses"][0] if info.get("server_addresses") else None
    _app_state["model_path"] = info.get("model_path", "")

    # ---- Conversation log store (Phase 2 — B1 analysis) ----
    from trainable_openclaw.logging.conversation_store import ConversationStore
    os.makedirs("data", exist_ok=True)
    _app_state["conversation_store"] = ConversationStore("data/conversations.db")

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
