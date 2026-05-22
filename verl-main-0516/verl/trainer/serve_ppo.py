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

import hydra
import ray
import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

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
        import asyncio

        async def _sleep():
            replicas = self.llm_server_manager.get_replicas()
            await asyncio.gather(*[r.sleep() for r in replicas])

        asyncio.run(_sleep())
        logger.info("All rollout replicas asleep")

    def wake_replicas(self):
        """Wake all rollout replicas after training to resume inference."""
        import asyncio

        async def _wake():
            replicas = self.llm_server_manager.get_replicas()
            await asyncio.gather(*[r.wake_up() for r in replicas])

        asyncio.run(_wake())
        logger.info("All rollout replicas awake")

    def train_step(self, samples: list[dict]) -> None:
        """Execute one training step.

        Currently a stub that exercises the sleep→train→wake cycle.
        Real implementation will invoke veRL's RayPPOTrainer.fit() for
        a single step, then merge/update weights before waking replicas.
        """
        import time

        logger.info(f"train_step called with {len(samples)} samples (stub)")
        # Simulate training time so the "training in progress" window is
        # observable for A2 integration tests.
        time.sleep(3.0)
        logger.info("train_step completed (stub)")
        # TODO: integrate RayPPOTrainer single-step training here
        # 1. Build DataProto from samples
        # 2. Compute old_log_probs with rollout engine (wake, forward, sleep)
        # 3. Run one PPO update step
        # 4. Merge LoRA / sync weights via CheckpointEngineManager


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


def run_serve(config) -> None:
    """Initialize Ray, create ServeRunner actor, and start FastAPI on the driver."""
    from verl.utils.device import is_cuda_available

    # In serve-only mode, load_format must NOT be "dummy" — there is no
    # training engine to sync real weights, so vLLM would generate garbage.
    load_format = OmegaConf.select(config, "actor_rollout_ref.rollout.load_format")
    if load_format is None or load_format == "dummy":
        config.actor_rollout_ref.rollout.load_format = "auto"
        print(f"[serve_ppo] load_format={load_format or 'default'} → auto (serve-only mode)")

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

    def _train_bridge(samples):
        """Bridge: orchestrator runs in thread, training runs on Ray actor."""
        batch = [
            {"prompt_ids": s.prompt_ids, "response_ids": s.response_ids, "metadata": s.metadata}
            for s in samples
        ]
        # NOTE: sleep/wake skipped for stub training — vLLM V1 sleep/wake can
        # trigger CUDA illegal memory access on some hardware.  A3 will
        # re-enable this when real training (LoRA merge + weight sync) is
        # implemented, since the weight sync path handles engine state properly.
        # ray.get(runner.sleep_replicas.remote())
        try:
            ray.get(runner.train_step.remote(batch))
        finally:
            # ray.get(runner.wake_replicas.remote())
            pass

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
