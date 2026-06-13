# Trainable OpenClaw — 项目设计文档

## 架构总览

```
┌─────────────────────────────────────────────────────┐
│  Driver 进程 (CPU)                                   │
│  ┌───────────────────────────────────────────────┐  │
│  │  FastAPI / uvicorn                            │  │
│  │  - /v1/chat/completions  (外部推理 API)       │  │
│  │  - /v1/health             (健康检查)          │  │
│  │                                               │  │
│  │  Async Monitor (uvicorn event loop)           │  │
│  │  - 空闲检测 (idle_timeout)                    │  │
│  │  - 训练触发 → runner.train_step.remote()      │  │
│  │  - 训练期间 → API 返回 503                    │  │
│  └───────────────────────────────────────────────┘  │
│                         │                            │
│                         │ Ray RPC                    │
│                         ▼                            │
│  ┌───────────────────────────────────────────────┐  │
│  │  ServeRunner (Ray Actor, GPU)                 │  │
│  │                                               │  │
│  │  run():                                       │  │
│  │    初始化 FSDP actor + vLLM rollout           │  │
│  │    创建 LLMServerManager + CheckpointEngine   │  │
│  │                                               │  │
│  │  train_step(prompts, ground_truths):          │  │
│  │    1. 生成回复 (vLLM rollout)                 │  │
│  │    2. 计算奖励                                 │  │
│  │    3. 构建 DataProto                          │  │
│  │    4. compute_log_prob (FSDP)                 │  │
│  │    5. compute_advantage (GRPO)                │  │
│  │    6. update_actor (FSDP)                     │  │
│  │    7. update_weights (FSDP → vLLM)            │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## 核心设计原则

1. **基于 veRL 原生训练流程** — 不修改 `ray_trainer.py` 等核心训练代码，复用
   `ActorRolloutRefWorker` 的 `compute_log_prob` / `update_actor` / `update_weights` 等能力

2. **Rollout 独立化为推理服务** — vLLM engine 同时承担两个角色：
   - **Serving 模式**: 对外提供 `/v1/chat/completions` API（外部用户请求）
   - **Training 模式**: 作为 GRPO 的 rollout engine（生成训练所需的 N 个答案）

3. **生成发生在 Ray Actor 内部** — `train_step()` 在 ServeRunner (Ray Actor) 中执行，
   生成回复直接调用同进程内的 vLLM server，避免 driver → Ray RPC 的 event loop 冲突

4. **训练数据注入** — 训练提示词可来自两个来源：
   - GSM8K 等外部数据集（预加载，绕过 min_samples 门槛）
   - 用户 API 请求积累的样本（通过 `record_request` 收集）

## 关键改造点

### 改造 1: 生成回调移至 Ray Actor 内部

**现状问题**: 之前的实现中，生成回复（`llm_client.generate()`）在 driver 的 uvicorn
event loop 中执行。无论用 `await`、`asyncio.run()` 还是 `ray.get()`，Ray actor 方法
调用在 driver 的异步上下文中都会挂起。

**解决方案**: `train_step()` 在 ServeRunner (Ray Actor) 内部完成生成。ServeRunner
已有 `llm_server_manager`（管理 vLLM replicas）和对应的 server addresses。生成时直接
通过 HTTP 调用 vLLM 的 `/v1/chat/completions` 端点（vLLM HTTP API 已验证可用），
或通过 Ray actor 内部调用 `server.generate.remote()`（actor 内部调用不应有 event loop
冲突）。

```
train_step(prompts, ground_truths):
    # ---- 都在 Ray Actor 内部执行 ----
    1. sleep_replicas()   # 可选：释放 GPU 显存给训练
    2. 生成 N 个回复      # 调用同进程 vLLM server
    3. 计算奖励           # GSM8K 答案匹配等
    4. build_training_batch()
    5. compute_log_prob()
    6. compute_advantage()
    7. update_actor()
    8. update_weights()   # FSDP → vLLM
    9. wake_replicas()    # 恢复推理服务
```

### 改造 2: 训练数据注入机制

**现状**: 训练提示词来自 `_load_gsm8k_data()` 预加载，或 API request 积累的样本。

**需要改造**: 
- API handler 中调用 `record_request(prompt_ids, response_ids)` 记录样本
- `train_step(prompts, ground_truths)` 接收 driver 传入的提示词和 ground truth
- Driver 在触发训练前收集样本，调用 `runner.train_step.remote(training_data)`
- 训练完成后，driver 更新 `_orch_state`，恢复 serving 模式

### 改造 3: 生命周期管理

```
SERVING ──(idle + 样本够)──> TRAINING ──(train_step 完成)──> SERVING
   │                              │
   │  API 正常响应                 │  API 返回 503
   │  record_request() 收集样本    │  "Training in progress"
```

状态机在 driver 端管理（`_orch_state`），训练执行在 Ray actor 端。

## 数据流

### Serving 模式 (正常推理)

```
用户 → POST /v1/chat/completions
     → FastAPI handler
     → await llm_client.generate()  [Ray RPC → vLLM replica]
     → 返回 ChatCompletionResponse
     → record_request(prompt_ids, response_ids)  [积累训练样本]
```

### Training 模式 (GRPO 训练)

```
Driver (async monitor):
  1. 检测空闲 + 样本达标
  2. 收集样本: samples = drain_buffer()
  3. ray.get(runner.train_step.remote({
       prompts: list[list[int]],
       ground_truths: list[float],
       rollout_n: int,
       sampling_params: dict,
     }))

ServeRunner.train_step() [Ray Actor, GPU]:
  1. 生成回复:
     for prompt_ids in prompts:
         response_text = http_post(vllm_addr, prompt_text)
         response_ids = tokenizer.encode(response_text)
  2. 计算奖励:
     reward = 1.0 if extract_answer(response_text) == ground_truth else 0.0
  3. 构建 DataProto:
     batch = _build_training_batch(prompts, responses, rewards)
  4. 计算 old_log_probs:
     old_log_probs = actor_rollout_wg.compute_log_prob(batch)
  5. 计算 GRPO advantages:
     batch = compute_advantage(batch, adv_estimator=GRPO)
  6. 更新 actor:
     actor_output = actor_rollout_wg.update_actor(batch)
  7. 同步权重:
     checkpoint_manager.update_weights(global_steps)
  8. 返回训练指标
```

## veRL 训练流程复用

完全复用 veRL 的以下能力，不修改其源码：

| 能力 | 调用路径 | 说明 |
|------|---------|------|
| `compute_log_prob` | `actor_rollout_wg.compute_log_prob(batch)` | FSDP 前向，计算 old_log_probs |
| `update_actor` | `actor_rollout_wg.update_actor(batch)` | FSDP 前向/反向/优化器步骤 |
| `update_weights` | `checkpoint_manager.update_weights(steps)` | FSDP → vLLM 权重同步 |
| `sleep_replicas` | `checkpoint_manager.sleep_replicas()` | 暂停 vLLM，释放 GPU 显存 |
| `wake_up_replicas` | `checkpoint_manager.wake_up_replicas()` | 恢复 vLLM（仅非 HYBRID 模式） |
| `compute_advantage` | `verl.trainer.ppo.ray_trainer.compute_advantage()` | GRPO advantage 计算 |

## 不需要修改的部分

- `verl/trainer/ppo/ray_trainer.py` — 不修改
- `verl/workers/engine_workers.py` — 不修改
- `verl/workers/rollout/` — 不修改
- `verl/checkpoint_engine/` — 不修改
- `verl/workers/utils/losses.py` — 不修改

## 需要修改的部分

1. `verl/trainer/serve_ppo.py` — `ServeRunner.train_step()`:
   - 接收训练数据（prompts + ground_truths）
   - 内部完成生成回复（调用 vLLM HTTP API）
   - 调用 veRL 训练流程
   - 返回训练指标

2. `trainable_openclaw/server/api.py` — API handler:
   - `record_request` 改为 async 回调
   - training 状态检查简化

3. `trainable_openclaw/training/orchestrator.py` — 可选简化:
   - 如果用 async monitor 替代线程 monitor，可以删除或简化

## 配置示例

```bash
python3 -m verl.trainer.serve_ppo \
  actor_rollout_ref.model.path=/data/models/Qwen3-4B \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.load_format=auto \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  +trainer.serve_port=8000 \
  +trainer.idle_timeout=15 +trainer.min_samples=1 \
  +trainer.gsm8k.enabled=true +trainer.gsm8k.num_prompts=4 \
  actor_rollout_ref.model.lora_rank=16 actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  ++actor_rollout_ref.rollout.enable_sleep_mode=false
```

## 当前状态 (2026/05/24)

- ✅ vLLM 推理服务正常运行
- ✅ LoRA 训练显存问题解决 (16.5 GiB training + 14.6 GiB vLLM)
- ✅ `train_step()` 训练链搭建完成 (compute_log_prob → advantage → update_actor → sync)
- ❌ 训练触发后生成回复挂起 — 根因定位为 driver event loop 中调用 Ray actor 方法死锁
- 🔧 待实现: 生成回复移至 ServeRunner actor 内部执行
