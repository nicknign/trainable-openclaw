# 代码说明文档 — trainable-openclaw Phase 1

> 覆盖 A1（推理 API）+ A2（空闲检测 + 训练触发）+ A3（权重同步 + GRPO 训练）
> 最后更新: 2026-05-25

---

## 目录

1. [整体架构](#1-整体架构)
2. [流程图](#2-流程图)
3. [serve_ppo.py — 核心引擎](#3-serve_ppopy--核心引擎)
4. [api.py — FastAPI 推理接口](#4-apipy--fastapi-推理接口)
5. [orchestrator.py — 训练编排器](#5-orchestratorpy--训练编排器)
6. [启动脚本](#6-启动脚本)
7. [关键设计决策](#7-关键设计决策)

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                     Driver Process                       │
│                                                          │
│  ┌──────────────────┐    ┌────────────────────────────┐ │
│  │   FastAPI (uvicorn)│    │  Async Monitor Loop        │ │
│  │                    │    │  (uvicorn event loop)      │ │
│  │  /v1/health        │    │                            │ │
│  │  /v1/chat/         │    │  每1s轮询:                 │ │
│  │    completions     │    │  空闲? + 样本够?           │ │
│  │                    │    │    → 触发训练              │ │
│  └────────┬───────────┘    └─────────┬──────────────────┘ │
│           │                          │                     │
│           │ record_request()         │ ray.get(            │
│           │                          │   runner.train_step │
│           ▼                          │   .remote(...))     │
│  ┌────────────────────┐             │                     │
│  │  _app_state (共享)  │             │                     │
│  │  - llm_client       │             │                     │
│  │  - tokenizer        │             │                     │
│  │  - orch_state       │◄────────────┘                     │
│  └────────────────────┘                                    │
│           │                                                │
└───────────┼────────────────────────────────────────────────┘
            │ Ray RPC
            ▼
┌─────────────────────────────────────────────────────────┐
│                   Ray Actor: ServeRunner                  │
│                                                          │
│  ┌─────────────────────┐  ┌───────────────────────────┐ │
│  │ ActorRolloutRefWorker│  │ LLMServerManager          │ │
│  │ (FSDP training)      │  │ (vLLM inference engines)  │ │
│  │                      │  │                           │ │
│  │ - compute_log_prob() │  │ - HTTP :8000/v1/chat/...  │ │
│  │ - update_actor()     │  │ - generate() for training │ │
│  └──────────┬───────────┘  └──────────┬────────────────┘ │
│             │                          │                  │
│             │  CheckpointEngineManager │                  │
│             │  - sleep_replicas()      │                  │
│             │  - update_weights()      │                  │
│             │  - wake_up_replicas()    │                  │
│             └──────────┬───────────────┘                  │
│                        │                                  │
│               train_step()                                │
│               (GRPO training loop)                        │
└─────────────────────────────────────────────────────────┘
```

**进程模型**:
- **Driver 进程**: 运行 FastAPI (uvicorn) + 异步监控循环
- **Ray Actor 进程**: ServeRunner 持有 veRL worker (FSDP) + vLLM engines
- 两者通过 Ray RPC 通信 (`ray.get(runner.train_step.remote(...))`)

---

## 2. 流程图

### 2.1 启动流程

```
main()
  │
  ├─ auto_set_device()           # 自动检测 GPU
  ├─ migrate_legacy_reward_impl() # 兼容旧版 reward 配置
  │
  └─ run_serve()
       │
       ├─ 1. 修正配置
       │    load_format=dummy → auto         (推理必须加载真实权重)
       │    checkpoint_engine.backend → naive (不需要 NCCL)
       │
       ├─ 2. ray.init()                     # 初始化 Ray 集群
       │
       ├─ 3. ServeRunner.run() [Ray Actor]
       │    │
       │    ├─ add_actor_rollout_worker()      # 注册 worker 类
       │    ├─ hf_tokenizer()                  # 加载 tokenizer
       │    ├─ init_resource_pool_mgr()        # 创建 GPU 资源池
       │    ├─ create_colocated_worker_cls()   # 组合 actor+rollout worker
       │    ├─ actor_rollout_wg.init_model()   # 初始化 FSDP + vLLM
       │    ├─ LLMServerManager.create()       # 启动 vLLM 推理副本
       │    ├─ CheckpointEngineManager()       # sleep/wake/sync 管理器
       │    ├─ Tracking()                      # veRL 指标日志 (console+tensorboard)
       │    └─ return {tokenizer_path, rollout_config, ...}
       │
       ├─ 4. Driver 加载 tokenizer
       ├─ 5. 获取 llm_client (Ray Actor 引用)
       ├─ 6. 注入 _app_state:
       │    llm_client, tokenizer, rollout_config, runner ...
       │
       ├─ 7. _load_gsm8k_data()              # 加载 GSM8K 数据集 (如果启用)
       │
       ├─ 8. 设置异步编排器:
       │    _orch_state = {mode, samples, idle_timeout, ...}
       │    _record_request_async()           # 样本记录回调
       │    _async_monitor_loop()             # 空闲检测协程
       │
       ├─ 9. create_app()                    # 创建 FastAPI app
       │
       └─ 10. uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2.2 推理请求流程 (SERVING 模式)

```
POST /v1/chat/completions
  │
  ├─ 1. 检查训练状态
  │    training_in_progress? → 503 "Training in progress"
  │
  ├─ 2. tokenizer.apply_chat_template(messages)
  │    构造 Qwen3 chat template (含 thinking block)
  │
  ├─ 3. tokenizer.encode(prompt_text)
  │
  ├─ 4. llm_client.generate(request_id, prompt_ids, sampling_params)
  │    │
  │    └─ Ray RPC → vLLM engine (HTTP /v1/chat/completions)
  │         返回: token_ids, stop_reason
  │
  ├─ 5. record_request(prompt_ids, response_ids) [A2]
  │    │
  │    ├─ 更新 last_request_time (重置空闲计时器)
  │    └─ 样本入队 samples.append({prompt_ids, response_ids, metadata})
  │
  ├─ 6. tokenizer.decode(output.token_ids)
  │
  └─ 7. 返回 ChatCompletionResponse
       {id, choices: [{message: {role: "assistant", content: ...}}], usage: {...}}
```

### 2.3 训练触发与执行流程 (TRAINING 模式)

```
_async_monitor_loop() [每1秒轮询]
  │
  ├─ 条件检查:
  │   ├─ training_in_progress? → skip (跳过本次)
  │   ├─ elapsed < idle_timeout? → skip (还没空闲够)
  │   ├─ samples < min_samples AND no GSM8K? → skip (样本不够)
  │   └─ 全部满足 → 触发训练
  │
  ├─ 设置 training_in_progress=True, mode="training"
  ├─ 排空样本队列: samples = list(_orch_state["samples"])
  │
  └─ for step in range(train_steps_per_cycle):
       │
       │  [Round-robin 选 prompt]
       │  start_idx = (step * prompts_per_step) % len(pool)
       │
       │  await asyncio.to_thread(
       │      ray.get, runner.train_step.remote(training_data)
       │  )
       │  │
       │  │  ┌────────────────────────────────────────┐
       │  │  │  train_step() [Ray Actor 内部]          │
       │  │  │                                        │
       │  │  │  1. Generate (并发 HTTP → vLLM)        │
       │  │  │     ThreadPoolExecutor(max_workers=16)  │
       │  │  │     n_prompts × rollout_n 个并发请求    │
       │  │  │                                        │
       │  │  │  2. Compute Rewards                     │
       │  │  │     gsm8k.compute_score(response, gt)  │
       │  │  │                                        │
       │  │  │  3. Sleep Replicas                      │
       │  │  │     checkpoint_manager.sleep_replicas() │
       │  │  │     (释放 vLLM GPU 内存给训练用)        │
       │  │  │                                        │
       │  │  │  4. Build DataProto Batch               │
       │  │  │     _build_training_batch()             │
       │  │  │     (prompts + responses → tensor)      │
       │  │  │                                        │
       │  │  │  5. Compute RM Scores                   │
       │  │  │     _compute_rm_scores()                │
       │  │  │     (reward → 仅最后 token 放置)        │
       │  │  │                                        │
       │  │  │  6. Compute Response Mask               │
       │  │  │     compute_response_mask(batch)        │
       │  │  │     (标记有效 response token)            │
       │  │  │                                        │
       │  │  │  7. Compute Old Log Probs               │
       │  │  │     actor_rollout_wg.compute_log_prob() │
       │  │  │     (FSDP 前向 → no_padding_2_padding)  │
       │  │  │                                        │
       │  │  │  8. Compute GRPO Advantages             │
       │  │  │     compute_advantage(GRPO)             │
       │  │  │     (组内归一化 advantage)              │
       │  │  │                                        │
       │  │  │  9. Update Actor                        │
       │  │  │     actor_rollout_wg.update_actor()     │
       │  │  │     (FSDP 前向+反向+优化器 step)        │
       │  │  │                                        │
       │  │  │  10. Update Weights                     │
       │  │  │      checkpoint_manager.update_weights()│
       │  │  │      (actor→rollout, naive IPC)         │
       │  │  │                                        │
       │  │  │  11. Compute Metrics + Tracker.log()    │
       │  │  │      data_metrics, timing_metrics,      │
       │  │  │      throughput_metrics                 │
       │  │  └────────────────────────────────────────┘
       │  │
       │  └─ 返回 {reward_mean, reward_correct, actor_loss, step_time}
       │
       └─ 打印 step 进度: "Step X/Y done — reward=... loss=..."
       │
  ├─ 训练成功:
  │   └─ 如果用了 GSM8K 数据 & 无 API 样本 → _gsm8k_exhausted=True
  │
  ├─ 训练失败:
  │   └─ logger.exception() — 不标记 exhausted，下次还可重试
  │
  └─ finally:
       mode="serving"
       training_in_progress=False
       last_request_time=now (重置空闲计时器)
```

### 2.4 状态机

```
                    ┌─────────────────┐
                    │    SERVING       │
                    │  (推理服务正常)   │
                    └────────┬────────┘
                             │
                  空闲超时 + 样本够
                  (或 GSM8K 可用)
                             │
                             ▼
                    ┌─────────────────┐
                    │   TRAINING       │
                    │  (返回 503)      │
                    │                 │
                    │  train_step()   │
                    │  × N steps      │
                    └────────┬────────┘
                             │
                    训练完成 / 训练失败
                             │
                             ▼
                    ┌─────────────────┐
                    │    SERVING       │
                    │  (恢复推理服务)   │
                    │  (失败时保留旧权重)│
                    └─────────────────┘
```

关键保护:
- 训练失败 → **不**标记 GSM8K 耗尽 → 下次触发可重试
- 训练失败 → **不**同步权重 → 旧模型继续推理
- 训练期间请求 → **返回 503** 而非挂死 (asyncio.to_thread 保活 event loop)

---

## 3. serve_ppo.py — 核心引擎

**文件**: `verl-main-0516/verl/trainer/serve_ppo.py`
**定位**: 整个系统的核心文件，实现 veRL 双模引擎的完整生命周期

### 3.1 `main(config)` (line 604)

Hydra 入口函数，由 `python -m verl.trainer.serve_ppo` 调用。

```
流程:
  1. auto_set_device(config)      — 自动检测 CUDA 设备
  2. migrate_legacy_reward_impl() — 兼容旧版 reward manager 配置
  3. run_serve(config)            — 启动推理服务
```

### 3.2 `class ServeRunner(TaskRunner)` (line 72)

**继承关系**: 继承 veRL 的 `TaskRunner`，复用其 `add_actor_rollout_worker()`、`init_resource_pool_mgr()` 等方法。

**为什么是 Ray Actor**: 通过 `ray.remote(num_cpus=1)(ServeRunner).remote()` 创建，在独立进程中运行，持有 GPU 资源（FSDP 模型 + vLLM 引擎）。

#### 3.2.1 `run(self, config) -> dict` (line 79)

**功能**: 初始化整个 veRL 服务引擎。在 Ray Actor 进程中执行。

**Step-by-step**:

| Step | 操作 | 说明 |
|------|------|------|
| 1 | `add_actor_rollout_worker(config)` | 注册 ActorRolloutRefWorker 类到 role_worker_mapping |
| 2 | `hf_tokenizer(local_path)` | 加载 HuggingFace tokenizer，支持 trust_remote_code |
| 3 | `init_resource_pool_mgr(config)` | 创建 GPU 资源池 (Ray placement groups) |
| 4 | `create_colocated_worker_cls()` | 将 actor worker 和 rollout worker 捆绑到同一 GPU |
| 5 | `wg.init_model()` | 初始化 FSDP 模型 + vLLM 推理引擎 (HYBRID 模式) |
| 5a | `LLMServerManager.create()` | 管理 vLLM 推理副本的地址和客户端 |
| 5b | `CheckpointEngineManager()` | 管理 sleep/wake/weight_sync 三个关键操作 |
| 6 | `Tracking()` | 初始化 veRL 原生日志系统 (console + tensorboard) |

**返回值**: 包含 tokenizer_path、rollout_config、gpu_count、server_addresses 等信息，供 driver 进程使用。

#### 3.2.2 `get_llm_client(self, _config=None)` (line 205)

返回 vLLM 客户端对象给 driver 进程。通过 Ray 序列化传递 ActorHandle。

#### 3.2.3 `sleep_replicas(self)` / `wake_replicas(self)` (lines 218-226)

委托给 `checkpoint_manager` 执行：
- `sleep_replicas()`: 释放 vLLM 推理引擎的 GPU 内存，腾出空间给 FSDP 训练
- `wake_replicas()`: 恢复 vLLM 推理引擎（重新分配 GPU 缓存）

`checkpoint_manager.update_weights()` 内部已包含 wake 逻辑，所以通常不需要显式调用 `wake_replicas()`。

#### 3.2.4 `train_step(self, training_data: dict) -> dict` (line 228)

**功能**: 执行一个完整的 GRPO 训练步骤。**这是整个系统最核心的函数**。

**输入** `training_data`:
```python
{
    "prompts": list[list[int]],       # N 个 prompt 的 token ID 列表
    "ground_truths": list[float|None], # 每个 prompt 的正确答案 (GSM8K)
    "rollout_n": int,                  # 每个 prompt 生成几个回答 (默认 4)
    "sampling_params": dict,           # temperature, top_p, max_tokens
}
```

**11 步流程详解**:

| # | 阶段 | 代码位置 | 说明 |
|---|------|---------|------|
| 1 | **生成回答** | lines 272-327 | 通过 HTTP 并发调用 vLLM 的 `/v1/chat/completions`，`ThreadPoolExecutor(max_workers=16)` 并发。每个 prompt 生成 `rollout_n` 个回答，超时 300s |
| 2 | **计算奖励** | lines 329-347 | 使用 veRL 原生 `gsm8k.compute_score(response, gt)` 对每个回答打分（0=错误, 1=正确），不再用自定义的 `_extract_gsm8k_answer()` |
| 3 | **休眠推理引擎** | lines 349-353 | `checkpoint_manager.sleep_replicas()` 释放 vLLM GPU 内存 |
| 4 | **构建训练批次** | lines 355-360 | `_build_training_batch()` 将 prompt+response token 列表转为 DataProto 张量（含 input_ids, attention_mask, position_ids）。**注意**: 这里不设置 reward, response_mask — 由后续步骤完成 |
| 5 | **放置 reward** | lines 362-365 | `_compute_rm_scores()` 将每个回答的 scalar reward 放到其**最后一个有效 token** 位置。这是与 veRL 原生对齐的关键 |
| 6 | **计算 response_mask** | lines 367-372 | `compute_response_mask(batch)` 标记哪些 token 属于 response（非 padding） |
| 7 | **计算 old_log_probs** | lines 374-391 | Actor (FSDP) 前向传播，计算当前策略下每个 token 的对数概率。`left_right_2_no_padding` 去掉 padding 以节省 FLOPs |
| 8 | **计算 GRPO Advantage** | lines 393-410 | `compute_advantage(GRPO)` 按 prompt 分组归一化 advantage。`compute_grpo_outcome_advantage` 对 `token_level_rewards.sum(dim=-1)` 做组内标准化 |
| 9 | **更新 Actor** | lines 412-447 | `actor_rollout_wg.update_actor()` FSDP 前向+反向+优化器 step。mini-batch 等参数从 `actor_config` 读取而非硬编码 |
| 10 | **同步权重** | lines 449-455 | `checkpoint_manager.update_weights()` 通过 naive IPC 将更新后的 LoRA 权重从 FSDP 同步到 vLLM。内部自动 wake replicas |
| 11 | **计算指标 + 日志** | lines 457-487 | 计算 data_metrics/timing_metrics/throughput_metrics，通过 `tracker.log()` 写入 tensorboard |

**返回值**:
```python
{
    "reward_mean": float,      # 平均奖励
    "reward_correct": int,     # 正确回答数
    "reward_total": int,       # 总回答数
    "actor_loss": float,       # PPO loss
    "step_time_seconds": float # 总耗时
}
```

**关键设计决策 — 为什么生成在 Ray Actor 内完成?**
最初生成在 driver 进程通过 `asyncio.run()` 调用，但 uvicorn event loop 会被 `ray.get()` 阻塞（即使放到线程也死锁）。改为在 Ray Actor 内用 `ThreadPoolExecutor` + `urllib.request.urlopen()` 并发 HTTP 调用 vLLM — 完全脱离 uvicorn event loop，避免死锁。driver 层用 `asyncio.to_thread(ray.get, ...)` 调用，确保 503 能正常返回。

#### 3.2.5 `_build_training_batch(prompts, responses) -> DataProto` (line 501, static)

**功能**: 将原始 token 序列转换为 veRL 的 DataProto 训练批次。

**详细步骤**:
1. **Padding**: Prompt 用 left-padding（0 在左边），Response 用 right-padding（0 在右边）
2. **构造张量**:
   - `input_ids`: `[prompt_tokens | response_tokens]`，padding 部分填 0
   - `attention_mask`: padding 位置=0，有效 token=1
   - `position_ids`: prompt 部分从 0 累计，response 部分从 prompt 最后位置继续累计
   - `responses`: response token 张量（用于 compute_response_mask）
3. **返回 DataProto**: veRL 的标准数据容器，包含 `batch` (tensor dict) 和 `non_tensor_batch` (numpy/metadata dict)

**重要**: 此函数只构建**结构**（input_ids, attention_mask, position_ids），不设置 reward 或 response_mask。那些由后续的 `_compute_rm_scores()` 和 `compute_response_mask()` 补全 — 这与 veRL 原生流程完全对齐。

#### 3.2.6 `_compute_rm_scores(batch, rewards) -> DataProto` (line 567, static)

**功能**: 将 scalar reward 放置到每个 response 的**最后一个有效 token** 位置。

**为什么这是关键修复 (之前 loss=0 的根因)**:

```
错误方式 (旧代码):  正确方式 (veRL 原生):
token 0:  reward     token 0:  0
token 1:  reward     token 1:  0
token 2:  reward     token 2:  0
...                  ...
token N-1: reward    token N-1: reward  ← 仅最后 token
token N:   0         token N:   0

GRPO sum = reward × N   GRPO sum = reward ✓
```

`compute_grpo_outcome_advantage` 用 `token_level_rewards.sum(dim=-1)` 计算每个回答的 scalar score。如果每个 token 都放 reward，分数 = reward × 回答长度 → 长度噪音淹没正确性信号 → advantage 无意义 → loss=0。

### 3.3 `_load_gsm8k_data(config, tokenizer) -> list[dict]` (line 617)

**功能**: 从 HuggingFace datasets 加载 GSM8K 训练集。

**配置** (通过 Hydra CLI 传入):
```bash
+trainer.gsm8k.enabled=true       # 启用 GSM8K 数据加载
+trainer.gsm8k.num_prompts=320    # 加载多少条
+trainer.gsm8k.split=train        # 数据分片
```

**Prompt Template** (关键 — 告知模型输出格式):
```
{question}
Let's think step by step. Your final line MUST be exactly "#### X" where X
is only the numerical answer (e.g., "#### 72"). Do not add extra text after
the number.
```

这个 template 经过多次迭代才确定。早期版本只说 "output after ####"，模型经常输出 `####\n72`（换行分隔），导致 veRL 的 `extract_solution(method="strict")` 正则 `"#### (\\-?[0-9\\.\\,]+)"` 匹配失败。

**Ground Truth 提取**: 使用 veRL 原生 `extract_solution(answer_text)` (strict 模式) 从 GSM8K 的 `"#### 42"` 格式提取答案。

**返回格式**:
```python
[
    {"prompt_ids": [1, 2, 3, ...], "ground_truth": 42.0, "question": "..."},
    ...
]
```

### 3.4 `run_serve(config)` (line 667)

**功能**: Driver 进程的主函数。初始化 Ray、创建 ServeRunner Actor、注入所有状态到 `_app_state`、启动 FastAPI。

**9 个步骤**:

| # | 操作 | 说明 |
|---|------|------|
| 1 | 修正 `load_format` | 如果是 `dummy` → `auto`（推理必须加载真实权重） |
| 2 | 修正 `checkpoint_engine.backend` | 如果不是 `naive` → `naive`（不需要 NCCL） |
| 3 | `ray.init()` | 初始化 Ray 集群 |
| 4 | 创建 `ServeRunner` Actor + 调用 `run()` | `ray.remote(num_cpus=1)(ServeRunner).remote()` |
| 5 | 加载 tokenizer (driver 端) | 供 API 端点使用（tokenize/decode） |
| 6 | 获取 `llm_client` | 通过 `ray.get(runner.get_llm_client.remote())` |
| 7 | 注入 `_app_state` | llm_client, tokenizer, rollout_config, runner |
| 8 | 加载 GSM8K 数据 | `_load_gsm8k_data()` (如果启用) |
| 9 | 设置异步编排器 | `_orch_state` + `_record_request_async()` + `_async_monitor_loop()` |
| 10 | 创建 FastAPI app + 启动 uvicorn | `create_app()` → `uvicorn.run(app, host="0.0.0.0", port=8000)` |

### 3.5 异步训练编排器 (内嵌于 `run_serve`)

#### 3.5.1 `_orch_state` (line 733)

```python
_orch_state = {
    "mode": "serving",              # 当前模式: "serving" | "training"
    "last_request_time": time(),    # 最后一次请求的时间戳
    "samples": [],                  # 累积的训练样本列表
    "training_in_progress": False,  # 是否正在训练
    "idle_timeout": 30.0,           # 空闲超时 (秒)
    "min_samples": 16,              # 最小样本数阈值
    "poll_interval": 1.0,           # 轮询间隔 (秒)
}
```

#### 3.5.2 `_record_request_async(prompt_ids, response_ids, metadata)` (line 744)

API 端点每次完成生成后调用。非阻塞 — 直接在 uvicorn event loop 上执行（不涉及线程）。

操作:
1. 更新 `last_request_time` = 当前时间（重置空闲计时）
2. 样本入队 `samples.append({...})`
3. 缓冲上限 10000 条，超出时丢弃最旧的一半

#### 3.5.3 `_async_monitor_loop()` (line 757)

uvicorn event loop 上的异步协程，每 1 秒轮询一次。

**触发条件** (三者同时满足):
1. `training_in_progress == False` — 不在训练中
2. `elapsed >= idle_timeout` — 空闲时间足够
3. `samples >= min_samples` OR `GSM8K data available` — 样本够或 GSM8K 数据可用

**训练循环**:
- `for step in range(train_steps_per_cycle)`:
  - Round-robin 从 prompt pool 选 `prompts_per_step` 个 prompt
  - `await asyncio.to_thread(ray.get, runner.train_step.remote(training_data))` — 关键，避免阻塞 event loop
  - 打印 step 进度

**GSM8K 耗尽保护** (line 887):
- 只有训练**成功**完成 + 无 API 样本时，才标记 `_gsm8k_exhausted = True`
- 训练失败的 cycle 不标记，下次空闲触发时可重试

**失败回滚** (line 893-898):
- `except Exception`: 记录日志，不崩溃
- `finally`: 无论如何都恢复 `mode="serving"` + `training_in_progress=False`
- 权重同步只在 step 10 成功执行 → 失败时旧权重继续使用

---

## 4. api.py — FastAPI 推理接口

**文件**: `trainable_openclaw/server/api.py`
**定位**: 独立的 FastAPI 层，不依赖 veRL/ray/torch。可在任何机器上 mock 测试。

### 4.1 Pydantic 模型

| 类 | 用途 |
|----|------|
| `ChatMessage` | 单条消息: `{role: "user", content: "..."}` |
| `ChatCompletionRequest` | OpenAI 兼容请求: model, messages, temperature, top_p, max_tokens, enable_thinking |
| `ChatCompletionResponse` | OpenAI 兼容响应: id, choices[{message, finish_reason}], usage |
| `ChatCompletionResponseChoice` | 单个回复选项 |
| `UsageInfo` | Token 用量: prompt_tokens, completion_tokens, total_tokens |
| `HealthResponse` | 健康检查响应: status, mode, uptime, active_requests, gpu_count |

### 4.2 `create_app() -> FastAPI` (line 95)

构建 FastAPI 应用，包含:

#### `GET /v1/health` (line 127)
- 返回服务健康状态
- `mode` 字段反映当前是 "serving" 还是 "training"

#### `POST /v1/chat/completions` (line 138)
- OpenAI 兼容的聊天补全接口
- **训练期间**: 返回 HTTP 503 `"Training in progress, try later"`
- 支持 `enable_thinking` 参数（Qwen3 thinking mode）
- 采样参数优先级: 请求参数 > rollout_config 默认值
- 生成完成后自动调用 `record_request()` 记录样本

### 4.3 `_app_state` (line 92)

全局共享状态字典，由 `run_serve()` 注入:

```python
_app_state = {
    "llm_client": ...,        # veRL LLMServerClient (Ray Actor 引用)
    "tokenizer": ...,         # HuggingFace tokenizer
    "rollout_config": {...},  # 采样默认参数
    "gpu_count": int,         # GPU 数量
    "runner": ...,            # ServeRunner Ray Actor 引用
    "start_time": float,      # 服务启动时间
    "active_requests": int,   # 当前活跃请求数
    "orch_state": {...},      # 训练编排器状态
    "record_request": fn,     # 样本记录回调
    "_monitor_coro": fn,      # 异步监控协程
}
```

### 4.4 Lifespan (line 109)

FastAPI 启动/关闭生命周期管理:
- **启动**: 调度 `_async_monitor_loop()` 协程到 uvicorn event loop
- **关闭**: 取消监控协程，记录日志

---

## 5. orchestrator.py — 训练编排器

**文件**: `trainable_openclaw/training/orchestrator.py`

> **注意**: 当前实际使用的是 `serve_ppo.py` 中内嵌的异步编排器（`_orch_state` + `_async_monitor_loop`），而非这个文件中的类。这个类是最初的线程版本，保留用于：
> 1. 独立的 mock 测试 (`test_orchestrator.py`, 24 个测试)
> 2. 将来非 uvicorn 环境的参考实现

### 5.1 `TrainingSample` (line 28, dataclass)

```python
@dataclass
class TrainingSample:
    prompt_ids: list[int]
    response_ids: list[int]
    timestamp: float
    metadata: dict
```

### 5.2 `TrainingOrchestrator` (line 38)

**线程安全设计**: `threading.Lock` 保护 `_samples` 和 `_last_request_time`。

**核心方法**:

| 方法 | 功能 |
|------|------|
| `record_request()` | API handler 调用，记录样本 + 更新时间戳 |
| `should_train()` | 检查: 空闲? 样本够? (或外部数据可用?) |
| `set_external_data_check()` | 注册 GSM8K 数据可用性回调，绕过 min_samples 门控 |
| `set_train_fn()` | 注册训练回调函数 |
| `start_monitoring()` | 启动后台 daemon 线程，定期轮询 |
| `_trigger_training()` | 排空样本 → 调用 train_fn → 恢复 mode（try/finally） |

**状态机**:
```
SERVING ──(should_train=True)──> TRAINING ──(callback done/error)──> SERVING
```

**与 serve_ppo 的区别**:
- 线程模型 vs 协程模型
- 类封装 vs 内嵌闭包
- 功能等价，但协程版本避免了 Ray 在线程中调用导致的死锁问题

---

## 6. 启动脚本

### 6.1 `scripts/run_serve_ppo.sh` — 生产启动

完整配置，用于实际训练:
```bash
idle_timeout=30
min_samples=1
gsm8k.enabled=true
gsm8k.num_prompts=20
train_steps_per_cycle=5
prompts_per_step=4
logger=["console","tensorboard"]
```

### 6.2 `scripts/run_serve_ppo_test.sh` — 测试启动

低阈值快速触发，用于集成测试 (A2/A3):
```bash
idle_timeout=5
min_samples=2
gsm8k.enabled=false
train_steps_per_cycle=1
prompts_per_step=2
```

### 6.3 `scripts/run_gsm8k_e2e_test.sh` — 端到端 GSM8K 测试

可配置参数的完整测试流程:

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `STEPS` | 20 | 训练步数 |
| `PROMPTS_PER` | 16 | 每步 prompt 数 |
| `GSM8K_N` | 320 | 加载的 GSM8K 数据量 |
| `GPU_MEM` | 0.35 | vLLM GPU 内存占比 |
| `RESP_LEN` | 4096 | 最大生成 token 数 |
| `IDLE_TO` | 10 | 空闲超时 (秒) |
| `LR` | 3e-6 | 学习率 |

**4 步流程**:
1. **Cleanup**: 杀掉旧 serve_ppo 进程 + GPU 进程，清理 Ray
2. **Start**: 启动 serve_ppo，参数化配置
3. **Monitor**: 等待服务就绪 → 等空闲触发训练 → 轮询 `train_step completed` 日志
4. **Summary**: 打印 rewards/loss 趋势 + 最终 health 状态

---

## 7. 关键设计决策

### 7.1 Reward 放置位置（loss=0 的根因修复）

```
旧代码: token_level_scores[i, :resp_len] = reward     → reward × length 噪音
新代码: rm_scores[i, valid_response_length - 1] = reward → 仅最后 token
```

GRPO 的 `compute_grpo_outcome_advantage` 用 `token_level_rewards.sum(dim=-1)` 计算 scalar score。reward 必须只放最后 token，否则 sum 被长度污染。

### 7.2 asyncio.to_thread 防死锁

```
旧代码: ray.get(runner.train_step.remote(...)) 在协程中直接调用
        → 阻塞 uvicorn event loop → HTTP 请求全部挂死

新代码: await asyncio.to_thread(ray.get, runner.train_step.remote(...))
        → ray.get 在独立线程执行 → event loop 不阻塞 → 503 正常返回
```

### 7.3 生成在 Ray Actor 内 vs Driver

```
Driver 生成 (旧):
  asyncio.run() 在监控线程中调用 llm_client.generate()
  → Ray RPC 和 uvicorn event loop 交叉导致死锁

Ray Actor 内生成 (新):
  train_step() 在 Ray Actor 内用 ThreadPoolExecutor + urllib.request
  → 完全脱离 uvicorn event loop → 无死锁
```

### 7.4 config 驱动 vs 硬编码

```
旧代码: epochs=1, mini_batch_size=total_samples, shuffle=False (硬编码)
新代码: epochs=actor_config.ppo_epochs, mini_batch_size=..., shuffle=actor_config.shuffle (从配置读取)
```

### 7.5 GSM8K 耗尽保护

```
旧代码: finally 中标记 exhausted → 训练失败也标记 → 无法重试
新代码: try 成功后标记 → 失败不标记 → 下次可重试
```

### 7.6 veRL 原生组件复用

| 组件 | 用途 |
|------|------|
| `CheckpointEngineManager` | sleep/wake/update_weights 统一管理 |
| `gsm8k.compute_score()` | GSM8K 答案评分 (替代自定义解析器) |
| `gsm8k.extract_solution()` | 从 `#### 42` 提取答案 (替代自定义 正则) |
| `compute_response_mask()` | 标记有效 response token |
| `compute_advantage(GRPO)` | GRPO 组内归一化 advantage |
| `compute_data_metrics()` | 数据统计指标 |
| `compute_timing_metrics()` | 各阶段耗时指标 |
| `compute_throughout_metrics()` | 吞吐量指标 |
| `Tracking` | 统一日志接口 (console/tensorboard/wandb) |

---

## 附录: 测试覆盖

| 测试文件 | 测试数 | 类型 | 覆盖内容 |
|---------|--------|------|---------|
| `tests/test_serve_ppo.py` | 9 | Mock | Pydantic 模型, FastAPI 端点 |
| `tests/test_orchestrator.py` | 24 | Mock | 编排器逻辑, 线程安全 |
| `tests/test_a1_integration.py` | 12 | GPU | 推理 API 全部功能 |
| `tests/test_a2_integration.py` | 5 | GPU | 空闲检测→训练触发→503→恢复 |
| `tests/test_a3_integration.py` | 9 | GPU | 权重同步+多周期稳定性 |
| **总计** | **59** | | |
