# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# 项目进展记录

## 2026/05/21

### 远程环境搭建
- 远程 Linux GPU 机器 (AutoDL, `connect.westd.seetacloud.com:29669`, `/data/wangye/trainable-openclaw`)
- vllm 0.12.0 + Qwen3-0.6B 推理通过验证
- serve_ppo 尝试启动，暂搁置

### 代码改动
- `requirements.txt`: 添加 fastapi/uvicorn/httpx/aiohttp/openai 等依赖
- `verl-main-0516/verl/workers/config/model.py`: 默认 attention `flash_attention_2` → `sdpa`
- `scripts/set_env.sh`: Linux 环境安装脚本
- `scripts/setup_server.sh`: 服务端部署脚本（模型下载 + vllm 推理测试）
- `scripts/run_serve.sh`: serve_ppo 启动脚本
- `scripts/check_models.py`: ModelScope 模型查询工具
- `scripts/download_model.py`: ModelScope 模型下载工具

### 待办
1. 手动启动远端机器，运行环境脚本
2. 解决 serve_ppo 遗留问题
3. 提交并推送本地改动

---

## 2026/05/22

### 环境更新
- 远端服务器安装了 conda python 环境（base, Python 3.13）
- vllm 升级到 0.18.1

### 问题修复：libstdc++ 版本不兼容
- **现象**: `ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version 'CXXABI_1.3.15' not found`
- **根因**: conda 安装的 pyarrow/sklearn 需要新版 libstdc++，系统自带版本太旧
- **修复**: 
  - `scripts/set_env.sh` 和 `scripts/setup_server.sh` 中添加 `export LD_LIBRARY_PATH=/data/anaconda3/lib:$LD_LIBRARY_PATH`
  - 永久化：写入 `/data/anaconda3/etc/conda/activate.d/env_vars.sh`，每次 conda activate 自动生效

### veRL Inference Server 启动成功
- FastAPI 服务正常监听 `http://0.0.0.0:8000`
- `/v1/health` 健康检查端点正常
- `/v1/chat/completions` 聊天端点返回 200

### 问题解决：推理输出乱码（load_format=dummy）
- **现象**: `/v1/chat/completions` 返回的 content 为多语言乱码、随机 token
- **根因**: vLLM 启动时使用了 `load_format=dummy`（随机权重），而不是加载真实模型权重
  - serve_ppo 使用 HYBRID 模式（rollout engine + training engine 混合部署）
  - HYBRID 模式下，`dummy` 是故意的——training engine 会在训练时通过 `sync_weights()` 同步真实权重
  - 但 serve_ppo 纯推理模式下没有 training engine，权重永远不会被同步，导致随机输出
  - `vllm_async_server.py:135` 的 `dummy→auto` 修正逻辑只对非 HYBRID 模式生效
- **修复**:
  - `scripts/start_server_4b.sh`: 添加 `actor_rollout_ref.rollout.load_format=auto` 显式覆盖
  - `verl-main-0516/verl/trainer/serve_ppo.py`: 代码层自动修正——serve-only 模式下检测到 load_format=dummy 时自动改为 auto
  - 确认 vLLM worker 日志显示 `Loading safetensors checkpoint shards` 加载真实权重
- **验证**: Qwen3-4B 推理输出正常——"Introduce yourself" → 正确自我介绍；thinking 模式也正常

### 交互式聊天 CLI
- `scripts/chat.py`: 多轮对话 CLI 工具
  - 用法: `python scripts/chat.py [-t temp] [-m max_tokens] [-M model] [--no-think] [server_url]`
  - 命令: `/think` 切换thinking模式, `/clear` 清空历史, `/undo` 撤销, `/quit` 退出

### A2：空闲检测 + 训练触发

- `trainable_openclaw/training/orchestrator.py`: 训练编排器
  - **空闲检测**: 后台监控线程，每1s轮询，记录最后请求时间
  - **样本积累**: API handler 在每次生成完成后调用 `record_request()`，存入 deque
  - **训练触发条件**: 空闲超时 (idle_timeout) AND 样本数 >= min_samples
  - **训练期间**: `training_in_progress=True` → API 返回 503 `Training in progress, try later`
  - **生命周期**: SERVING → (idle+样本够) → TRAINING → (完成) → SERVING
- `verl-main-0516/verl/trainer/serve_ppo.py` 中的集成:
  - `_train_bridge`: orchestrator 线程回调 → Ray actor 执行 train_step
  - sleep/wake 暂未启用 (vLLM V1 引擎 sleep 后 CUDA 崩溃)，A3 重新接入
- `tests/test_orchestrator.py`: 24 个 mock 测试 (TrainingSample, 构造/校验, 记录, 触发, 监控, 线程安全)
- `tests/test_a2_integration.py`: 5 个 GPU 集成测试 (serving→503→recovery→inference)
- **验证**: 24 mock + 5 GPU 全部通过 ✅

### A3：权重同步 + 恢复推理

- **weight sync 机制**: train_step 完成后调用 `actor_rollout_wg.update_weights(global_steps, mode="naive")` 将 actor engine 参数同步到 vLLM rollout engine
  - Naive mode: `get_per_tensor_param` → `BucketedWeightSender` (ZMQ/IPC) → vLLM 加载
  - **free_cache_engine=False**: 绕过 vLLM V1 sleep/wake 的 CUDA 崩溃问题。权重同步在-place 进行，不释放/重新分配 GPU 内存
  - **checkpoint_engine.backend=naive**: 直接 IPC 传输，不需要 NCCL
  - 两个覆盖在 `run_serve()` 中自动设置
- `train_step()` 流程: 训练计算(当前stub 3s) → weight sync → 日志记录 sync 耗时
- `_train_bridge`: 简化为直接调用 `runner.train_step.remote(batch)`，移除 sleep/wake 注释
- **失败回滚**: orchestrator 的 try/finally 确保训练失败后 `mode=serving`，旧权重仍在使用（因为 sync 只在训练成功后执行）
- `tests/test_a3_integration.py`: 10 个 GPU 集成测试
  - 基础: serving 模式验证
  - Cycle 1: 样本积累 → 空闲触发 → 503 验证 → 恢复 → 推理验证（含乱码检查）
  - Cycle 2: 多轮训练/推理循环稳定性验证
  - 最终状态: health check 确认 serving 模式

### 当前状态
- serve_ppo 支持 A1+A2+A3：推理 + 空闲检测 + 训练编排 + 权重同步 + GRPO 训练
- 启动命令示例（含 GSM8K + 低阈值）:
  ```bash
  ... +trainer.idle_timeout=5 +trainer.min_samples=1 \
      +trainer.gsm8k.enabled=true +trainer.gsm8k.num_prompts=4
  ```
- train_step 为真实 GRPO 训练（生成N个答案→休眠→计算奖励→计算old_log_probs→计算GRPO advantage→更新actor→同步权重→唤醒）

### 2026/05/22 (下午) — A3 架构重构 + GRPO 训练

- **架构重构**: 按用户要求，使用 veRL 的 CheckpointEngineManager 管理 sleep/wake/sync
  - `ServeRunner.run()`: 新增 CheckpointEngineManager 创建（step 5b）
  - `train_step()`: 使用 `checkpoint_manager.sleep_replicas()` / `wake_up_replicas()` / `update_weights()` 替代手动 ray.get(server.sleep.remote())
  - `sleep_replicas()` / `wake_replicas()`: 简化为委托给 checkpoint_manager
  - CheckpointEngineManager 的 `@auto_await` 处理 Ray actor 内部的 event loop 冲突（thread pool 方式）

- **GRPO 训练实现**:
  - 生成阶段在 driver (`_train_bridge`) 完成：通过 asyncio.run() 在监控线程中调用 llm_client.generate() 生成 N 个答案
  - 奖励计算在 driver 完成：解码→提取 GSM8K 答案（`#### <number>`）→与 ground truth 比较
  - 训练阶段在 Ray actor (`train_step()`) 完成：
    1. `_build_training_batch()`: 从原始 token 序列构建 DataProto（padding/attention_mask/position_ids/token_level_scores）
    2. `compute_log_prob()`: actor 前向计算 old_log_probs
    3. `compute_advantage()`: GRPO group-norm advantage（按 prompt 分组）
    4. `update_actor()`: FSDP 前向/反向/优化器步骤
    5. `update_weights()`: 权重同步到 vLLM（naive/IPC）
    6. `wake_up_replicas()`: 恢复推理服务

- **GSM8K 数据加载**:
  - `_load_gsm8k_data()`: 从 HuggingFace datasets 加载 GSM8K 训练集
  - `_extract_gsm8k_answer()`: 从 `#### <number>` 格式提取数值答案
  - 配置: `+trainer.gsm8k.enabled=true +trainer.gsm8k.num_prompts=8`

- **orchestrator 增强**: 新增 `set_external_data_check()` 回调，允许 GSM8K 等外部数据绕过 min_samples 门控
  - 24 个已有测试全部通过

- **待办**: 上传到远程 Linux 服务器，启动 serve_ppo + GSM8K GRPO 训练，验证完整闭环

### 关键路径
- SFTP: `connect.westd.seetacloud.com:29669` → `/data/wangye/trainable-openclaw`
- 模型目录: `/data/models/Qwen3-4B`
- conda: `/data/anaconda3`
- verl editable install: `/root/autodl-tmp/wangye/trainable-openclaw/verl-main-0516`（与 `/data/wangye/` 同文件系统，文件一致）

---

## 2026/05/25

### GRPO Bug Fix: Prompt Template 缺少 `####` 指令

- **根因**: `_load_gsm8k_data()` 的默认 prompt template 为 `"{question}\nLet's think step by step.\n"`，没有告知模型输出 `#### <answer>` 格式
- **影响**: `_extract_gsm8k_answer()` 找不到 `####` 分隔符 → 所有 reward=0 → GRPO advantage 始终为 0 → loss=0 → 模型永远不学习
- **修复**: prompt template 改为 `"{question}\nLet's think step by step and output the final answer after \"####\".\n"`
- **验证**: rewards 从恒为 0/16 提升到 8/16~4/16（但还波动）

### GRPO 遗留问题

- **loss 恒为 0**: 即使有 non-zero rewards，loss 仍为 0.0000
  - 分析: 当组内 4 个回答都正确/都错误时，GRPO advantage=0（组内 reward 无方差）
  - `compute_grpo_outcome_advantage` 在 `core_algos.py:304` 用 `token_level_rewards.sum(dim=-1)` 计算 scores，不同回答长度导致 score 差异 → advantage 非零但信号来自长度而非正确性
- **max_tokens 不足**: 原默认 1024，模型推理未完成就被截断 → `####` 未输出
  - 修复: max_tokens 默认值 1024→2048，orchestrator 内 `max(rollout_config.response_length, 2048)`
- **已知限制**: 模型不总是输出 `####` 格式，温度 1.0 但同组内回答结果趋同

### Event Loop Fix: asyncio.to_thread

- **问题**: `ray.get(runner.train_step.remote(...))` 在训练监控协程中直接调用 → 阻塞 uvicorn event loop → 训练期间 HTTP 请求全部挂死（连 503 都返回不了）
- **修复**: 改为 `await asyncio.to_thread(ray.get, runner.train_step.remote(...))`，ray.get 在独立线程执行，event loop 不阻塞
- **验证**: A2/A3 测试中训练期间 503 正常返回

### 集成测试全部通过 (59/59)

- 新增测试记录文件: `tests/TEST_RECORD.md`
- 修复测试超时: `test_a2` 的 `TRAINING_TIME` 从 8s→90s, `test_a3` 从 10s→120s
- 测试服务器配置: `scripts/run_serve_ppo_test.sh` (idle_timeout=5, min_samples=2, train_steps_per_cycle=1, gsm8k=false)

### Git 状态
- 未提交: `serve_ppo.py` (prompt template fix + max_tokens + asyncio.to_thread + debug logging)
- 未提交: `test_a2_integration.py`, `test_a3_integration.py` (TRAINING_TIME 修复)
- 新增未跟踪: `tests/TEST_RECORD.md`

---

## 2026/05/29

### 代码清理
- 删除 `trainable_openclaw/` 下 5 个空壳/占位目录（19 个文件）：
  `agents/`, `dashboard/`, `engine/`, `evaluation/`, `logging/`
- 保留: `server/api.py` (推理API) + `training/orchestrator.py` (测试用)

### B0: 对话日志系统
- `trainable_openclaw/logging/conversation_store.py` — SQLite + WAL 模式, sessions/messages 双表
  - session CRUD: create/get/list/delete, 按 user 过滤, 按时间排序
  - message CRUD: add/get, 自动更新 session 的 updated_at + message_count
  - 分析查询: query_messages (user/role/time 组合过滤), search_content (关键词模糊搜索)
  - 统计: get_statistics (session数/消息数/用户分布/角色分布/时间范围)
  - 线程安全: threading.Lock 保护写入, WAL 模式读并发
- `trainable_openclaw/logging/viewer.py` — CLI 离线查看工具 (5 子命令)
  - `users`: 列出用户及统计
  - `sessions`: 列出会话 (--user, --limit 过滤)
  - `view <id>`: 查看完整对话 (ANSI 色区分角色)
  - `search <kw>`: 搜索消息内容
  - `stats`: 聚合统计概览
- api.py 集成: ChatCompletionRequest 新增 `user` 字段, 生成完成后自动写入日志
- serve_ppo.py 集成: run_serve() 初始化 ConversationStore 注入 _app_state
- `tests/test_conversation_store.py`: 23 测试全部通过 ✅

### B1.2: OASST2 数据处理
- 数据集选择: OpenAssistant/oasst2 (Apache 2.0, 135K messages, ~47K conversation trees)
- `scripts/prepare_oasst2.py` — 数据准备脚本:
  1. 下载 (hf-mirror.com 镜像自动检测)
  2. 解析 flat messages → conversation trees (主路径遍历)
  3. Train/Test split (80/20, 按 tree_id 防止泄漏)
  4. Train → ConversationStore (含模拟用户反馈)
  5. Test → JSONL (供模型评测)
  6. Labels + rank → 模拟中文用户反馈 (quality/creativity/humor/helpfulness)
  7. quality_score 归一化 0-1 (可用作 reward 信号)
- `requirements.txt`: 新增 `datasets>=3.0.0`
- 待办: 基础模型评测 (需 GPU), 远程运行数据下载

### 文档更新
- `docs/roadmap.md`: Phase 1 里程碑完成, B0 已完成, B1.2 进行中
- `docs/code_guide.md`: 完整代码说明文档 (架构图+流程图+函数详解)
- README.md / README_zh.md: 更新项目结构和阶段状态

### Git 状态
- 已提交:
  - `e65c713` fix: GRPO align with veRL native (reward placement, prompt template, e2e GSM8K)
  - `65602e6` docs: update roadmap — Phase 1 milestone complete
  - `d312f20` chore: remove dead stub directories
- 未提交: B0日志系统 + B1.2数据准备 + 文档更新
