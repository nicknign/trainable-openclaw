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

# 硬件环境
远程linux的访问方式见文件：.vscode\sftp.json  ssh连接与sftp使用相同的端口连接


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
  - `3addddd` feat: conversation log system (B0) + OASST2 data preparation (B1.2)
  - `9325cdd` feat: Phase 1.5 S1+S2 — seed extraction + User Sim Agent correction dialogue engine
- 未提交: Phase 1.5 S2 脚本完善 + 训练/测试数据集生成

---

## 2026/05/30

### Phase 1.5 S2: GPU 服务器 + 仿真流水线搭建

- **GPU 服务器配置**: RTX 4080 SUPER (32GB), `connect.westc.seetacloud.com:27814`
- **serve_ppo 启动**: Qwen3-4B + LoRA rank=16, HYBRID 模式
  - `max_model_len=4096` 适配 32GB 显存
  - `gpu_memory_utilization=0.4` 给 FSDP 训练 worker 留空间
  - `trainer.logger='[console]'` 避免 WandB API key 错误
  - `idle_timeout=999999` 禁用自动训练触发（纯推理模式）
- **API 路径修复**: base_url 从 `http://localhost:8000` 改为 `http://localhost:8000/v1`（serve_ppo 实际服务路径）
- **`.env` 文件**: DeepSeek API 密钥通过 `.env` 注入，已加入 `.gitignore`

### Phase 1.5 S2: 用户模拟纠错对话流水线

- **`scripts/run_simulation.py`** — 自包含仿真脚本 (~800 行)，无需导入项目包
  - **5 种用户画像**: 张工(代码)、李编辑(写作)、王同学(数学)、Alex(工程)、陈测试(QA)
  - **32 个 LMSYS 类别全覆盖**: 均衡分配到 5 种画像（之前 53% 落在默认画像，已修复）
  - **多轮纠错对话**: User Sim (DeepSeek-v4-flash) 审查 Qwen3-4B 回答 → 发现错误 → 给出具体纠错 → Qwen 修正 → 循环直到通过或超限
  - **Messages 格式记录**: `[{role, content, reasoning}]`，`<think>` 内容提取到 `reasoning` 字段
  - **全中文化**: 字段名、判定值、界面文字统一中文
  - **串行处理**: 逐条执行，终端实时打印每轮对话详情
  - **进度条**: tqdm 显示实时通过率
  - **模式**: `--mock`（DeepSeek 模拟双方）/ `--no-mock`（真实 GPU）/ `--dry-run` / `--stats-only`
- **画像分布修复**: `PERSONA_CATEGORY_MAP` 覆盖全部 32 个 LMSYS 类别
  - 之前: qa_tester 53% → 现在: 李编辑 28% / 王同学 25% / 陈测试 19% / Alex 16% / 张工 12%
- **库文件同步**: `user_sim.py` 和 `engine.py` 同步更新画像映射

### 训练/测试数据集制作

- **种子打乱拆分**: 3200 条种子 → `seed_train_100.jsonl` (100) + `seed_test_50.jsonl` (50)，5 种画像均有覆盖
- **训练集生成**: 远程 GPU 服务器后台运行中（预计 ~2 小时完成 100 通）
- **测试集生成**: 训练集完成后接着跑 50 通
- **数据用途**: 
  - 训练数据: `(错误回答, 纠错意见, 修正后回答)` 三元组 → 用于后续 B1/B2 流程
  - 测试数据: 验证自进化引擎训练提升效果
  - Rubric 种子: 纠错维度聚合 → B2 Rubric 生成器输入

### 远程环境更新

- 主机名修正: `connect.westc.seetacloud.com`（非 westd）
- 端口 27814（GPU 机器），serve_ppo 常驻运行

### Phase 2: 自进化评判系统 — 代码开发

#### S3: 轨迹评估与数据导出 (`evaluation/trajectory_eval.py`)
- `grade_trajectory(traj)` → "直接通过" / "纠错后通过" / "部分通过" / "失败"
- `extract_training_pairs(traj)` → `(错误回答, 纠错意见, 修正回答, 修正思考)` 三元组
- `extract_rubric_seeds(trajectories)` → 聚合纠错维度（17 个维度类别，关键词推导）
- `process_trajectories(input_file, output_dir)` → 完整 S3 流水线
- **实际运行**: 训练集 100 条 → 67 直接通过 / 32 纠错后通过 / 1 部分通过 → 48 training pairs, 14 rubric 种子维度
  - 维度分布: 计算准确性(22) / 事实准确性(18) / 结构逻辑(11) / 步骤完整性(10) / 信息完整性(9) / 文笔流畅(8) / 其他(7) / 意境表达(6) / 清晰易懂(4) / 表达严谨性(4) / 性能(2) / 类型注解(2) / 代码正确性(1) / 错误处理(1)

#### B1: 用户反馈分析 (`evaluation/feedback.py`)
- `FeedbackPattern` dataclass: 模式名称, 描述, 频次, 严重程度, 典型示例, 建议检查项
- `FeedbackAnalyzer.analyze()` — async LLM 分析（DeepSeek-v4-flash），从种子维度聚类为高层次反馈模式
- `FeedbackAnalyzer.analyze_simple()` — 无 API 模式，直接转换 rubric seeds → FeedbackPattern
- **实际运行 (LLM 模式)**: 14 维度 → 7 个高层次模式: 事实性错误 / 计算与推理错误 / 格式与规范遵循 / 语言表达与意图理解 / 细节准确性不足 / 安全与技术实现错误 / 内容冗余与重复

#### B2: LLM 自主生成 Rubrics (`evaluation/rubric.py`)
- `Rubric` dataclass: 完整生命周期管理（活跃 → 命中更新 → 归档）
- `RubricStore`: JSON 持久化 + 版本管理 + 模糊匹配
- `RubricGenerator.generate()` — async LLM 生成严格量化评分 prompt
- `RubricGenerator.generate_simple()` — 模板回退模式（4 级评分）
- `RubricGenerator._generate_fallback()` — API 失败时的模板回退（新增）
- `generate()` 新增验证: 空响应/短响应 (<50 chars) 自动抛异常触发回退
- **实际运行 (LLM 模式)**: 7 个模式 → 6 个高质量 Rubric（详细扣分细则 + JSON 输出格式），4 个 API 返回空被清理
- **实际运行 (Simple 模式)**: 14 个种子 → 14 个模板 Rubric
- **当前 rubrics.json**: 20 条（14 simple + 6 LLM 高质量）

#### B3: LLM Judge 执行器 (`evaluation/judge.py`)
- `JudgeExecutor.score_one(rubric, answer)` — 单 rubric × 单回答
- `JudgeExecutor.score_answer(answer, rubrics)` — M 条 rubric × 单回答
- `JudgeExecutor.score_answers(prompt, answers, rubrics)` — N 答案 × M rubric 矩阵
- `JudgeExecutor.compute_grpo_rewards(score_results, reward_mode)` — "mean"/"total"/"pass_fail"
- `_parse_score_json()` — JSON 解析 + 正则回退提取分数
- Judge 可执行性验证: 简单模式跳过 API 验证，LLM 模式测试了第一条 rubric

#### 流水线脚本 (`scripts/run_evaluation.py`)
- `run_full_pipeline()` — S3 → B1 → B2 → B3 串联
- `--simple` 模式（无 API）/ `--s3-only` 模式
- 完整流水线验证通过（simple 模式），LLM 模式验证通过

### 待办
- 测试集 50 通生成完成 → 跑 S3 评估 + 流水线
- 对比训练集/测试集评估结果
- 修复 B3 Judge JSON 解析失败问题（测试答案太通用）
- 提交 Phase 2 代码

---

## 2026/05/31

### veRL GRPO 训练日志恢复

- **问题**: 重构 serve_ppo 后 veRL 原生指标（actor/loss, grad_norm, critic/rewards, perf/throughput, response_length, timing_per_token_ms 等）从日志中消失
- **修复**: `train_step()` 返回值新增 `step_metrics`，`_log()` 和监控循环恢复完整 veRL 指标输出
- **FSDP 兼容**: `_s()` / `_ss()` / `_avg()` helper 处理 FSDP 返回的 list 值（如 `actor/grad_norm`），自动求均值
- **输出目标**: `/tmp/serve_ppo_train.log`（per-step veRL 详细指标） + stderr `[TRAIN STEP]`（实时摘要）

### 训练批次增大 + Rubric 优化

- **批次**: 4 prompts/step → 8 prompts/step，8 × 8 rollout = 64 答案/step（之前 32）
- **Rubric 优化**: 20 条 → **5 条高质量 rubric**（`data/rubrics_v2.json`）
  - 旧 20 条：14 条模板通用 rubric + 6 条 LLM 生成且有大量重叠
  - 新 5 条：事实与知识准确性 / 逻辑与计算正确性 / 完整性与步骤清晰度 / 格式与指令遵循 / 语言表达质量
  - 每条包含详细扣分规则 + 严格 JSON 输出格式
  - 格式大小: ~1415 chars vs 旧版 ~15000+ chars

### 训练 Round 1 — 20 条旧 Rubric（基线）

- **配置**: 8p×8r=64, lr=3e-6, 10 steps
- **结果**: reward 0.186~0.296, mean=0.248, 76 min
- **问题**: rubric 太多太泛，区分度低，reward 信号弱

### 训练 Round 2 — 5 条优化 Rubric

- **配置**: 8p×8r=64, lr=5e-6, 10 steps, `data/rubrics_v2.json`
- **启动**: `scripts/start_train.sh`（PID 32113, port 13738）
- **结果**:

| Step | Reward | >0.5 | Loss | Grad | Time |
|------|--------|------|------|------|------|
| 1 | 0.631 | 48/64 | 0.0001 | 0.132 | 315s |
| 2 | 0.654 | 50/64 | 0.0099 | 0.155 | 327s |
| 3 | 0.575 | 42/64 | 0.0045 | 0.144 | 335s |
| 4 | 0.693 | 53/64 | -0.0061 | 0.150 | 319s |
| 5 | 0.595 | 45/64 | 0.0002 | 0.142 | 308s |
| 6 | 0.682 | 53/64 | 0.0126 | 0.150 | 312s |
| 7 | 0.590 | 40/64 | -0.0028 | 0.145 | 331s |
| 8 | 0.643 | 51/64 | 0.0142 | 0.142 | 311s |
| 9 | 0.616 | 46/64 | 0.0058 | 0.137 | 311s |
| 10 | **0.698** | 55/64 | 0.0063 | 0.155 | 304s |

- **总耗时**: 3172s (~53 min)，throughput=130 tok/s/gpu，response_len~504
- **对比 Round 1**: reward 2.5x 提升（0.248→0.638），区分度更好，耗时更短（53min vs 76min）

### 分析

- Rubric 优化是关键杠杆：少量精炼 rubric 优于大量泛化 rubric
- Reward 波动震荡（0.575~0.698），10 步无明显上升趋势——lr=5e-6 偏低，或每步 prompt 采样不同导致
- GRPO advantage 信号仍偏弱（同 prompt 8 个回答的 rubric 评分区分度有限），loss 在 -0.006~0.014 波动
- 下一步方向：增大 lr、增加步数、或改用 pairwise 排名 reward

### 远程环境

- GPU 机器: `connect.westc.seetacloud.com:13738`（RTX 4090, 已停止训练进程）
- 模型: Qwen3-4B + LoRA rank=16
- serve_ppo 日志: `/tmp/phase3_train.log`, `/tmp/serve_ppo_train.log`

---

## 2026/06/01

### Judge API 费用优化

**问题**: 每步训练 64 回答 × 5 rubric = 320 次 API 调用，thinking 模式白白消耗 tokens，10 步花了 30+ 元。

**三项优化：**

1. **合并多 rubric 为单次调用** (`use_merged=True`, 默认)
   - `score_merged()`: 将所有 rubric 合并为一个 prompt，一次 API 返回所有分数
   - API 调用量: 64×5=320 → 64×1=64（减少 5x）
   - `_build_merged_prompt()`: 清理 {content} 占位符，统一放置待评估内容
   - `_parse_merged_response()`: 解析 JSON 数组 → RubricScore 列表，含容错
   - `score_answer()` 在 `use_merged=True` 时走合并路径，异常时回退零分

2. **关闭 thinking 模式** (`enable_thinking=False`, 默认)
   - 评分是结构化任务，不需要深度推理
   - 节省 thinking tokens（之前约占总 token 1/3）

3. **降低 max_tokens**
   - `score_one`: 2000 → 500
   - `score_merged`: 800（新方法）

**改动文件：**
- `trainable_openclaw/evaluation/judge.py`: 新增 `score_merged()`, `_build_merged_prompt()`, `_parse_merged_response()`；默认 `enable_thinking=False`, `use_merged=True`
- `trainable_openclaw/training/reward_bridge.py`: 默认 `enable_thinking=False`, `use_merged=True`；传递 `use_merged` 给 JudgeExecutor

**预估节省**: 总 token 消耗降至原来的 ~1/5，费用从 ~3 元/步 → ~0.6 元/步

### 训练批次进一步增大

- **问题**: 8 prompts/step 太少，只有 48 个 prompt 池，每步梯度信号不一致
- **改动**: `scripts/start_train.sh`
  - `prompts_per_step`: 8 → **48**（全量，每步覆盖全部 prompt 池）
  - `ppo_mini_batch_size`: 4 → **16**（配合更大 batch）
  - 48 prompts × 8 rollout = 384 答案/步（之前 64）
- **tradeoff**: 每步生成+评分+训练时间更长，但梯度信号更稳定

### 扩展仿真数据规模（计划中）

- **现状**: 仅 48 个训练对（来自 100 条种子的试跑），种子池有 3200 条
- **计划**: 跑 500 条种子 → 预计 ~200+ 训练对
- **要求**: 所有外部模型调用用 `deepseek-v4-flash` + 关闭思考（仿真脚本 LLMClient 默认已满足）
- **流程**: 启动 GPU → serve_ppo 纯推理 → `run_simulation.py --no-mock --max-prompts 500` → 重新提取训练对 → 用更大 prompt 池重新训练

### 今日改动的文件汇总

| 文件 | 改动 |
|------|------|
| `trainable_openclaw/evaluation/judge.py` | +merged scoring, enable_thinking=False, max_tokens 500/800 |
| `trainable_openclaw/training/reward_bridge.py` | enable_thinking=False, use_merged=True |
| `scripts/start_train.sh` | prompts_per_step=48, mini_batch=16, echo 更新 |

### 远程服务器状态

- GPU 机器已关机（端口 13738 不可达），需在 AutoDL 控制台启动
- 启动后待做：上传改动文件 → 启动 serve_ppo → 跑仿真扩数据 → 重新训练验证

---

## 2026/06/02

### 扩展仿真数据（500 条种子）

- **运行**: 3200 条种子 → 采样 500 条 → User Sim 多轮纠错 → 570 训练对 + 80 测试对
- **产物**: `data/phase3_datasets/`
  - `train_prompts.jsonl`: 557 pairs / 496 unique prompts（训练，去重叠后）
  - `test_prompts.jsonl`: 80 pairs / 78 unique prompts（测试，与训练无重叠）
  - `all_prompts.jsonl`: 650 pairs（全部）
  - `baseline_eval.json`: 训练前纠错率基线（含 per_category / per_prompt）
- **11 个类别**: coding(117), brainstorming(113), copywriting(88), creative writing(70), explanation(30), debugging(28), debating(28), translation(28), instruction following(26), logical reasoning(24), math(18)

### 动态 Rubric（8 条 category-aware）

- **文件**: `data/rubrics_dynamic.json`（8 条，按类别适配不同检查维度）
- **机制**: 训练时从 prompt 类别匹配对应 rubric（`max_rubrics=8`），非固定 rubric 集合
- 替换了之前的 5 条固定 `rubrics_v2.json`，每类 prompt 得到针对性评分

### 关键 Bug 修复

1. **Ray actor event loop 冲突** — `asyncio.run()` 在 Ray actor 内部创建新 event loop，与已有 uvloop 冲突，导致所有 reward=0
   - 修复: judge.py 新增完整 sync API（`score_merged_sync`, `score_answers_sync`），使用 `openai.OpenAI` + `ThreadPoolExecutor`
   - reward_bridge.py 从 `asyncio.run()` 改为直接调用 sync 方法

2. **Merged JSON 截断** — 8 条 rubric 合并 prompt 超过 `max_tokens=800`，JSON 被截断 → 解析失败 → reward=0
   - 修复: `max_tokens` 动态缩放 → `max(800, len(rubrics) * 200)`

3. **数据字段不匹配** — `_load_trajectory_data()` 期望 `种子提示词` 但文件用 `prompt`
   - 修复: `p.get("种子提示词", "") or p.get("prompt", "")`

4. **Rubric 字段不匹配** — `Rubric.from_dict()` 收到 `适用类别` 等未知字段
   - 修复: 过滤已知 dataclass 字段

5. **Train/Test 泄漏** — 13 个 prompt 同时出现在训练和测试集
   - 修复: `tmp/fix_leak.py` 从训练集去除重叠 prompt → 557 pairs / 496 unique，train/test 零重叠

### 模型 Checkpoint 保存

- **`ServeRunner.save_checkpoint()`**: 新增方法，委托 veRL 的 `actor_rollout_wg.save_checkpoint()` → FSDPCheckpointManager 保存 per-rank sharded model/optimizer/extra state + HF config/tokenizer
- **训练循环集成**: `_async_monitor_loop` 每步检查 `global_step % save_ckpt_interval == 0`，触发 `runner.save_checkpoint.remote()`
- **配置**: `+trainer.save_ckpt_interval=10`, `+trainer.checkpoint_dir=/data/wangye/trainable-openclaw/checkpoints`, `max_ckpt_to_keep=3`
- **产物**: `model_world_size_1_rank_0.pt` (LoRA adapter ~631KB) + `huggingface/` (config/tokenizer)
- **测试脚本**: `scripts/test_checkpoint.sh` — 1 步训练 + 立即保存 + 验证文件

### 正式训练启动

- **配置**: 5 rounds × 10 steps, 48 prompts/step × 4 rollouts = 192 answers/step, lr=5e-6, LoRA rank=16
- **数据**: 509→496 unique prompts（动态 rubric 已启用），与测试集零重叠
- **Checkpoint**: 每 10 步保存到 `/data/wangye/trainable-openclaw/checkpoints/`，保留最近 3 个
- **远程**: `connect.westc.seetacloud.com:13738`, PID 152756, RTX 4090
- **日志**: `/tmp/phase3_train.log` (服务), `/tmp/serve_ppo_train.log` (训练详情)
- **预估**: ~9 小时 (50 steps × ~11 min/step)

### 今日改动文件

| 文件 | 改动 |
|------|------|
| `verl-main-0516/verl/trainer/serve_ppo.py` | +save_checkpoint, +checkpoint config, checkpoint call in monitor loop |
| `trainable_openclaw/evaluation/judge.py` | +sync API (score_merged_sync, score_answers_sync), dynamic max_tokens |
| `trainable_openclaw/training/reward_bridge.py` | asyncio.run → sync path, removed async methods |
| `trainable_openclaw/evaluation/rubric.py` | from_dict() filter unknown kwargs |
| `scripts/start_train.sh` | +save_ckpt_interval, +checkpoint_dir, rollout_n 8→4, mini_batch 16→8, max_rounds 10→5 |
| `scripts/test_checkpoint.sh` | 新建: checkpoint 验证脚本 |
| `tmp/fix_leak.py` | 新建: train/test 重叠修复 |

---

## 2026/06/03

### 训练总结与分析

- **总步数**: 42 steps（Phase 1: 15 小batch debug + Phase 2: 27 全batch）
- **Checkpoint**: 仅 Phase 1 的 `global_step_10` 被保存（Phase 2/3 重启后 `save_ckpt_interval=10` 未触发，丢失后续检查点）
- **奖励趋势**: Phase 2 mean=0.184, 范围 0.142~0.297, 轻微下行
- **分析产物**: `runs/2026-06-03_42steps/` — 训练日志 + 分析报告 + 基线/训练后评测

### Checkpoint 评测：模型显著退化

- **LoRA 提取**: 从 FSDP checkpoint 提取 504 个 LoRA 参数 (rank=16) → PEFT adapter 格式 → vLLM 加载
- **工具**: `scripts/extract_lora.py` + `scripts/convert_lora.py` — FSDP → safetensors + adapter_config.json
- **评测**: `scripts/run_post_eval.py` — LoRA adapter 推理 + 纠错率评估 (78 prompts, ~15min)

| 指标 | 基线 (Qwen3-4B) | 训练后 (ckpt step_10) | Delta |
|------|-----------------|----------------------|-------|
| 纠错率 | 0.6000 | 0.8846 | **+0.28 (变差)** |
| 直接通过 | 32 (40%) | 9 (11.5%) | -23 |
| 失败 | 12 (15%) | 35 (44.9%) | +23 |
| 平均纠错轮次 | 0.95 | 2.04 | +1.09 |

- **10/11 类别恶化**，仅 logical reasoning 改善
- **结论**: Phase 1 早期 checkpoint (step_10) 处于训练劣化阶段；Phase 2 checkpoint 丢失无法评估后期效果

### 框架代码完整评测

**全部 154 个单测通过** (6 个测试文件)，0 失败：

| 模块 | 测试数 | 状态 |
|------|--------|------|
| conversation_store (B0) | 23 | ✅ |
| orchestrator (A2) | 24 | ✅ |
| metrics (S5) | 27 | ✅ |
| rubric_evolver (B4) | 25 | ✅ |
| pipeline (C1) | 20 | ✅ |
| dashboard (C2) | 6 | ✅ |

**远程真实 API 验证** (`scripts/validate_modules.py`, 4 tests):
- JudgeExecutor 单 rubric 评分 ✅
- JudgeExecutor 合并多 rubric 评分 ✅
- RubricEvolver 维度提取 + 触发条件 + 统计 ✅
- Pipeline 训练配置生成 ✅

**Rubric Evolver 端到端验证** (`scripts/verify_evolver.py`):
- get_rubric_stats / check_and_evolve / archive_stale 全部正常

**修复的 Bug：**
- `pipeline.py`: `asyncio.to_thread()` 不能包装 async 方法 → 改为直接 `await`
- `validate_modules.py`: RubricScore 字段名错误 (`理由`→`总结`), `use_merged` 是构造参数不是方法参数
- `test_dashboard.py`: Windows GBK 编码 + 路径解析

**Pipeline e2e 远程 GPU 验证**: 5 prompts, 274s, correction rate 0.6, 正常完成

### 今日改动文件

| 文件 | 改动 |
|------|------|
| `trainable_openclaw/pipeline.py` | fix asyncio.to_thread for async methods |
| `tests/test_dashboard.py` | fix Windows GBK encoding + path resolution |
| `scripts/validate_modules.py` | 新建: judge + evolver + pipeline 4 项真实 API 验证 |
| `scripts/extract_lora.py` | 新建: FSDP checkpoint → LoRA weights 提取 |
| `scripts/convert_lora.py` | 新建: FSDP LoRA → PEFT safetensors 格式转换 |
| `scripts/run_post_eval.py` | 新建: LoRA adapter 纠错率评测 |
| `scripts/analyze_run.py` | 新建: 训练日志解析 + 分析报告生成 |
| `docs/validation_report.md` | 新建: 框架代码完整评测报告 |
| `runs/2026-06-03_42steps/` | 新建: 训练日志/分析/评测结果存档 |

---

## 2026/06/06

### 完整训练运行（104 steps，~16h GPU）

- **5 个阶段**，逐步调参：
  - Phase 1 (15 steps): debug，2p×2r=4, rlen=504 (截断)
  - Phase 2 (27 steps): 48p×4r=192, rlen=497 (截断)
  - Phase 3 (25 steps): 32p×4r=128, rlen=500 (截断，reward 暴跌至 0.031)
  - Phase 4 (6 steps): 过渡配置
  - Phase 5 (31 steps): **16p×4r=64, rlen=1925** (完整回复，修复成功)
- **关键发现**: `response_length` 默认 512 导致 75% 代码被截断。改为 4096 后 reward 从 0.06 跳到 0.67（5x 提升）
- **Phase 5 结果**: mean reward=0.588, 60.8% >0.5, loss=0.045 (非零，模型在学)
- **3 步移动平均**: 0.52-0.64 震荡，无单调上升趋势（31 步不够，或 lr=1e-5 太低）
- **Checkpoint**: step_10 (17:48) 和 step_20 (19:17) 均在 Phase 5 期间保存，LoRA 已提取为 PEFT 格式（126MB）
- **前 73 步浪费**: ~11.6h GPU 时间在截断回复上训练，效果极差
- **结论**: response_length 是最关键的杠杆；Qwen3-4B 容量有限，更大规模训练调优留到后续阶段

### 项目决策

- **暂停当前规模训练**：当前阶段聚焦功能快速开发，大规模训练调优留到后续
- **推进 Phase 4**：Agent 引擎集成是下一步核心工作
- **产物归档**: `reports/training_summary_2026-06-06.md` + 日志下载

### 今日改动文件

| 文件 | 改动 |
|------|------|
| `reports/training_summary_2026-06-06.md` | 新建: 完整训练分析报告 |
| `reports/phase3_train.log` | 下载: 服务端日志 (6.7MB) |
| `reports/serve_ppo_train.log` | 下载: 训练详情日志 (1.3MB) |
| `reports/generation_samples.json` | 下载: 生成样本缓存 (160KB) |
| `reports/start_train.sh` | 下载: 训练启动脚本 |
| `tmp/run_eval.py` | 新建: 通用测试集评测脚本 (generation + individual scoring) |
| `verl-main-0516/verl/trainer/config/rollout/rollout.yaml` | prompt_length 512→2048, response_length 512→4096, temperature 1.0→0.6 |
| `scripts/start_train.sh` | +prompt_length=2048, +response_length=4096, +max_model_len=8192, prompts_per_step=16, rollout_n=4 |

---

## 2026/06/09

### Phase 4: nanobot Agent 引擎集成

nanobot 源码 (v0.2.1) 调研完成，三大集成任务代码完成：

#### T1: nanobot 作为 serve_ppo 前端
- `trainable_openclaw/agent/nanobot_adapter.py`: 核心适配层
  - `NanobotAdapter.build_config()` → 生成 nanobot config JSON，`custom` provider 指向 serve_ppo API
  - `create_bot()` → 程序化创建 `Nanobot` 实例
  - `check_serve_ppo()` → cook serve_ppo 健康检查
  - `test_connection()` → 端到端连通性验证
- `scripts/start_nanobot.sh`: 一键启动脚本（生成 config + 检查 serve_ppo + 启动 nanobot gateway）
- 配置要点: `provider=custom`, `apiBase=http://localhost:8000/v1`, `apiKey=no-key`（本地服务无需认证）

#### T2: Agent rollout 生成器
- `trainable_openclaw/agent/rollout.py`: 使用 nanobot + 外部 LLM 生成训练 rollout
  - `NanobotRolloutGenerator.generate_rollouts()` — 每个 prompt 启动 N 个 nanobot agent，带 filesystem+shell 工具自我验证代码
  - `generate_simple()` — 无 agent 工具的纯 LLM 生成（更快，适合作数据增强）
  - `make_training_pool()` — 转换 rollout 输出为 serve_ppo `_training_pool` 格式
  - Agent 模板要求: 写代码 → 运行测试 → 修 bug → 返回最终代码
- 用途: serve_ppo 训练期间 vLLM 休眠，nanobot+DeepSeek 替代生成高质量代码

#### T3: 日志系统兼容
- `trainable_openclaw/agent/log_bridge.py`: nanobot SessionManager (JSONL) ↔ ConversationStore (SQLite) 双写桥
  - `wrap_session_manager()` — monkey-patch `SessionManager.save()`，写入 JSONL 同时同步 SQLite
  - `sync_from_nanobot()` — 离线批量导入 nanobot sessions → SQLite
  - `export_to_nanobot()` — 反向导出 SQLite session → nanobot JSONL 格式
  - `_load_nanobot_session()` / `_import_session()` — JSONL 解析器

#### 测试
- `scripts/test_phase4.py`: 6 项集成测试（config生成、serve_ppo健康检查、log同步、live wrap、rollout生成、training pool创建）
  - `--quick` 模式跳过 API 依赖测试
  - Windows 上 6/6 通过，154 已有单测全部通过（无回归）

#### 集成架构

```
  ┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
  │  nanobot    │────▶│  serve_ppo API   │────▶│  vLLM        │
  │  (gateway)  │     │  :8000/v1        │     │  Qwen3-4B    │
  │  :18790     │     └────────┬─────────┘     └──────────────┘
  └──────┬──────┘              │
         │                     │ (idle→training)
         ▼                     ▼
  ┌──────────────┐    ┌──────────────────┐
  │ SessionMgr   │    │ NanobotRollout   │
  │ (JSONL)      │    │ Generator        │
  └──────┬───────┘    │ + DeepSeek API   │
         │            └──────────────────┘
         ▼
  ┌──────────────┐
  │ LogBridge    │
  │ (SQLite)     │
  └──────────────┘
```

### 新增文件

| 文件 | 说明 |
|------|------|
| `trainable_openclaw/agent/__init__.py` | Phase 4 agent 模块初始化 |
| `trainable_openclaw/agent/nanobot_adapter.py` | T1: nanobot 适配层 |
| `trainable_openclaw/agent/rollout.py` | T2: Agent rollout 生成器 |
| `trainable_openclaw/agent/log_bridge.py` | T3: JSONL ↔ SQLite 日志桥 |
| `scripts/start_nanobot.sh` | 一键启动 nanobot + serve_ppo |
| `scripts/test_phase4.py` | 6 项集成测试 |

### 远程部署待办
1. 上传新增文件到远程 Linux GPU 机器
2. 安装 nanobot 依赖: `pip install loguru pydantic pydantic-settings typer rich tiktoken`
3. 启动 serve_ppo: `bash scripts/start_train.sh`
4. 启动 nanobot gateway: `bash scripts/start_nanobot.sh`
5. 验证: 浏览器访问 `http://<host>:18790/webui/`
6. 测试 rollout 生成: `python scripts/test_phase4.py`（需 DEEPSEEK_API_KEY）

### Git 状态
- 6 个新文件未提交
- 154 个已有测试全部通过，无回归
