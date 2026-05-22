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

### 当前状态
- serve_ppo 支持 A1+A2：推理 + 空闲检测 + 训练编排
- 启动命令示例（含 A2 低阈值）:
  ```bash
  ... +trainer.idle_timeout=5 +trainer.min_samples=2
  ```
- train_step 为 stub（3s 延迟），真正训练待 A3 实现

### 关键路径
- SFTP: `connect.westd.seetacloud.com:29669` → `/data/wangye/trainable-openclaw`
- 模型目录: `/data/models/Qwen3-4B`
- conda: `/data/anaconda3`
- verl editable install: `/root/autodl-tmp/wangye/trainable-openclaw/verl-main-0516`（与 `/data/wangye/` 同文件系统，文件一致）
