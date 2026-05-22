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

### 待解决问题：Qwen3-0.6B 推理输出乱码
- **现象**: `/v1/chat/completions` 返回的 content 为多语言乱码、随机 token
- **怀疑**: 模型文件下载不完整/损坏，或 vllm 0.18.1 与 Qwen3 架构不兼容
- **诊断方向**: 
  1. 检查模型文件完整性（`ls -la` / `du -sh`）
  2. 用 transformers 直接加载测试，排除 vllm 因素
  3. 必要时重新下载模型

### 关键路径
- SFTP: `connect.westd.seetacloud.com:29669` → `/data/wangye/trainable-openclaw`
- 模型目录: `/root/autodl-tmp/models/Qwen3-0.6B`
- conda: `/data/anaconda3`
