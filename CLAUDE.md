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

远程 Linux GPU 机器 (AutoDL)，SSH 连接与 SFTP 使用相同端口。
详见 `.vscode/sftp.json` 以及 memory 中的 `remote_env.md`。

---

# 多 Agent 协作模式

本项目使用 5 个自定义 subagent 分工协作。由主 agent (Claude Code) 担任调度者，主动识别任务类型并派发给对应的专业 agent。

## Agent 分工表

| Agent | 模型 | 职责 | 触发场景 |
|-------|------|------|----------|
| **research-scout** | opus | ML 文献检索、数据集发现、实验设计建议 | 找论文、找数据集、技术调研、竞品分析 |
| **disciplined-coder** | sonnet | 代码编写/重构/调试，严格遵循本文件四大原则 | 非平凡代码改动、新功能实现、bug 修复、重构 |
| **e2e-code-tester** | sonnet | 集成测试、鲁棒性验证、跨组件端到端验证 | coder 完成一个逻辑模块后，验证整体功能和边界情况 |
| **research-experiment-planner** | sonnet | 算法分析、实验协议设计、训练诊断、超参数调优 | 设计实验方案、诊断训练问题、分析 reward 曲线 |
| **academic-content-writer** | sonnet | 学术论文、技术博客、社交媒体宣传 | 里程碑达成后撰写论文/博客/推文推广 |

## 协作模式

### 串行流水线（有依赖关系）

```
research-scout (调研) → research-experiment-planner (实验设计) → disciplined-coder (实现) → e2e-code-tester (验证)
```

### 并行分派（无依赖关系）

当任务之间相互独立时，同时启动多个 agent 并行工作。例如：
- 文献调研 + 数据集搜索 → 同时启动 2 个 research-scout
- 多个模块的编码 → 同时启动多个 disciplined-coder

### 主动调度原则

主 agent 应**主动识别**当前任务适合哪个专业 agent，而非等待用户指令：
- 用户要求写代码 → 自动派 disciplined-coder
- coder 完成模块 → 自动派 e2e-code-tester
- 用户讨论实验方向 → 自动派 research-experiment-planner
- 用户问"有什么新方法" → 自动派 research-scout
- 重大功能完成 → 主动派 academic-content-writer 撰写宣传

## 项目文档索引

- 路线图与进度: `docs/roadmap.md`
- MUA-RL 调研: `docs/mua-rl_research.md`
- 远程环境: memory `remote_env.md`
- 项目状态: memory `project-phase4-status.md`
