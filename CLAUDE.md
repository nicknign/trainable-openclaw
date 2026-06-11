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

本项目使用 5 个自定义 subagent 分工协作。由主 agent (Claude Code) 担任调度者，但子 agent 之间也可通过消息系统直接通信。

## Agent 分工表

| Agent | 模型 | 职责 | 触发场景 |
|-------|------|------|----------|
| **research-experiment-planner** | sonnet | 制定开发计划、算法分析、实验设计、训练诊断 | 新功能/新实验开始前制定 plan，诊断训练问题 |
| **research-scout** | opus | ML 文献检索、数据集发现 | planner 需要调研时调用，或主 agent 需要文献时触发 |
| **disciplined-coder** | sonnet | 按 plan 编码实现、编写单元测试、Linux 调试 | 有 plan 后实现代码，收到 bug report 后修复 |
| **e2e-code-tester** | sonnet | 端到端测试、集成验证、bug 报告 | coder 完成模块后被通知，编写并执行 e2e 测试 |
| **academic-content-writer** | sonnet | 学术论文、技术博客 | 主 agent 触发，收集实验结果和项目进展后撰写 |

## 开发工作流

### 1. 制定计划（Plan First）

```
主 agent 提出需求 → research-experiment-planner 制定 plan
                      │
                      ├─ 需要文献调研 → 发 task_request 给 research-scout
                      │                  research-scout 返回 status_update
                      │
                      └─ 输出: plans/{feature}_plan.md
```

每个具体功能实现前，**必须先由 planner 写出 `plans/xxx_plan.md`**，包含：
- 目标与背景
- 技术方案与架构
- 实现步骤与验证标准
- 依赖与风险

### 2. 编码与测试循环

```
planner ──handoff──→ disciplined-coder (按 plan 实现 + 写单测)
                            │
                            │ 完成 → task_request → e2e-code-tester
                            │                            │
                            │        测试通过 ←── reply ─┤
                            │                            │
                            │   ←── task_request ────────┘ (发现 bug)
                            │   (修复 bug) 
                            │   ── task_request ────────→ (再次送测)
                            │                            │
                            └── 循环直至测试全部通过 ─────┘
```

- coder 按 plan 编写代码 + 单元测试
- coder 完成后通过消息系统通知 tester
- tester 编写端到端测试，运行完整功能测试
- 有 bug → tester 发 `task_request` 给 coder → coder 修复 → 再次送测
- **循环往复直至所有测试通过**

### 3. Linux 远程实验

```
disciplined-coder (Linux 调试)
       ↕ (bug 修复 / 代码调整)
research-experiment-planner (执行实验、观察结果、分析)
```

- 代码部署到 Linux 后，coder 负责主要调试
- experiment-planner 执行实验并观察训练曲线、收敛情况
- 发现问题 → planner 反馈给 coder 修改代码
- 持续迭代优化

### 4. 文献撰写

```
主 agent 触发 → academic-content-writer
                  │
                  ├─ 收集各 agent 的实验结果
                  ├─ 汇总项目进展
                  └─ 撰写论文/博客/推文
```

文献 agent **仅由主 agent 触发**，不参与日常开发循环。

## 并行分派

当任务之间相互独立时，同时启动多个 agent 并行工作。

## 子 Agent 间直接通信

通过 `.claude/messages/{agent-name}/inbox/` 文件消息系统：

**消息类型：** `task_request` | `status_update` | `question` | `handoff` | `reply`

**工具：**
```bash
python .claude/agent_message.py send --to AGENT --type TYPE --subject "..." --body "..."
python .claude/agent_message.py check --agent AGENT --unread-only
python .claude/agent_message.py list-agents
```

详细协议：`.claude/messages/PROTOCOL.md`

## 项目文档索引

- 路线图与进度: `docs/roadmap.md`
- MUA-RL 调研: `docs/mua-rl_research.md`
- 远程环境: memory `remote_env.md`
- 项目状态: memory `project-phase4-status.md`

---

## 2026/06/12

### Phase 1 完成：数据准备流水线

**P1.4 — 反馈收集模块**: `trainable_openclaw/feedback/`
- `signal_extractor.py`: Layer 2 用户反馈信号提取（20+ 中英文正则模式）
- `deterministic_verifier.py`: Layer 1 确定性验证（工具格式/执行结果/危险操作/任务完成）
- `reward_combiner.py`: 三层 reward 加权组合（L3=None 时权重自动重分配）
- `tests/test_feedback.py`: 61 测试通过

**P1.5 — 格式转换**: `scripts/convert_tau_bench.py`
- 支持 5 个数据源：GPT-4o/Sonnet × airline/retail + APIGen
- 统一训练样本格式（plan.md 定义），自动合成缺失 tool_call_id/name

**P1.6 — 过滤拆分**: `scripts/filter_split_tau_bench.py`
- Reward ≥ 0.5 过滤，domain-aware task_id 防泄漏
- 使用 tau-bench 官方 train/test 分区

**P1.7 — 数据验证**: `scripts/validate_tau_bench_data.py`
- 25 项自动检查（UUID/字段完整性/tool调用格式/跨集泄漏），全部通过

**产出**:
- 训练集 `data/tau_bench/train.jsonl`: 867 样本 / 100 唯一任务
- 测试集 `data/tau_bench/test.jsonl`: 416 样本 / 55 唯一任务
- GRPO prompts `data/tau_bench/grpo_prompts.jsonl`: 164 任务 (104 train + 60 test)
- 零 train/test 泄漏，25/25 验证通过

**测试**: 215/215 全部通过（154 已有 + 61 新增，零回归）

### 待办
- Phase 2 (SFT): 需远程 Linux GPU 可用
- Phase 3 (GRPO): 同上
- Phase 4 (评测+自进化): Agent 引擎集成完成 (T1/T2/T3)，待远程部署联调
