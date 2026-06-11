# CLAUDE.md — trainable-openclaw

## 项目目标

训练 Qwen3-4B 使其能熟练调用 nanobot skill 完成客服业务任务（tau-bench 场景）。
上线后通过分层 reward 收集真实用户反馈，维持自进化训练闭环。

## 功能设计流程

```
                        ┌──────────────────────────┐
                        │   tau-bench 任务定义      │
                        │   (164 airline + retail)  │
                        └────────────┬─────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────┐
│                    Phase 1: 数据准备                          │
│                                                              │
│  原始数据 ──→ convert ──→ filter/split ──→ validate          │
│  (5源/2080条)   统一格式     train 867 / test 416   25/25✓    │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                Phase 2-3: 训练 (远程 Linux GPU)               │
│                                                              │
│  SFT 冷启动 ──→ GRPO 训练                                    │
│  Qwen3-4B      164 prompts, 3-layer reward                   │
│  LoRA rank=16   16p×4r=64, lr=1e-5                           │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│               Phase 4: 评测 + 上线自进化                      │
│                                                              │
│  ┌──────────┐   ┌───────────┐   ┌────────────────────┐      │
│  │ Layer 1  │   │  Layer 2  │   │     Layer 3        │      │
│  │ 确定性验证│   │ 用户反馈  │   │   LLM Judge        │      │
│  │ (免费)   │   │ 信号提取  │   │   (低频调用)       │      │
│  │ 工具格式 │   │ (免费)    │   │   沟通质量评分      │      │
│  │ 执行结果 │   │ 正/负反馈 │   │   安全边界检查      │      │
│  │ 危险拦截 │   │ 纠错行为  │   │                     │      │
│  └────┬─────┘   └────┬──────┘   └─────────┬──────────┘      │
│       └──────────────┼────────────────────┘                  │
│                      ▼                                       │
│          final = 0.5×L1 + 0.3×L2 + 0.2×L3                   │
│                      │                                       │
│                      ▼                                       │
│             真实用户反馈 → 追加训练池 → 触发再训练 → 权重更新  │
└──────────────────────────────────────────────────────────────┘
```

## 里程碑总览

| 阶段 | 状态 | 内容 | 产出 |
|------|------|------|------|
| Phase 1: 数据准备 | ✅ 完成 | 数据集下载、mock工具、反馈模块、格式转换、验证 | 867 train / 416 test / 164 prompts |
| Phase 2: SFT | ⬜ 待 GPU | Qwen3-4B + LoRA 冷启动 | 工具调用格式正确率 > 60% |
| Phase 3: GRPO | ⬜ 待 GPU | 分层 reward 训练 | 任务完成率 > SFT +5% |
| Phase 4: 评测+自进化 | 🟡 部分完成 | 标准评测 + 上线自进化循环 | T1 集成完成，T2-T4 待远程 |

---

# Claude Agent 使用规范

## Agent 分工表

| Agent | 模型 | 职责 | 触发方式 |
|-------|------|------|----------|
| **research-experiment-planner** | sonnet | 制定 `plans/{name}_plan.md`、算法分析、实验设计、训练诊断 | 新功能/实验开始前由主 agent 派发 |
| **research-scout** | opus | ML 文献检索、数据集发现 | planner 在制定 plan 时通过消息调用，或主 agent 需要文献时触发 |
| **disciplined-coder** | sonnet | 按 plan 编码实现、编写单元测试、Linux 远程调试 | 收到 planner 的 handoff 后开始 |
| **e2e-code-tester** | sonnet | 端到端测试、集成验证、bug 报告 | 收到 coder 的 task_request 后开始 |
| **academic-content-writer** | sonnet | 学术论文、技术博客、社交媒体宣传 | **仅由主 agent 触发**，不参与日常循环 |

## 开发工作流

### 1. Plan First — 制定计划

**任何功能实现前，必须先由 planner 输出 `plans/{feature_name}_plan.md`。**

```
主 agent 提出需求
      │
      ▼
research-experiment-planner
      │
      ├─ 需要文献调研 → task_request → research-scout
      │                  scout 返回 findings → 融入 plan
      │
      └─ 输出: plans/{feature_name}_plan.md
           │
           └─ handoff → disciplined-coder
```

Plan 文档需包含：目标与背景、技术方案、实现步骤与验证标准、依赖与风险。

### 2. 编码与测试循环

```
disciplined-coder                     e2e-code-tester
      │                                      │
      ├─ 按 plan 编码 + 写单元测试             │
      │                                      │
      ├─ task_request ──────────────────────→│ (通知测试)
      │                                      ├─ 编写 e2e 测试
      │                                      ├─ 运行完整功能测试
      │                                      │
      │           ←── reply ────────────────┤ (测试通过 ✓)
      │                                      │
      │           ←── task_request ─────────┤ (发现 bug ✗)
      │  (修复 bug)                           │
      │  task_request ──────────────────────→│ (再次送测)
      │                                      │
      └── 循环直至全部测试通过 ───────────────┘
```

- coder **禁止跳过 tester 直接声称完成**
- tester 发现 bug 后发送详细信息（复现步骤、严重程度、涉及文件）
- 每次修复后重新走完整测试流程

### 3. Linux 远程实验

```
disciplined-coder ←→ research-experiment-planner
    (调试 & 修复)          (执行实验 & 观察 & 分析)
```

- 代码部署到 Linux 后，coder 负责主要调试
- planner 执行实验，观察训练曲线和收敛情况
- 发现问题 → planner 反馈给 coder → 修改代码 → 重新实验
- **每次实验的观察结论写入对应 `plans/` 文档的 "实验结果" 段落**

### 4. 文献撰写

**仅由主 agent 触发**。主 agent 判断里程碑达成后，启动 writer 收集实验结果和项目进展进行撰写。

## 子 Agent 间消息通信

通过 `.claude/messages/{agent-name}/inbox/` 文件消息系统：

| 消息类型 | 用途 |
|----------|------|
| `task_request` | 派发任务（如 coder→tester: "测试这个模块"） |
| `status_update` | 汇报进展（如 scout→planner: "找到相关论文"） |
| `question` | 请求澄清 |
| `handoff` | 移交工作（如 planner→coder: "计划已完成，开始实现"） |
| `reply` | 回复消息 |

**规则：**
- 每个 agent 在**会话开始时**检查 inbox 未读消息
- 完成任务时**主动通知下游 agent**
- 消息文件**绝不删除**（审计追溯）

**CLI 工具：** `python .claude/agent_message.py`
```bash
python .claude/agent_message.py send --to AGENT --type TYPE --subject "..." --body "..."
python .claude/agent_message.py check --agent AGENT --unread-only
python .claude/agent_message.py list-agents
```

详细协议：`.claude/messages/PROTOCOL.md`

## 行为准则（四大原则）

**这些原则约束所有 agent，尤其是 disciplined-coder。**

### 1. 先想后写
明确假设，有歧义先问。有多种方案时先列出而非静默选择。

### 2. 简洁优先
最小代码解决问题。不为单次使用创建抽象。不需要的"灵活性"一律不写。

### 3. 外科手术式修改
只改必须改的。不"顺手优化"相邻代码。只清理自己引入的孤立代码。

### 4. 目标驱动
把任务转化为可验证目标。"修 bug"→"写复现测试→修复→确认通过"。定义明确的成功标准。

---

## 硬件环境

远程 Linux GPU 机器 (AutoDL, RTX 3090 24GB)，SSH 连接。
详见 `.vscode/sftp.json` 以及 memory `remote_env.md`。

## 项目文档索引

| 文档 | 说明 |
|------|------|
| `docs/plan.md` | 完整实验计划（3-layer reward + 4 phases） |
| `docs/roadmap.md` | 路线图与历史进度 |
| `docs/tau_bench_format.md` | tau-bench 数据格式分析 |
| `docs/mua-rl_research.md` | MUA-RL 调研报告 |
| `.claude/messages/PROTOCOL.md` | 子 agent 消息通信协议 |
| memory `remote_env.md` | 远程服务器连接信息 |
| memory `project-phase4-status.md` | 项目当前状态 |

## 当前进度 (2026-06-12)

- **Phase 1 完成**: 215/215 测试通过，数据流水线就绪（867 train / 416 test / 164 prompts）
- **Phase 2-3**: 待远程 Linux GPU 可用
- **Phase 4 T1**: nanobot 集成完成
- **Agent 通信**: 消息系统已部署，agent 定义已更新
