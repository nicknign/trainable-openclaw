# CLAUDE.md — trainable-openclaw

## 项目目标

训练 Qwen3-4B 使其能熟练调用 nanobot skill 完成客服业务任务（tau-bench 场景）。
上线后通过分层 reward 收集真实用户反馈，维持自进化训练闭环。

## 总体规划与路线图

> **新加入的 subagent 必须先读这些文档建立全局视角。**

| 文档 | 用途 | 何时读 |
|------|------|--------|
| `docs/roadmap.md` | 路线图、历史进度、各阶段状态 | **必读** — 了解项目全貌 |
| `docs/plan.md` | 完整实验计划：3-layer reward + 4 phases | 做训练相关任务前 |
| `docs/DESIGN.md` | 系统架构设计 | 做架构级改动前 |
| `docs/tau_bench_format.md` | tau-bench 数据格式说明 | 做数据处理前 |
| `docs/serve_guide.md` | vllm + nanobot 服务部署指南 | 做远程部署前 |
| `docs/code_guide.md` | 代码规范与模块说明 | 写代码前 |
| `docs/mua-rl_research.md` | MUA-RL 调研报告 | 做训练方法调研前 |
| `plans/` | 各功能的详细实施计划 | 执行具体任务前找对应 plan |

## 当前功能设计流程

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
│                Phase 2: GRPO 训练 (远程 Linux GPU)            │
│                                                              │
│  Qwen3-4B + LoRA rank=16                                     │
│  164 prompts, 3-layer reward, 16p×4r=64                      │
│  直接用 GRPO，无需 SFT — Layer 1 提供格式学习信号              │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│               Phase 3: 评测 + 上线自进化                      │
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
| Phase 2: GRPO | ⬜ 待 GPU | Qwen3-4B + LoRA，164 prompts，3-layer reward | 任务完成率 > baseline |
| Phase 3: 评测+自进化 | 🟡 部分完成 | 标准评测 + 上线自进化循环 | T1 集成完成 |

---

# Claude Agent 使用规范

## 角色定位

| 角色 | 担任者 | 职责 |
|------|--------|------|
| **Planner / 总控** | 主 agent (Claude Code) | 制定 `plans/`、掌控全局进度、维护项目记忆、决策调度 |
| **Implementer** | disciplined-coder | 编码实现 + 单元测试 + Linux 远程部署调试 + 训练执行 + 实验分析 + 结果验证 |
| **Literature Scout** | research-scout | ML 文献检索、数据集调研 |
| **Tester** | e2e-code-tester | 端到端集成测试、bug 报告、回归验证 |
| **Writer** | academic-content-writer | 论文/博客/推文，仅主 agent 触发 |

> **2026-06-12 合并**: research-experiment-planner 已并入 disciplined-coder。分离实验 agent 的失败教训——不读代码就写脚本，重复造轮子。编码和实验共享同一代码库知识，分开徒增协调成本。

## Agent 分工表

| Agent | 模型 | 职责 | 触发方式 |
|-------|------|------|----------|
| **disciplined-coder** | sonnet | 编码实现、单元测试、Linux 远程部署调试、训练执行、实验分析、结果验证 | 主 agent 派发 plan 后开始，或直接派发实验/调试任务 |
| **research-scout** | opus | ML 文献检索、数据集发现 | 主 agent 需要文献调研时触发 |
| **e2e-code-tester** | sonnet | 端到端测试、集成验证、bug 报告 | 收到 coder 的 task_request 后开始 |
| **academic-content-writer** | sonnet | 学术论文、技术博客、社交媒体宣传 | **仅由主 agent 触发**，不参与日常循环 |

## 何时派发 Subagent vs 主 Agent 自己做

**核心原则：能写成 spec 的独立任务派发，需要摸着石头过河的自己做。**

### 适合派发 Subagent（有明确输入/输出边界）

| 场景 | Agent | 原因 |
|------|-------|------|
| 按 plan 写独立模块 | disciplined-coder | 有完整设计文档，不需试错 |
| E2E 集成测试 | e2e-code-tester | 明确输入（代码模块）+ 明确输出（pass/fail/bug report） |
| ML 文献/数据集搜索 | research-scout | 一次性检索，返回结果即可 |
| 论文/博客撰写 | academic-content-writer | 给定素材和里程碑，无迭代 |
| 多个互不依赖的独立任务 | 并行派发 | 节省时间 |

### 必须主 Agent 亲自做（需要判断和快速试错）

| 场景 | 原因 |
|------|------|
| **交互式调试** | 读日志 → 改参数 → 重跑，循环快。Subagent 卡在 thinking 阶段不产出 |
| **探索性环境检查** | "看看环境有什么问题"——没有明确边界，agent 迷失在读文件中 |
| **依赖实时反馈的操作** | 启动服务等 5 分钟，agent 不知道在等还是卡了，主 agent 可直接等 |
| **需要判断和决策的任务** | agent 不敢做决定，反复读代码而不行动 |
| **每一步依赖上一步结果** | 串行任务，subagent 的沟通 overhead 高于直接做 |

### 判断标准

```
✓ 派发: 任务有完整的书面 spec → subagent
✗ 自己做: 任务需要"摸着石头过河" → 主 agent
```

### Subagent 失败模式记录

- **2026-06-12**: experiment agent 不读已有代码就写脚本（忽略 `start_experience.sh`），已合并入 coder
- **2026-06-12**: coder agent 卡死，0 行输出——调试任务不能派 subagent

## 开发工作流

### 1. Plan First — 主 agent 制定计划

**任何功能实现前，主 agent 先输出 `plans/{feature_name}_plan.md`。**

```
主 agent (Planner)
      │
      ├─ 分析需求，参考 CLAUDE.md / memory / docs/plan.md
      │
      ├─ 需要文献调研 → 启动 research-scout 检索
      │                  scout 返回 findings
      │
      └─ 输出: plans/{feature_name}_plan.md
           │
           └─ 派发 disciplined-coder (编码 + 实验一体化)
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

**disciplined-coder 全权负责**——编码、部署、启动服务、执行训练、监控曲线、诊断问题、修改代码、重新实验，全在一个 agent 内闭环。

- 使用 `scripts/start_experience.sh` 启动 vllm + nanobot 服务栈
- 使用 `scripts/autodl_sync.py` 同步代码和数据到远程
- **每次实验的观察结论写入对应 `plans/` 文档的 "实验结果" 段落**
- **读代码优先**: 执行任何实验前必须读完相关代码和已有脚本

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

远程 Linux GPU 机器 (AutoDL, RTX 4090 48GB)，SSH 连接。
同步工具: `scripts/autodl_sync.py`，详见 `.vscode/sftp.json` 以及 memory `remote_env.md`。

## 其他参考

| 文档 | 说明 |
|------|------|
| `.claude/messages/PROTOCOL.md` | 子 agent 消息通信协议 |
| memory `remote_env.md` | 远程服务器连接信息 |

## 当前进度 (2026-06-12)

- **Phase 1 完成**: 215/215 测试通过，数据流水线就绪（867 train / 416 test / 164 prompts）
- **Phase 2-3**: 待远程 Linux GPU 可用
- **Phase 4 T1**: nanobot 集成完成
- **Agent 通信**: 消息系统已部署，agent 定义已更新
