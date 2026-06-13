# Interactive Evaluation System — Plan

> 日期: 2026-06-12 | 状态: 完成 (待 API key 实测)

## 目标

构建基于模拟用户的交互式评测系统。核心理念：评测逻辑与上线自进化逻辑完全一致——唯一区别是反馈来源（模拟用户 vs 真实用户）。

## 核心指标

- **平均完成轮数** (核心) — 越少越好，连续值，训练信号丰富
- **首次正确率** — % 任务 1 轮完成
- **纠错后完成率** — % 任务最终完成
- **放弃率** — % 超时未完成

## 架构

```
测试集 N 条 prompt
      │
      ▼
┌──────────────────────────────────────────────┐
│           InteractiveEvaluator               │
│                                              │
│  for each task:                              │
│    │                                         │
│    ├─ SimulatedUser.start() → initial req    │
│    │                                         │
│    └─ loop (max 10 rounds):                  │
│        ├─ Agent.run(user_msg)                │
│        │   → calls nanobot → mock tools      │
│        │   → returns text response           │
│        │                                     │
│        ├─ SimulatedUser.respond(agent_msg,   │
│        │      tool_results)                  │
│        │   → natural language feedback       │
│        │   → status: continue|complete|quit  │
│        │                                     │
│        └─ if complete/quit → record & break  │
│                                              │
│  → EvalReport { rounds, completed, ... }     │
└──────────────────────────────────────────────┘
```

## SimulatedUser 设计

### 方案：LLM 驱动的模拟用户

使用 DeepSeek API 扮演用户角色：
- System prompt: 用户画像 + 任务目标 + 评估标准
- Context: 完整对话历史
- Output: 自然语言反馈 + 结构化状态

### 为什么用 LLM 而非规则

- tau-bench 任务多样性强，规则无法覆盖
- 需要自然语言多样性（"不对" / "不是这个" / "我要的是..."）
- 需要理解部分正确的情况（做对了一半）
- 与真实用户行为一致 → 评测结果可迁移到上线

### SimulatedUser 决策逻辑

```
Agent 执行了工具调用 → 查看工具执行结果 →
  ├─ 完全符合预期 → "谢谢，正是我想要的" → complete
  ├─ 部分正确 → "X 是对的，但 Y 需要改一下" → continue
  ├─ 完全错误 → "不对，我要的是 Z" → continue
  └─ 无法完成 → "算了，我找人工客服吧" → give_up
```

## 产出文件

```
trainable_openclaw/evaluation/
├── simulated_user.py     # LLM 驱动的模拟用户
├── interactive_eval.py   # 交互式评测编排器
└── __init__.py           # 更新 exports

tests/
└── test_interactive_eval.py  # 测试
```

## 实现步骤

1. SimulatedUser — LLM-backed user simulator → verify: correct responses for mock conversations
2. InteractiveEvaluator — harness → verify: runs task loop, counts rounds
3. Agent adapter — pluggable agent (LLM API for now, trained model later) → verify: agent calls tools
4. Integration test — full loop on sample tasks → verify: produces EvalReport
