# Phase 3 改进：Trajectory.ai 竞争分析借鉴

> 日期：2026-06-01
> 来源：[竞争分析：Trajectory.ai](./competitive-analysis-trajectory.md)

---

## 一、改动概览

从 Trajectory.ai 的 SDK 设计中提取了 3 个模式，应用到 trainable-openclaw：

| 优先级 | 借鉴项 | 状态 | 改动文件 |
|--------|--------|------|---------|
| P1 | Reward 组件化 — rubric 加权评分 | ✅ 已完成 | judge.py, reward_bridge.py, serve_ppo.py |
| P0 | Telemetry 事件 + trace_id 反馈闭环 | ✅ 已完成 | conversation_store.py, api.py |
| P0 | Step 自包含训练样本导出 | ✅ 已完成 | conversation_store.py |

---

## 二、Reward 加权 (P1)

### 改动

`JudgeExecutor.compute_grpo_rewards()` 新增 `weights` 参数：

```python
# 之前：所有 rubric 等权
rewards = judge.compute_grpo_rewards(results, reward_mode="mean")

# 之后：可指定 rubric 权重
rewards = judge.compute_grpo_rewards(
    results, reward_mode="mean",
    weights=[0.4, 0.3, 0.1, 0.1, 0.1],  # 5 条 rubric 对应权重
)
```

### 透传路径

```
serve_ppo config → training_data["rubric_weights"]
  → RewardBridge(rubric_weights=...)
    → JudgeExecutor.compute_grpo_rewards(weights=...)
```

### 配置示例

```bash
+trainer.trajectory.rubric_weights=[0.4,0.3,0.1,0.1,0.1]
```

### 涉及文件

- `trainable_openclaw/evaluation/judge.py:239-276` — `compute_grpo_rewards` 加权计算
- `trainable_openclaw/training/reward_bridge.py:39-66` — `__init__` 接收 `rubric_weights`
- `verl-main-0516/verl/trainer/serve_ppo.py:340-347` — RewardBridge 构造传参
- `verl-main-0516/verl/trainer/serve_ppo.py:1056-1058` — 训练数据注入

---

## 三、Telemetry 事件 + /v1/feedback 端点 (P0)

### 数据模型

新增 `telemetry_events` 表（SQLite，WAL 模式）：

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增主键 |
| session_id | TEXT FK | 关联 sessions 表 |
| trace_id | TEXT | 关联 chat completion 的 request_id |
| event_type | TEXT | `user.accepted` / `user.corrected` / `user.abandoned` |
| rating | INTEGER | 1-5 评分 |
| correction | TEXT | 用户纠错意见（可选） |

### API 端点

**`POST /v1/feedback`**

```json
// Request
{
    "session_id": "abc123",     // 或 trace_id: "chatcmpl-..."
    "rating": 3,
    "correction": "请用中文回答"  // 可选
}

// Response
{
    "status": "ok",
    "event_id": 42,
    "event_type": "user.corrected"
}
```

**事件类型推导规则：**

| 条件 | event_type |
|------|-----------|
| rating >= 4 | `user.accepted` |
| 有 correction 文本 | `user.corrected` |
| rating == 1 且无 correction | `user.abandoned` |

### trace_id 链路

```
POST /v1/chat/completions
  → response.session_id = "abc123"
  → 消息 metadata = {"trace_id": "chatcmpl-a1b2c3"}
  → POST /v1/feedback {trace_id: "chatcmpl-a1b2c3", rating: 5}
    → 查找 session_id → 写入 telemetry_events
```

`ChatCompletionResponse` 新增 `session_id` 字段，客户端可直接引用该值发起 feedback 请求。

### 新增方法

- `ConversationStore.record_telemetry(session_id, event_type, ...)` — 写入事件
- `ConversationStore.get_telemetry(session_id?, event_type?)` — 查询事件
- `ConversationStore.get_telemetry_stats()` — 聚合统计

### 涉及文件

- `trainable_openclaw/logging/conversation_store.py:61-73` — telemetry_events DDL
- `trainable_openclaw/logging/conversation_store.py:291-342` — 3 个 CRUD 方法
- `trainable_openclaw/server/api.py:84-104` — FeedbackRequest / FeedbackResponse 模型
- `trainable_openclaw/server/api.py:76-77` — ChatCompletionResponse 新增 session_id
- `trainable_openclaw/server/api.py:207-218` — chat 响应返回 session_id + trace_id
- `trainable_openclaw/server/api.py:244-285` — submit_feedback 端点

---

## 四、Step 自包含训练样本导出 (P0)

### 概念

Trajectory 的 Step 层设计：每个 Step 是"截止当前轮的完整上下文 + Agent 下一步做了什么"，是天然的训练样本。

### 实现

`ConversationStore.export_steps()` 从对话历史切分 Step：

```python
steps = store.export_steps(
    session_id=None,         # 过滤特定 session，None = 全部
    limit=100,               # 最大返回数
    with_feedback_only=True, # 只要带反馈信号的训练样本
)

# 返回结构
{
    "messages_so_far": [           # 截止用户消息的完整上下文
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ],
    "agent_action": {              # Agent 在本轮的回复
        "role": "assistant",
        "content": "A2",
        "token_count": 42,
        "latency_ms": 500.0,
    },
    "feedback": {                  # 关联的遥测事件（如有）
        "event_type": "user.corrected",
        "rating": 3,
        "correction": "变量名不表意",
    },
    "is_training_sample": true,   # 有反馈即为训练样本
    "session_id": "abc123",
    "user_id": "alice",
    "model": "qwen3-4b",
}
```

### 与现有训练数据的对比

| 维度 | 旧三元组 (bad, correction, good) | 新 Step 格式 |
|------|--------------------------------|-------------|
| 上下文 | 只有单轮 prompt | 完整对话上下文 |
| 反馈来源 | User Sim 模拟 | 真实用户 telemetry |
| 可训练性 | 需要 S2 预处理 | 直接可用 |
| 数据量 | 手工生成，有限 | 随用户使用自动增长 |

### 涉及文件

- `trainable_openclaw/logging/conversation_store.py:344-418` — export_steps 方法

---

## 五、测试覆盖

| 模块 | 测试数 | 本次新增 |
|------|--------|---------|
| ConversationStore (telemetry) | 6 | 写入/查询/过滤/纠错/级联删除/空表 |
| ConversationStore (export_steps) | 8 | 空表/单轮/多轮/带反馈/过滤/限数/元数据 |
| API (Feedback models) | 3 | 请求/响应/trace_id 变体 |
| API (Feedback endpoint) | 5 | accepted/corrected/abandoned/缺参/无store |
| RewardBridge (已有) | 7 | 未变，全部通过 |
| Orchestrator (已有) | 24 | 未变，全部通过 |
| CorrectionRate (已有) | 9 | 未变，全部通过 |
| **总计** | **105** | **+22** |

---

## 六、与 Trajectory 的差距

| 维度 | Trajectory | trainable-openclaw（当前） |
|------|-----------|---------------------------|
| Reward 组件化 | 多组件 name/value/range/weight | ✅ 已支持 rubric_weights |
| Telemetry 事件 | 4 种标准事件 + SDK 埋点 | ✅ 3 种事件 + /v1/feedback 端点 |
| Step 训练样本 | 4 层格式 + SDK 自动切分 | ✅ export_steps() |
| BYO Data 适配器 | OpenAI/Anthropic/Vercel 等 | ⬜ 未做（P2，生产时再说） |
| Transform 管道 | PII 脱敏/截断/过滤 | ⬜ 未做（P2，有真实用户才需要） |
| 数据来源 | 企业产品日志 SDK | API 反馈端点（自动增长） |

---

*文档版本: v1.0 | 相关文档: [competitive-analysis-trajectory.md](./competitive-analysis-trajectory.md)*
