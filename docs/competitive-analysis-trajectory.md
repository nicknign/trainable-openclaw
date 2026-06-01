# 竞争分析：Trajectory.ai

> 分析日期：2026-06-01
> 来源：[新智元报道](https://mp.weixin.qq.com/s/fwHG46El4Jj3aocz5Tbw-Q) + [Trajectory 官方文档](https://docs.trajectory.ai)

---

## 一、Trajectory 是什么

**Trajectory** 是一家 2026 年 5 月 27 日走出隐身模式的 AI 基础设施公司，核心产品是"让模型越用越聪明"的持续学习平台。

### 关键信息

| 维度 | 详情 |
|------|------|
| 融资 | $1500万种子轮，投后估值 $1.15亿 |
| 领投 | Conviction |
| 个人投资者 | Jeff Dean (Google DeepMind)、李飞飞 (World Labs) |
| 团队 | 11人，来自 OpenAI/DeepMind/Apple/Meta |
| CEO | Ronak Malde（前 Windsurf AI 研究员，Windsurf 被 Google $24亿收购后进入 DeepMind） |
| 学术背书 | Rich Sutton（强化学习之父，图灵奖得主）在 NeurIPS 2025 做同名 keynote |

### 核心理念

> "今天最强的 AI 仍然是静态的。你昨天用的那个模型，今天还会犯同样的错。"

Trajectory 不卷大模型，而是做模型之外的"持续学习基础设施"——把 Cursor 用真实用户数据做 post-training 的成功秘密，做成所有企业都能用的标准化平台。

---

## 二、Trajectory 的技术方案

### 数据管线

```
企业产品日志 → SDK 接入 → Trajectory 4层格式 → 客户圈定训练数据 → 审批 → 训练
```

**Trajectory 4 层格式：**

```
Trajectory（整段对话）
  └─ Step（每轮累积快照 = 自包含训练样本）
       └─ Turn（一次用户与 Agent 的来回）
            └─ Message（单条消息）
```

每个 Step 是自包含训练样本：给定完整上下文 + Agent 接下来做了什么。

### SDK 设计要点

- **Trace 接入**：从 LangSmith 等可观测平台直接导入
- **BYO Data**：支持 CSV/JSONL/OpenAI/Anthropic/Vercel 格式
- **Telemetry 事件**：独立的 `TelemetryEvent` 原语，与对话轨迹通过 `trace_id` 关联
- **PII 脱敏**：可插拔的 transform 管道，数据不离开本地
- **幂等推送**：event_id 确定性哈希，支持 at-least-once 投递

### 训练方案

- **模型**：开源模型全量 post-training（不动 OpenAI/Anthropic 的权重）
- **触发**：手动/定时，最快**周级**更新
- **审批**：每次模型更新需客户评估和审批
- **合规**：SOC 2 认证

### 早期客户

| 客户 | 领域 | 场景 |
|------|------|------|
| Clay | GTM/销售 | 销售线索智能化 |
| Decagon | AI 客服 | 企业客服自动退货等 |
| Harvey | 法律 AI | 法律文书/判例分析 |

---

## 三、与 trainable-openclaw 的对比

### 3.1 定位差异

| 维度 | Trajectory | trainable-openclaw |
|------|-----------|-------------------|
| **定位** | 企业级 B2B SaaS 平台 | 开源 Agent 自进化推理引擎 |
| **目标用户** | 有 AI 产品的企业 | AI 开发者/研究者 |
| **商业模式** | 订阅制 SaaS | 开源（MIT） |
| **部署** | 云端多租户 | 单卡本地部署 |
| **核心理念** | 帮客户用自己的数据训练自己的模型 | 让 Agent 从交互中自主进化 |

### 3.2 技术对比

| 维度 | Trajectory | trainable-openclaw |
|------|-----------|-------------------|
| **推理引擎** | 未公开（推测 API 代理） | **veRL HYBRID 模式**（vLLM + FastAPI） |
| **训练算法** | 未公开（推测 SFT+RL） | **GRPO**（已验证 GSM8K 20-step E2E） |
| **模型底座** | 开源模型全量 post-training | Qwen3-4B + **LoRA** rank=16 |
| **触发机制** | 手动/定时，**周级** | **空闲自动检测 → 训练 → 恢复**，全自动 |
| **评估体系** | 客户自定义 + 人工审批 | **LLM Judge 自动 Rubric 打分**（20 条） |
| **数据来源** | 企业产品日志 | LMSYS 种子 + User Sim 纠错模拟 |
| **闭环速度** | 周级 | 分钟级（idle_timeout=30s） |
| **硬件需求** | 云端 | **单卡 RTX 4080 SUPER** |

### 3.3 自动化程度（核心差异）

```
Trajectory:
  企业数据 → SDK导入 → 人工圈定数据 → 客户审批 → 周级训练 → 部署

trainable-openclaw:
  用户对话 → 自动日志 → LLM自动Rubric → 自动GRPO打分 → 空闲触发训练 → 权重同步 → 恢复推理
  └──────────────────── 全自动闭环，无人值守 ────────────────────────┘
```

**Trajectory 的 B2B 模式决定了必须有审批环节**（客户不敢让 AI 自己改自己），而 **Agent 场景天然适合自动化**——用户不满意本身就是信号，不需要人工审批。这是 trainable-openclaw 的核心护城河。

---

## 四、可借鉴的设计

从 Trajectory 的 SDK 文档中提取了 5 个可借鉴的设计模式：

### 4.1 Step 自包含训练样本 ⭐⭐⭐

**借鉴**：Trajectory 的 Step 层设计——每个 Step 包含"截止当前轮的完整上下文 + Agent 下一步做了什么"，是一个天然的训练样本。

**当前状态**：trainable-openclaw 使用 `(bad, correction, good)` 三元组

**改进方向**：
```python
class Step:
    messages_so_far: list[Message]  # 完整上下文
    agent_action: Message           # Agent 做了什么
    user_feedback: str | None       # 用户反馈信号
    is_training_sample: bool        # 是否导出为训练数据
```

### 4.2 Telemetry 事件 + trace_id ⭐⭐⭐

**借鉴**：对话轨迹和产品遥测分离，通过 `trace_id` 关联。

**具体事件类型**：

| 事件 | 含义 | 训练价值 |
|------|------|---------|
| `user.corrected` | 用户纠正了 Agent | 高价值负样本 |
| `user.accepted` | 用户直接采纳 | 高价值正样本 |
| `user.abandoned` | 用户放弃对话 | 潜在负样本 |
| `agent.tool_call` | Agent 调用了工具 | 工具使用训练数据 |

### 4.3 Reward 结构化 ⭐⭐

**借鉴**：多维度 reward 组件化，每个组件有独立的 name/value/range/weight。

```python
@dataclass
class RewardComponent:
    name: str
    scaled_value: float  # [0, 1]
    score_range: tuple
    weight: float
```

### 4.4 BYO Data 适配器 ⭐⭐

**借鉴**：统一的 Message 适配器层，支持多种数据来源。

```python
MessageAdapter.from_openclaw(raw)
MessageAdapter.from_nanobot(raw)
MessageAdapter.from_openai_chat(raw)
```

### 4.5 Transform 管道 ⭐

**借鉴**：可插拔的数据处理管道（PII 脱敏、截断、过滤）。

### 改造优先级

| 优先级 | 借鉴项 | 改造成本 | 收益 |
|--------|--------|---------|------|
| P0 | Step 自包含训练样本 | 低 | 高 |
| P0 | Telemetry 事件 + trace_id | 中 | 高 |
| P1 | Reward 组件化 | 低 | 中 |
| P1 | BYO Data 适配器 | 低 | 中 |
| P2 | Transform 管道 | 低 | 低（生产才需要） |

---

## 五、trainable-openclaw 的差异化优势

### 5.1 自动化闭环（Trajectory 做不到）

```
trainable-openclaw: 空闲自动检测 → GRPO训练 → 权重同步 → 恢复推理
Trajectory:         等待客户提交数据 → 等待审批 → 手动触发训练
```

### 5.2 轻量化（单卡可跑）

Qwen3-4B + LoRA rank=16 在 RTX 4080 SUPER 上运行，无需云端 GPU 集群。

### 5.3 veRL 深度改造

不是简单包装 API，而是基于 veRL 的 HYBRID 模式做双模引擎，复用 `sleep/wake` + `CheckpointEngine` 的成熟机制。

### 5.4 开源 + 社区

Trajectory 是闭源 SaaS，trainable-openclaw 开源（MIT），可以吸引社区贡献和二次开发。

---

## 六、风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| Trajectory 开源 SDK | 中 | 低 | 核心壁垒在自动化闭环，不在数据格式 |
| 大厂做类似功能 | 中 | 中 | 护城河是 veRL 深度改造 + 空闲自动训练，大厂做 API 方案不会碰这个 |
| 模拟数据 vs 真实数据 gap | 中 | 中 | Phase 1.5 S4 反思模块 + Phase 4 生产评估 |
| 单卡性能瓶颈 | 低 | 低 | 验证通过后可通过 veRL 多卡扩展 |

---

## 七、结论

**Trajectory 的出现是对 trainable-openclaw 方向的市场验证。**

Jeff Dean、李飞飞、Rich Sutton 三个人同时押注"持续学习基础设施"方向，说明这不是一个小众想法，而是 AI 产业下一层基础设施的共识。

**trainable-openclaw 的差异化路径清晰：**

1. **更自动化** — 无人值守闭环 vs 人工审批
2. **更轻量** — 单卡可跑 vs 云端 SaaS
3. **更聚焦** — Agent 自进化 vs 通用企业平台
4. **更开放** — 开源 MIT vs 闭源 SaaS

**下一步行动**：
- [ ] P0: 改造 B0 对话日志为 Step 自包含训练样本格式
- [ ] P0: 新增 Telemetry 事件模块
- [ ] P1: Reward 输出组件化
- [ ] P1: BYO Data 适配器
- [ ] 保持 MVP 6/16 时间线不变

---

*文档版本: v1.0 | 作者: 蓝精灵 (基于 王烨 的讨论)*
