# tau-bench Agent 训练实验计划

> 版本: v1.1 | 日期: 2026-06-12 | 状态: 执行中

## 目标

用 tau-bench 数据集训练 Qwen3-4B，使其能熟练调用 nanobot skill 完成客服业务任务。上线后通过分层 reward 收集真实用户反馈，维持自进化训练闭环。

## 核心设计：分层 Reward 架构

### 问题

模拟环境有 ground truth（DB 状态比对），但上线后没有"订单数据库"来验证——rubric LLM judge 成本高且主观，不能作为主要 reward 来源。

### 三层 Reward 设计

```
Layer 1 — 确定性验证 (客观, 零 API 成本)
  ├─ 工具调用 JSON Schema 校验 (格式是否正确)
  ├─ exec 返回 exit code (0 = 成功)
  ├─ write_file 后文件存在性检查
  ├─ 危险操作拦截 (rm -rf / 等)
  └─ 任务完成标记 (agent 调用了 complete_goal)
  → reward: binary 0/1, 自动计算

Layer 2 — 用户反馈信号 (客观, 零 API 成本, 需提取)
  ├─ 显式正反馈: "谢谢" "完美" "正是我想要的"
  ├─ 显式负反馈: "不对" "再试一次" "不是这个"
  ├─ 隐式负反馈: 用户重新问同一问题 / 直接关闭会话
  ├─ 纠错行为: 用户指出具体错误 → 类似现有纠错管线
  └─ 会话完成度: 用户说了"再见" vs 中途离开
  → reward: binary or ordinal, 自动提取

Layer 3 — LLM Judge (主观, 有 API 成本, 仅必要时调用)
  ├─ 沟通质量: 是否确认了用户接受结果
  ├─ 不确定时的询问行为: "您是要 X 还是 Y？"
  ├─ 信息充分性: 执行前是否告知用户将要做什么
  └─ 安全边界: 拒绝危险操作时是否给出了解释
  → reward: continuous 0-1, 低频调用 (非每次生成)
```

### 与旧方案的对比

| 维度 | 旧 (纯 rubric judge) | 新 (分层 reward) |
|------|---------------------|-------------------|
| 主要 reward 源 | LLM judge 主观评分 | Layer 1 确定性验证 |
| API 成本 | 每步数百次 judge 调用 | 零到低频 |
| 客观性 | 有噪声，易被 reward hack | 客观确定性 |
| 上线自进化 | 依赖 judge 质量 | Layer 2 真实反馈驱动 |
| Judge 角色 | 核心评分 | 辅助（仅沟通质量维度） |
| Rubric 系统 | 被替代 | 保留，退到 Layer 3 |

---

## 数据源

### 模拟训练数据

| 数据源 | 数量 | 用途 |
|--------|------|------|
| `sierra-research/tau2-bench` 历史轨迹 | ~860 条 (reward=1 的 ~500 条) | SFT 冷启动 |
| `amityco/apigen-tau-bench-split-turn` | 46,127 条 | SFT 冷启动（主要） |
| tau2-bench tasks.json | 164 任务定义 | GRPO 训练 prompt + 评测 |

### 真实反馈数据（上线后）

| 来源 | 信号类型 | 收集方式 |
|------|---------|---------|
| 工具执行结果 | Layer 1 确定性验证 | 自动记录（已有 ConversationStore） |
| 用户消息解析 | Layer 2 显式/隐式反馈 | `trainable_openclaw/feedback/` (新增) |
| LLM Judge | Layer 3 沟通质量 | 低频异步调用 |

---

## 数据格式设计

### 统一训练样本格式 (Standard Training Sample)

所有训练数据（模拟 + 真实反馈）统一为此格式：

```json
{
  "id": "sample_uuid",
  "source": "taubench_retail | apigen | real_user_feedback",
  "task_id": "retail_task_3",
  "context": {
    "system_prompt": "You are a retail agent. Current time: 2024-05-15...",
    "tools": [{ "type": "function", "function": {...} }, ...],
    "user_request": "I need to modify my pending order #P123..."
  },
  "trajectory": [
    {"role": "assistant", "content": "Let me look up that order.", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "name": "get_order_details", "content": "{...}"},
    {"role": "assistant", "content": "Your order has 2 items. What would you like to change?"},
    {"role": "user", "content": "Please remove the blue shirt."},
    {"role": "assistant", "tool_calls": [{"function": {"name": "modify_pending_order_items", "arguments": "{...}}"}}]},
    {"role": "tool", "content": "{...}"},
    {"role": "assistant", "content": "Done! The blue shirt has been removed. Is there anything else?"}
  ],
  "outcome": {
    "task_completed": true,
    "reward": {
      "layer1": 1.0,
      "layer2": 1.0,
      "layer3": 0.85,
      "final": 0.95
    },
    "reward_weights": { "layer1": 0.5, "layer2": 0.3, "layer3": 0.2 }
  }
}
```

### 真实反馈数据的关键字段

模拟数据和真实数据**使用完全相同的格式**，区别仅在 `source` 字段和 `reward` 来源：

| 字段 | 模拟数据 | 真实数据 |
|------|---------|---------|
| `reward.layer1` | DB 状态比对 | 工具执行结果验证 |
| `reward.layer2` | 脚本判定 (tau-bench 有标注) | 用户消息解析 |
| `reward.layer3` | 脚本判定 | LLM Judge 低频调用 |

---

## 执行阶段

### Phase 1: 数据准备 (Windows 本地，无需 GPU)

**P1.1 — 下载数据集** ✅ (已完成)
- 输入: HuggingFace API + GitHub clone
- 输出: `data/tau_bench/raw/` (全部原始数据)
- 验证: 文件存在 + 格式检查

**P1.2 — 数据格式分析** ✅ (已完成)
- 输出: `docs/tau_bench_format.md`

**P1.3 — 实现 tau-bench 工具 mock 后端** 🟡 (执行中)
- 输出: `trainable_openclaw/agent/tau_bench_tools/`
- 零售 15 工具 + 航空 13 工具，Python dict mock 数据库

**P1.4 — 实现反馈收集模块** (新增)
- 输出: `trainable_openclaw/feedback/` (新建)
  - `signal_extractor.py` — 从用户消息提取 Layer 2 信号
  - `deterministic_verifier.py` — Layer 1 确定性验证
  - `reward_combiner.py` — 三层 reward 加权组合
- 验证: 给定一条对话，正确输出 layered reward

**P1.5 — 格式转换与合并**
- tau-bench 轨迹 → 统一训练样本格式
- 模拟数据 + 真实反馈数据合并（上线后持续追加）
- 输出: `data/training/train.jsonl`

**P1.6 — 数据质量过滤与拆分**
- 过滤: reward.final > 0 的轨迹, 去掉不完整/格式错误
- Train/test split: 80/20, 按 task_id 防泄漏

**P1.7 — 数据验证脚本**
- 自动检查: 工具调用格式/参数类型/消息角色/task 泄漏/reward 字段完整性
- 输出: 通过/失败报告

### Phase 2: SFT 冷启动训练 (远程 Linux GPU)

**P2.1 — SFT 训练**
- 基座: Qwen3-4B + LoRA rank=16
- 数据: Phase 1 产出的 train.jsonl
- 配置: lr=2e-5, batch_size=8, 3 epochs
- 验证: 工具调用格式正确率 > 60%

### Phase 3: GRPO 训练 (远程 Linux GPU)

**P3.1 — GRPO 训练**
- Prompt: tau-bench 164 个任务定义
- Rollout: nanobot + tau-bench mock 工具
- 奖励: 三层加权 (layer1=0.5, layer2=0.3, layer3=0.2)
- 配置: 16p×4r=64, lr=1e-5, max_turns=15

### Phase 4: 评测 + 上线自进化

**P4.1 — 标准评测**
- tau-bench airline (20 test tasks) + retail (40 test tasks)
- 指标: pass^k, step efficiency, tool selection accuracy

**P4.2 — 上线后自进化循环**
```
真实用户请求 → agent 执行 → 反馈收集 (Layer 1+2+3)
  → 追加到训练数据池 → 积累 N 条 → 触发 GRPO → 权重更新
```

---

## 产出文件清单

```
data/tau_bench/
├── raw/                          # 原始下载数据 ✅
├── train.jsonl                   # SFT 训练数据
├── test.jsonl                    # 评测数据
└── grpo_prompts.jsonl            # GRPO 训练 prompt

trainable_openclaw/agent/tau_bench_tools/
├── __init__.py
├── base.py                       # MockTool 基类
├── mock_db.py                    # Mock 数据库引擎 (Python dict)
├── retail.py                     # 零售 15 工具
├── airline.py                    # 航空 13 工具
└── registry.py                   # 工具注册到 nanobot

trainable_openclaw/feedback/      # 新增 — 反馈收集与 reward 计算
├── __init__.py
├── signal_extractor.py           # Layer 2 用户反馈信号提取
├── deterministic_verifier.py     # Layer 1 确定性验证
└── reward_combiner.py            # 三层 reward 加权组合

scripts/
├── download_tau_bench.py         # P1.1 ✅
├── convert_tau_bench.py          # P1.5
├── filter_split_tau_bench.py     # P1.6
└── validate_tau_bench_data.py    # P1.7
```

## 成功标准

- [x] 46K+ 条训练数据下载完成
- [ ] 28 个 tau-bench 工具 mock 实现完毕
- [ ] 反馈收集模块单元测试通过 (signal_extractor / verifier / combiner)
- [ ] 模拟训练数据成功转为统一格式
- [ ] 真实用户反馈可自动追加到训练数据池
- [ ] SFT 后模型工具调用格式正确率 > 60%
- [ ] GRPO 后任务完成率 > SFT baseline + 5%
- [ ] 上线后每 100 条真实反馈触发一次 GRPO 再训练
