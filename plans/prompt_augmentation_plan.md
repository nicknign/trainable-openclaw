# Prompt Augmentation Plan — 混合扩写 (C 方案)

> 日期: 2026-06-12 | 状态: 执行中

## 目标

将 131 train + 33 test prompts 扩写至 500 train + 50 test。

## 核心约束

**LLM 只换说法，不改实体。** 所有实体值必须从 mock DB 中提取，确保 prompt 可执行。

## 扩写策略 (C 方案)

```
原始 prompt (131 train / 33 test)
      │
      ▼
Step 1: 实体提取
  ├─ 解析每条 prompt 中的实体: 人名、user_id、订单号、zip code、产品名、航班号...
  ├─ 从 MockDatabase seed data 中提取同类型实体池
  └─ 输出: {原始实体 → 替换实体} 映射表
      │
      ▼
Step 2: 实体替换
  ├─ 用池中实体替换原始实体（确保 mock DB 中存在）
  ├─ 同 task 的不同变体使用不同实体组合
  └─ 输出: 实体已替换的 prompt 草稿
      │
      ▼
Step 3: LLM 措辞润色
  ├─ 保持所有实体值不变
  ├─ 只改变句式、语气、措辞（"I want to..." → "Could you help me..."）
  ├─ 输入: 实体替换后的 prompt 草稿
  └─ 输出: 自然语言 prompt 变体
      │
      ▼
Step 4: 可执行性验证
  ├─ 对每个变体: 用 AgentRunner + mock tools 执行一次
  ├─ 检查: 工具调用不报 "not found" 错误（说明实体存在）
  └─ 过滤: 不可执行的变体丢弃，重新生成
```

## 扩写比例

| Split | 原始 | 目标 | 倍数 | 每原始 prompt 生成变体数 |
|-------|------|------|------|------------------------|
| Train | 131 | 500 | ~3.8× | 3-4 |
| Test | 33 | 50 | ~1.5× | 1-2 |

变体数按 task 复杂度加权分配（简单任务少变体，复杂任务多变体）。

## 模型配置

- **LLM 润色**: DeepSeek-V4-Flash（成本低、速度快，措辞改写足够）
- **验证**: DeepSeek-V4-Flash（同上）

- 生成 prompt 总数 ≥ 500 train + 50 test
- 抽样 20 条跑 agent + mock tools，通过率 ≥ 95%（实体不存在算失败）
- Train/test task_id 零泄漏保持不变
- 每个原始 task_id 至少保留 1 个变体

## 产出

```
data/tau_bench/
├── train_prompts.jsonl          # 原始 131 (覆盖)
├── test_prompts.jsonl           # 原始 33 (覆盖)
├── train_prompts_augmented.jsonl  # NEW: 500 扩写后
└── test_prompts_augmented.jsonl   # NEW: 50 扩写后

scripts/
└── augment_prompts.py           # 扩写脚本
```
