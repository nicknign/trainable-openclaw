# Mock DB Enhancement & Evaluation Alignment Plan

> 日期: 2026-06-12 | 状态: **完成**

## 实验结果 (2026-06-12)

### Part A: Mock DB 增强 — 完成
- **A1 用户表**: 为 retail 15 用户 + airline 15 用户添加 `gift_card_balance`（$0-$200 分布）
- **A2 订单表**: 为 15 零售订单添加 `payment_method`（type/last_four/brand）+ `gift_card_applied` bool
- **A3 新工具**: `get_user_orders` (retail) + `get_user_reservations` (airline) 加入工具注册表
- **A4 验证**: `get_user_details` 自动返回所有新字段（已返回完整 dict 副本）

### Part B: Evaluation 断言更新 — 完成
- 脚本 `scripts/align_eval_assertions.py` 完成
- 身份提取支持 10+ prompt 格式变体（名+邮编、email、snake_case 等）
- 550/550 prompts 均已填充 `nl_assertions`（100% 覆盖）
- 无 DeepSeek API key 时使用基于 mock DB 数据的 fallback 断言
- API key 可用时支持 LLM 生成更精确断言

### Part C: 验证 — 全部通过
- 45/45 e2e 测试通过
- 6 项手动验证通过（新字段、新工具、数据一致性）
- 所有 JSONL 文件有效

### 已知限制
- ~2 个 prompts（airline_task_3 variant）未匹配到 DB 用户（prompt 格式异常）
- Fallback 断言较泛化，LLM 生成可提供更精准断言

## 问题

扩写 prompt 时换了实体（user_id、订单号等），但 mock DB 和 evaluation 断言没跟着更新，导致三个组件之间数据不一致：

```
Prompt: "Lisa Nakamura, U012" ──→ Mock DB: 没有 gift_card_balance ──→ Eval: "balance is $60"
                                       没有 get_user_orders              "paid with Mastercard"
                                       订单没有 payment_method
```

## 目标

确保评测循环中三个组件的数据一致性：**Prompt 实体 → Mock DB 数据 → Evaluation 断言**。

## A — 增强 Mock DB

### A1. 用户表增强
- 为每个用户添加 `gift_card_balance` 字段（随机 $0-$200）
- 确保至少部分用户有非零余额（供 gift card 相关任务使用）

### A2. 订单表增强  
- 为每个订单添加 `payment_method` 字段（从用户的 payment_methods 中选）
- 添加 `gift_card_applied` 布尔字段

### A3. 新增工具
- `get_user_orders(user_id)` — 根据 user_id 查找该用户所有订单
- 这是 tau-bench 任务的常见需求，当前缺失

## B — 更新 Evaluation 断言

### B1. 重新生成 nl_assertions
- 读取每条 prompt 的任务描述
- 查询 mock DB 中实际数据
- 用 LLM 生成匹配 mock DB 实际值的断言
- 例如：原断言 "balance is $60" → "balance is $0.00" 或 "balance is $XX.XX (实际值)"

### B2. 更新 augmented prompt 文件
- 替换 `train/val/test_prompts_augmented.jsonl` 中的 `evaluation` 字段
- 确保断言中的数值与 mock DB 一致

## C — 验证

- 随机抽样 10 条 prompt，逐一验证：
  1. Prompt 中提到的 user_id 在 mock DB 中存在
  2. Prompt 中提到的订单属于该用户
  3. Evaluation 断言中的数值与 mock DB 一致
  4. Agent 可以仅通过工具调用完成任务（不需要"猜"数据）

## 产出

```
trainable_openclaw/agent/tau_bench_tools/
├── mock_db.py                  # 修改：增强用户和订单字段
├── retail.py                   # 修改：新增 get_user_orders 工具

data/tau_bench/
├── train_prompts_augmented.jsonl  # 修改：更新 evaluation 断言
├── val_prompts_augmented.jsonl    # 修改：更新 evaluation 断言
└── test_prompts_augmented.jsonl   # 修改：更新 evaluation 断言

scripts/
└── align_eval_assertions.py    # 新增：断言重新生成脚本
```
