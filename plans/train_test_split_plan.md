# Train/Test Split & Test Set Generation Plan

> 日期: 2026-06-12 | 状态: 待派发

## 背景

当前 164 条 GRPO prompts (114 unique task_ids) 需要重新切分：
- 旧 train/test 有 23 个 task_id 泄漏，不能直接复用
- 新评测范式下，test set = 实际运行交互式评测的 prompts

## 目标

1. **干净的 train/test split** — 按 task_id 切分，零泄漏
2. **生成 test set** — 对 test prompts 运行交互式评测，产出 baseline 指标

## 步骤

### Step 1: 数据切分

- 输入: `data/tau_bench/grpo_prompts.jsonl` (164 prompts, 114 unique task_ids)
- 策略: 按 task_id 分组 → stratified split (airline/retail) → 80/20
- 零泄漏保证: 同一 task_id 的所有 prompt 必须在同一 split
- 输出:
  - `data/tau_bench/train_prompts.jsonl` (~131 prompts)
  - `data/tau_bench/test_prompts.jsonl` (~33 prompts)
  - 切分报告: 各 split 的 task 数、domain 分布、prompt 数

### Step 2: 测试集生成（交互式评测）

- 输入: `data/tau_bench/test_prompts.jsonl`
- Agent: DeepSeek API + tau-bench mock tools
- Simulated User: DeepSeek API
- 输出:
  - `data/eval/baseline_report.json` — 评测报告 (avg_rounds, first_try_rate, recovery_rate, abandonment_rate)
  - `data/eval/baseline_trajectories.jsonl` — 完整对话轨迹

## 产出文件

```
data/tau_bench/
├── grpo_prompts.jsonl          # 原始 164 prompts (不变)
├── train_prompts.jsonl         # NEW: 训练用 ~131 prompts
├── test_prompts.jsonl          # NEW: 测试用 ~33 prompts
├── train.jsonl                 # 旧历史轨迹 (保留参考)
└── test.jsonl                  # 旧历史轨迹 (保留参考)

data/eval/
└── baseline_report.json        # NEW: baseline 评测报告
└── baseline_trajectories.jsonl # NEW: baseline 轨迹
```

## 验证标准

- train/test task_id 交集为空
- 每个 split 的 domain 比例接近原始分布 (airline ~30%, retail ~70%)
- 所有 164 prompts 都被分配到 train 或 test
- Step 2 评测至少跑通 5 条 test prompts
