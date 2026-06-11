# tau-bench Agent 训练实验计划

> 版本: v1.0 | 日期: 2026-06-12 | 状态: 执行中

## 目标

用 tau-bench 数据集训练 Qwen3-4B，使其能熟练调用 nanobot skill 完成客服业务任务。核心指标：任务完成率（pass^k）。

## 数据源

| 数据源 | 数量 | 用途 |
|--------|------|------|
| `sierra-research/tau2-bench` 历史轨迹 | ~860 条 (reward=1 的 ~500 条) | SFT 冷启动 |
| `amityco/apigen-tau-bench-split-turn` | 46,127 条 | SFT 冷启动（主要） |
| tau2-bench tasks.json | 164 任务定义 | GRPO 训练 prompt + 评测 |

## 执行阶段

### Phase 1: 数据准备 (Windows 本地，无需 GPU)

**P1.1 — 下载数据集**
- 输入: HuggingFace API
- 输出: `data/tau_bench_raw/` (原始数据)
- 验证: 文件存在 + 格式检查

**P1.2 — 数据格式分析**
- 输入: 原始数据
- 输出: `docs/tau_bench_format.md` (格式文档)
- 验证: 逐字段确认与 nanobot 消息格式的映射关系

**P1.3 — 实现 tau-bench 工具 mock 后端**
- 输入: tau-bench 工具定义
- 输出: `trainable_openclaw/agent/tau_bench_tools/` (nanobot skill 实现)
  - 零售 15 个工具 + 航空 13 个工具
  - 每个工具 = Python dict mock 数据库 + JSON Schema 定义
- 验证: 工具调用 → 返回正确格式

**P1.4 — 格式转换器**
- 输入: tau-bench 轨迹 (OpenAI messages 格式)
- 输出: serve_ppo 兼容的 `_training_pool` 格式
  - `{prompt: str, messages: list[dict], ground_truth: dict}`
- 验证: 格式校验通过 + nanobot API 可接受

**P1.5 — 数据质量过滤与拆分**
- 过滤: reward=1 的轨迹, 去掉不完整/格式错误
- Train/test split: 80/20, 按 task_id 防泄漏
- 输出: `data/tau_bench/train.jsonl` + `data/tau_bench/test.jsonl`

**P1.6 — 数据验证脚本**
- 13 项自动检查 (工具调用格式/参数类型/消息角色/task 泄漏等)
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
- 奖励: 任务完成 (binary) + rubric 评分 (continuous)
- 配置: 16p×4r=64, lr=1e-5, max_turns=15

### Phase 4: 评测

**P4.1 — 标准评测**
- tau-bench airline (20 test tasks)
- tau-bench retail (40 test tasks)
- 指标: pass^k, step efficiency, tool selection accuracy

## 产出文件清单

```
data/tau_bench/
├── raw/                          # 原始下载数据
├── train.jsonl                   # SFT 训练数据 (~40000 条)
├── test.jsonl                    # 评测数据 (~100 条)
└── grpo_prompts.jsonl            # GRPO 训练 prompt

trainable_openclaw/agent/tau_bench_tools/
├── __init__.py
├── base.py                       # MockTool 基类
├── mock_db.py                    # Mock 数据库引擎
├── retail.py                     # 零售 15 工具
├── airline.py                    # 航空 13 工具
└── registry.py                   # 工具注册到 nanobot

scripts/
├── download_tau_bench.py         # P1.1
├── analyze_tau_bench_format.py   # P1.2
├── convert_tau_bench.py          # P1.4
├── filter_split_tau_bench.py     # P1.5
└── validate_tau_bench_data.py    # P1.6
```

## 成功标准

- [ ] 46K+ 条训练数据下载完成
- [ ] 28 个 tau-bench 工具 mock 实现完毕
- [ ] 格式转换器能正确处理所有消息角色
- [ ] train/test 零 task 泄漏
- [ ] SFT 后模型工具调用格式正确率 > 60%
- [ ] GRPO 后任务完成率 > SFT baseline + 5%
