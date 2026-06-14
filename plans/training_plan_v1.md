# GRPO 首次训练计划 v1

> 日期: 2026-06-14 | 基座: Qwen3.5-4B + LoRA | 数据: 84 零售训练 + 23 开发 + 37 测试

## 架构

```
┌─ Train Prompt (84条) ─┐
│  initial user message │
└───────────┬───────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│          Rollout Environment             │
│                                          │
│  vllm (4B) ──→ agent msg ──→ Parse      │
│       ↑                          │       │
│       │                          ▼       │
│       │              Tool Executor (mock DB)
│       │                    │             │
│       │                    ▼             │
│       └──── tool results ──┤             │
│                            │             │
│              SimulatedUser (rule-based)  │
│                            │             │
│                            ▼             │
│       └──── user msg ──────┘             │
│                                          │
│  Loop max 10 turns, return trajectory    │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│          Rubric Rule Engine              │
│                                          │
│  20 rules → score breakdown → reward     │
│  (zero API cost, deterministic)          │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│          GRPO Update (verl)              │
│                                          │
│  N=4 rollouts/group, advantage normalize │
│  LoRA rank=16, lr=2e-6, KL=0.05          │
└──────────────────────────────────────────┘
```

## 组件清单

### A. Rollout Environment (`trainable_openclaw/training/rollout_env.py`)

多轮 agent 交互环境，替代原有 `batch_eval_runner.py` 的单任务循环，改为批量生产。

核心类: `TauBenchRolloutEnv`
- `__init__(prompts, max_turns, sim_user_mode)` — 初始化 84 条训练 prompt
- `rollout_one(prompt, model_fn)` — 单条 prompt 完整 rollout
  - `model_fn` 是 vllm generate 接口，由 verl 注入
  - 循环: model generate → parse tool calls → execute → sim user respond → repeat
- `rollout_batch(prompts, model_fn)` — 批量 rollout

内部组件:
- `ToolExecutor`: 直接导入 tau_bench mock DB，本地执行工具调用 (不调 nanobot API)
- `RuleSimulatedUser`: 基于任务定义 + nl_assertions 的确定性用户模拟器
  - 解析 prompt 中的用户需求和身份
  - 每轮检查 agent 是否满足需求 → 给反馈
  - 检测到所有 assertions 满足 → 返回 complete
  - 检测到 agent 偏离方向 → 给出导向性回复

### B. Rubric Rule Engine (`trainable_openclaw/training/rubric_rules.py`)

将 20 条 rubric 转化为可执行规则，零 API 成本。

核心类: `RubricRuleEngine`
- `__init__(rubric_specs_path)` — 加载 rubric 规则定义
- `score(trajectory)` → `{dimension: score, ...}`
- `compute_reward(trajectory)` → `float` (加权总分)

20 条规则分为 6 组:
| 组 | 规则数 | 检查方式 |
|----|--------|---------|
| 工具选择 | 4 | 统计 tool_calls 类型，对比任务需要的工具 |
| 信息充分性 | 4 | 检查是否在回复用户前获取所有必要信息 |
| 步骤效率 | 3 | 检查冗余调用、重复查询 |
| 错误恢复 | 3 | 检查 tool error 后是否重试 |
| 任务完成 | 4 | 检查是否覆盖所有 nl_assertions |
| 沟通质量 | 2 | 检查回复是否包含具体结果 |

### C. GRPO Reward Bridge (`trainable_openclaw/training/grpo_reward.py`)

verl 兼容的 reward 函数。

```python
def compute_reward(
    data: DataProto,           # verl 标准输入
    rubric_engine: RubricRuleEngine,
    rollout_env: TauBenchRolloutEnv,
    **kwargs
) -> list[float]:             # 每条 rollout 一个 reward
```

流程:
1. 从 DataProto 提取 prompt
2. 对每条 prompt 跑完整 multi-turn rollout
3. 用 RubricRuleEngine 评分轨迹
4. 返回 reward 列表

### D. 训练配置 (`scripts/train/`)

`grpo_retail.yaml`:
```yaml
model:
  base: Qwen/Qwen3.5-4B
  lora_rank: 16
  lora_alpha: 32
  
training:
  batch_size: 8        # prompts per step
  n_rollouts: 4        # rollouts per prompt (group size)
  max_turns: 10
  learning_rate: 2e-6
  kl_coef: 0.05
  max_steps: 200
  save_every: 50
  
reward:
  custom_function: "trainable_openclaw.training.grpo_reward.compute_reward"
```

`run_grpo.sh` — 单卡 RTX 4090 启动脚本

## 验证标准

### 单元测试 (coder 写)
- [ ] `ToolExecutor.execute()` — 正确执行 tau-bench retail 工具
- [ ] `RuleSimulatedUser.respond()` — 正确判断 agent 回复并生成下一条 user 消息
- [ ] `TauBenchRolloutEnv.rollout_one()` — 完整跑通一条 prompt
- [ ] `RubricRuleEngine.score()` — 对 DeepSeek 好轨迹打高分、4B 差轨迹打低分

### 集成测试 (tester 做)
- [ ] 用 mock model 跑 3 条 prompt 完整 rollout
- [ ] 好轨迹 (DeepSeek) vs 坏轨迹 (随机) 分数区分度验证
- [ ] GRPO reward 函数与 verl DataProto 兼容性验证
- [ ] 84 条训练数据全部加载正确

## 实现顺序

```
A. ToolExecutor            (1h) — 本地工具执行
A. RuleSimulatedUser       (1h) — 规则用户模拟
A. TauBenchRolloutEnv      (1h) — 组装环境
B. RubricRuleEngine        (1h) — 规则评分引擎
C. GRPO Reward Bridge      (1h) — verl 接口对接
D. Training Config         (30m) — yaml + shell
测试                       (1h) — e2e tester
```

## 参考

- IRC 论文: same model (Qwen3.5-4B), same task, same method
- MT-GRPO 论文: turn-level credit assignment, format reward design
- 现有 `batch_eval_runner.py`: multi-turn rollout 模式参考
- 现有 `deterministic_verifier.py`: L1 格式检查复用
