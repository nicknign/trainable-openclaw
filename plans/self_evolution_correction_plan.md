# 自进化训练纠正计划 v2

> 日期: 2026-06-14 | 状态: 起草

## 问题

之前偏离到「手动 SFT + GRPO + 人工评测」的纯手动路线。正确的方向是：**用 DeepSeek 生成高质量 rubric 作为 reward 信号，驱动 4B 自主训练进化。**

## 为什么用 Rubric 而不是轨迹生成

| 方案 | API 成本 | 复用性 | 可解释性 | 可进化性 |
|------|---------|--------|---------|---------|
| DeepSeek 生成轨迹 | $2-5/次，一次性 | 无 | 低 | 无 |
| DeepSeek 生成 rubric | $0.2/20条，持续用 | 高（每次训练都用） | 高（人能读懂） | 高（失败模式→新 rubric） |
| 4B 自己评自己 | 零 | - | 低 | 垃圾进垃圾出 |

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    离线：Rubric 生成（一次）                   │
│                                                             │
│  tau-bench 零售任务分类 → DeepSeek 分析每类任务特征            │
│    → 生成 task-specific rubrics（工具选择/步骤/错误恢复/沟通）  │
│      → RubricStore 持久化                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    训练时：Rubric 评分（零 API 成本）           │
│                                                             │
│  4B rollout (tau-bench retail)                               │
│    → rubric judge 自动评分每条轨迹                            │
│      → reward = rubric 分数                                  │
│        → GRPO 训练                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    评测后：Rubric 进化（低频）                  │
│                                                             │
│  评测发现低分模式 → DeepSeek 分析失败轨迹                      │
│    → 生成针对性新 rubric → 加入 RubricStore                   │
│      → 下一轮训练 reward 信号更精准                            │
└─────────────────────────────────────────────────────────────┘
```

## 和已用组件的对接

| 已有组件 | 在本计划中的角色 |
|---------|----------------|
| rubric_engine.py + judge.py (B2/B3) | **核心** — rubric 生成、存储、执行评分 |
| rubric_evolver.py (B4) | 低分触发 DeepSeek 生成新 rubric |
| pipeline.py (C1) | eval → train → re-eval 主循环 |
| orchestrator.py (A2) | 空闲检测 + 训练触发 |
| simulated_user.py | 4B rollout 时的交互对手 |
| tau-bench tools (mock_db) | 4B rollout 时的工具环境 |
| batch_eval_runner.py | 标准评测（37条零售） |
| deterministic_verifier.py | 辅助 — 工具调用格式验证（Layer 1） |
| signal_extractor.py | 辅助 — SimulatedUser 反馈提取（Layer 2） |

## 执行步骤

### Step 1: 生成 tau-bench 专用 rubric（离线，一次）

用 DeepSeek 分析 tau-bench 零售任务，生成 15-20 条 rubric：

**输入：** tau-bench 零售任务定义（task prompts + 工具列表 + evaluation criteria）

**分析维度：**
- 工具选择：该任务应该用哪些工具、不该用哪些
- 步骤效率：最少几步完成、哪些冗余步骤该扣分
- 错误恢复：工具返回 error 后是否正确重试
- 信息充分性：是否查了所有需要的信息再回答
- 沟通质量：是否确认用户需求、是否给出具体结果

**输出：** 15-20 条量化 rubric → RubricStore

**验证：** 用 DeepSeek 上限评测中的 10 条通过/27 条失败轨迹做校准——rubric 对通过轨迹打高分、失败轨迹打低分

### Step 2: 冷启动验证（确认 rubric 有效）

在 4B baseline 上跑 10 条零售 rollout，用 rubric 评分：
- 如果 rubric 分数和 SimulatedUser satisfaction 正相关 → rubric 有效
- 如果不相关 → 调整 rubric，重新验证
- 目标：rubric 评分能区分好轨迹和差轨迹（p < 0.05）

### Step 3: GRPO 训练

- 基座: Qwen3.5-4B + LoRA rank=16
- Prompt: tau-bench 零售 train prompts（~300条）
- Reward: rubric 评分（主）+ deterministic_verifier（辅）
- 配置: 16p×4r=64, lr=1e-5, max_turns=10, response_length=4096
- 3-5 轮训练

### Step 4: 评测 + Rubric 进化 + 再训练

```
Round 1 训练 → retail 37 条评测 → 对比 baseline
    → DeepSeek 分析失败轨迹 → 生成 3-5 条新 rubric
        → Round 2 训练 → 评测 → ...

期望: baseline (0.191) → R1 (~0.25) → R2 (~0.30) → R3 (~0.35+)
```

### Step 5: 上线自进化

- pipeline 自动检测：新轨迹 ≥ N 条 + 空闲 GPU → 触发训练
- rubric evolver 自动检测：低分模式 → DeepSeek 生成新 rubric
- 权重自动同步

## 丢弃的

- DeepSeek 轨迹生成 → 不需要，rubric 评分替代
- SFT 冷启动 → 跳过，Layer 1 确定性验证 + rubric 提供足够格式信号
- 航空任务 → 先聚焦零售，跑通再说
- 4B 自评自训 → 4B 太弱，评不准

## 成功标准

- [ ] 15-20 条 tau-bench 零售 rubric 生成并校准
- [ ] Rubric 评分与 SimulatedUser satisfaction 正相关
- [ ] GRPO R1 后 retail 37 条完成率 > baseline (27% → 30%+)
- [ ] Rubric evolver 正确触发——低分模式 → 新 rubric 生成
- [ ] 2+ 轮完整训练闭环跑通
