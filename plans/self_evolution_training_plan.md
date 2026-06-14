# 自进化训练完整计划 v3

> 日期: 2026-06-14 | 覆盖: 从 baseline 到多轮自进化闭环

## 0. 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 0: 准备 (本机 ✓ / GPU 待)               │
│                                                                 │
│  ✓数据分析 → ✓Split隔离 → ✓Rubric生成 → ✓代码+测试 → ⬜Baseline│
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Phase 1-N: 训练循环 (Linux GPU)                 │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐ │
│  │ GRPO训练  │ → │ Val评测   │ → │ 失败分析  │ → │ Rubric进化  │ │
│  │ (66任务)  │   │ (18任务)  │   │ (DeepSeek)│   │ (+3-5条)   │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────┬──────┘ │
│       ↑                                               │        │
│       └───────────────────────────────────────────────┘        │
│                                                                 │
│  Test 23 任务只在最终评测触碰一次                                 │
│  Target: baseline(待测) → R1(~0.25) → R2(~0.30) → R3(~0.35+)  │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 0: 训练前准备 (本机，不需要 GPU)

### 0.1 数据 Split 分析与隔离 (已完成 ✓)

**原始数据:**

| | 原 Train | 原 Val | Test |
|---|---|---|---|
| 条目 | 320 | 28 | 37 |
| 唯一任务 | **84** | **7** | **23** |

**重叠度: 零。** Train ∩ Test = ∅

**原 Val 问题:** 28 条仅 7 个任务，43% 是 return 类 — 太小太偏，不适合做 checkpoint 选择。

**新 Split 方案 — 三层隔离:**

```
原 Train 84 任务 (320条)
  ├─→ Train 66 任务 (251条) ──→ GRPO 训练 + Rubric 校准
  └─→ Val   18 任务 ( 69条) ──→ Checkpoint 选择

Test 23 任务 (37条) ──→ 只碰一次，最终报告
```

**Val 拆分策略:** 分层抽样，保证 12 种任务类型全部在 Val 中有代表（每种至少 1 个任务）。

**核心原则:**
- Rubric 校准用**训练轨迹**（rubric 是规则不是可学习参数，看训练集不泄漏）
- Checkpoint 选择用 **Val**（从未参与训练）
- Test **仅在最终评测触碰一次**（不用于校准、不用于选 checkpoint）

### 0.2 重新建立 Baseline (需要 GPU)

**为什么重测**: 之前的 4B 0.191 是混合 30 条(含航空)，评测环境已变(bug 修复后)。需要专测零售 37 条。

```
评测: Qwen3.5-4B (vllm) + nanobot + SimulatedUser(DeepSeek)
测试集: 37 条零售 (test_prompts_augmented.jsonl)
配置: max_rounds=7, concurrency=4, early_stop=true
产出: evaluation_results/baseline_4b_retail_37.json
```

### 0.3 DeepSeek 生成零售专用 Rubric (已完成 ✓)

已通过 DeepSeek API 分析 12 个采样训练 prompt 生成 20 条零售专用 rubric，
写入 `data/rubrics_retail.json`。Python 规则引擎 `rubric_rules.py` 已更新为
23 条可执行规则（6 维度），使用精确 tau-bench 工具名称。
29 单元测试通过，合成轨迹上 good/bad 区分度 2.15x。

### 0.4 Rubric 在真实轨迹上验证 (待 GPU)

**在训练集轨迹上验证**（不碰 Test）:
```
Baseline 评测 66 个训练任务 → 收集真实 Qwen3.5-4B 轨迹
  - 完成轨迹 → 期望得分 > 0.6
  - 失败轨迹 → 期望得分 < 0.4
  - 区分度: p < 0.05 (Mann-Whitney U)
  - 与 satisfaction 相关性: Pearson r > 0.5

不通过 → 调整 rubric → 重新在训练轨迹上验证
通过 → 进入 Phase 1 训练
```

Rubric 是确定性规则不是可学习参数，在训练集上调优不构成数据泄漏。

---

## Phase 1: GRPO R1 训练 (Linux GPU)

### 1.1 部署

```bash
# 1. 同步代码到远程
python scripts/tools/autodl_sync.py

# 2. 启动服务栈
bash scripts/deploy/start_experience.sh

# 3. 验证服务
curl http://localhost:8900/health    # nanobot
curl http://localhost:8000/health    # vllm
```

### 1.2 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | Qwen3.5-4B | |
| 适配器 | LoRA rank=64, alpha=128 | 多轮工具调用需要更大容量 |
| 训练 prompt | 66 任务 (~251 条) | train/val split 后的训练部分 |
| Rollout 数 | n=4 / prompt | 组内归一化 |
| Batch size | 8 prompts | 每步 8×4=32 条 rollout |
| 学习率 | 2e-5 | LoRA 常规范围 5e-6~5e-5，取中间偏稳 |
| KL 系数 | 0.05 | |
| 温度 | 0.9 | 鼓励探索 |
| Max turns | 10 | 超过给 time penalty |
| Max steps | 200 | 约 6-7 epochs |
| Save checkpoint | 每 50 steps | |
| GPU | 单卡 RTX 4090 48GB | |

### 1.3 训练监控

**每步记录:**
```
step, reward_mean, reward_std, kl_div, completion_rate, avg_turns
```

**关键指标:**
| 指标 | 健康信号 | 不健康信号 |
|------|---------|-----------|
| reward_mean | 逐步上升 | 不变或下降 |
| reward_std | >0.15 (组内有区分) | <0.05 (组内一致→梯度退化) |
| kl_div | 0.01-0.1 | >0.5 (策略跳变过大) |
| completion_rate | 逐步上升 | 不变 |
| avg_turns | 逐步下降(更高效) | 上升(越训越啰嗦) |

### 1.4 IRC 校准 (基于 IRC 论文)

训练 50 步后，收集 rollout 数据做首次校准:

```
收集 50 steps × 32 rollouts = 1600 条 rollout
  → 按 reward tier 分类每轮行为
    → 计算每 tier 的 point-biserial correlation with task success
      → 调整 reward 值:
        - |ρ| > 0.1 → 保留当前值
        - |ρ| < 0.1 → 设为 0 (无判别力)
        - ρ < -0.1 → 翻转为负
```

### 1.5 检查点评估

每 50 步用 Val 集 (18 任务, ~69 条) 快速评测:
```
python ai_scripts/batch_eval_runner.py --max-tasks 18 --domain retail --val-split --output /tmp/r1_step50.json
```

Val 任务从未参与训练，completion rate 反映真实泛化能力。
**Test 集此时不碰。**

---

## Phase 2: R1 评测 + 失败分析

### 2.1 标准评测 — Test 首次触碰

```
全量 Test 23 任务 (37 条) — 训练全程首次触碰:
  → 对比 baseline
    → completion_rate, satisfaction, first_try_rate
      → 与 deepseek_upper_bound (0.390) 的差距
  → 同时报告 Val 分数做对照（Val 用于 checkpoint 选择，可能略高）
  → 如果 Test ≪ Val → 过拟合，停止迭代，检查原因
```

### 2.2 失败分析 (DeepSeek)

选 R1 评测中 completion=false 且 satisfaction<0.3 的轨迹 (最多 10 条):

```
DeepSeek 分析每类失败模式:
  1. 核心问题: agent 在哪一步卡住/走偏?
  2. 根因: 工具选择错误? 信息不充分? 步骤顺序错?
  3. 建议: 增加什么 rubric 可以防止这类失败?

输出: 失败模式报告 + 3-5 条新 rubric
成本: ~$0.005
```

### 2.3 Rubric 进化

```
新增 3-5 条针对性 rubric → 加入 RubricStore
  → 重新验证区分度
    → 通过 → 进入 R2 训练
```

---

## Phase 3-N: 迭代循环

```
for round in 2..N:
    1. 加载上一轮最佳 checkpoint
    2. 用进化后的 rubric 做 GRPO
    3. 37 条评测
    4. 如果 completion_rate 连续 2 轮无提升 (>0.02):
       → 停止训练闭环
       → 检查: 是 rubric 区分度不够? 还是模型能力上限?
    5. 如果持续提升:
       → 继续进化 rubric → 下一轮
```

**期望轨迹:**
```
baseline → R1 → R2 → R3
 0.19?  → ~0.25 → ~0.30 → ~0.35+
```

**每轮成本:**
| 项目 | API 成本 |
|------|---------|
| 失败分析 (DeepSeek) | ~$0.005 |
| 进化 rubric 生成 | ~$0.002 |
| 开发集评测 (SimUser) | ~$0.02 |
| 测试集评测 (SimUser) | ~$0.03 |
| **合计/轮** | **~$0.06** |

---

## 成功标准

### Phase 0
- [ ] 4B 零售 baseline 重测完成
- [ ] 15-20 条零售专用 rubric 生成
- [ ] Rubric 区分度验证通过 (p<0.05)

### Phase 1
- [ ] GRPO R1 训练完成 (200 steps)
- [ ] 训练过程无崩溃
- [ ] Reward 曲线上升趋势
- [ ] KL divergence 在健康范围

### Phase 2
- [ ] R1 37 条评测完成
- [ ] Completion rate > baseline
- [ ] 失败模式分析完成
- [ ] 3-5 条新 rubric 生成

### Phase 3+
- [ ] 2+ 轮完整训练闭环
- [ ] 最终 completion rate > 0.30
- [ ] Rubric evolver 自动触发验证通过

---

## 文件清单

| 文件 | 用途 | 状态 |
|------|------|------|
| `data/rubrics_retail.json` | DeepSeek 生成的零售专用 rubric | ⬜ 待生成 |
| `trainable_openclaw/training/rubric_rules.py` | 替换通用规则为专用规则 | ⬜ 待更新 |
| `evaluation_results/baseline_4b_retail_37.json` | 4B 零售 baseline | ⬜ 待重测 |
| `evaluation_results/grpo_r1_37.json` | R1 评测结果 | ⬜ 待训练 |
| `evaluation_results/rubric_evolution_r1.json` | R1 失败分析 + 新 rubric | ⬜ 待分析 |
| `scripts/train/grpo_retail.yaml` | 训练配置 | ✅ 已完成 |
| `scripts/train/run_grpo.sh` | 启动脚本 | ✅ 已完成 |
| `trainable_openclaw/training/rollout_env.py` | 多轮 rollout 环境 | ✅ 已完成 |
| `trainable_openclaw/training/grpo_reward.py` | verl reward bridge | ✅ 已完成 |

## 下一步

1. **马上能做**: Phase 0.2-0.3 — 聚类训练集 + DeepSeek 生成 rubric (不需要 GPU)
2. **启动 Linux 后**: Phase 0.1 + Phase 1 — baseline 重测 + GRPO 训练
3. **训练完成后**: Phase 2-3 — 评测 + 进化 + 迭代
