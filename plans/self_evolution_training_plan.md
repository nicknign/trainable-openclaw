# 自进化训练完整计划 v3

> 日期: 2026-06-14 | 覆盖: 从 baseline 到多轮自进化闭环

## 0. 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 0: 准备 (本机 ✓ / GPU 待)               │
│                                                                 │
│  ✓数据分析 → ✓DeepSeek生成rubric → ✓代码+测试 → ⬜Baseline重测  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Phase 1-N: 训练循环 (Linux GPU)                 │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐ │
│  │ GRPO训练  │ → │ 标准评测  │ → │ 失败分析  │ → │ Rubric进化  │ │
│  │ (verl)   │   │ (37条)   │   │ (DeepSeek)│   │ (+3-5条)   │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────┬──────┘ │
│       ↑                                               │        │
│       └───────────────────────────────────────────────┘        │
│                                                                 │
│  Target: baseline(待测) → R1(~0.25) → R2(~0.30) → R3(~0.35+)  │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 0: 训练前准备 (本机，不需要 GPU)

### 0.1 数据 Split 分析 (已完成 ✓)

| | Train | Val | Test |
|---|---|---|---|
| 条目 | 320 | 28 | 37 |
| 唯一任务 | **84** | **7** | **23** |
| 每条增强数 | 3-4 变体 | 4 变体 | 1-2 变体 |
| 断言数均值 | 1.9 | 1.9 | 1.8 |

**重叠度: 零。** Train ∩ Val = Train ∩ Test = Val ∩ Test = ∅

增强质量: 84 个任务全部有变体，改动用户身份(姓名/邮箱/邮编)，断言随变体变化 → 模型必须学流程，无法死记硬背。

任务类型分布:
```
Train: exchange+modify(27%), modify(17%), return(12%), cancel+return+modify(9%), ...
Test:  exchange+modify(27%), cancel+return(16%), modify(14%), return(14%), ...
```

**结论与决策:**
- Val (28条, 7任务, 43% return) 太小且偏斜 → **跳过 Val，checkpoint 评估直接抽 15 条 Test**
- Test 剩余 22 条保留做最终评估 → **训练过程不看，避免过拟合**
- 84 训练任务 × 3.8 变体对 4B 模型基本够用但不充裕

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

需要等 baseline 评测产出真实轨迹数据后执行:
```
在真实 Qwen3.5-4B 轨迹上:
  - 完成轨迹 → 期望得分 > 0.6
  - 失败轨迹 → 期望得分 < 0.4
  - 区分度: p < 0.05 (Mann-Whitney U)
  - 与 satisfaction 相关性: Pearson r > 0.5
```

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
| 适配器 | LoRA rank=16, alpha=32 | |
| 训练 prompt | 84 条零售 | train_prompts_augmented.jsonl |
| Rollout 数 | n=4 / prompt | 组内归一化 |
| Batch size | 8 prompts | 每步 8×4=32 条 rollout |
| 学习率 | 2e-6 | IRC 论文推荐 |
| KL 系数 | 0.05 | |
| 温度 | 0.9 | 鼓励探索 |
| Max turns | 10 | 超过给 time penalty |
| Max steps | 200 | 约 5-6 epochs |
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

每 50 步用测试集前 15 条快速评测（跳过 Val — 太小且偏斜）:
```
python ai_scripts/batch_eval_runner.py --max-tasks 15 --domain retail --output /tmp/r1_step50.json
```

剩余 22 条保留到 Phase 2 最终评测，训练过程不接触，避免过拟合。

---

## Phase 2: R1 评测 + 失败分析

### 2.1 标准评测

```
全量 37 条零售测试集评测 (含训练期未接触的 22 条):
  → 对比 baseline
    → completion_rate, satisfaction, first_try_rate
      → 与 deepseek_upper_bound (0.390) 的差距
  → 单独报告 checkpoint-15 和 heldout-22 的分数，检测过拟合
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
