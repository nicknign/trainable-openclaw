# Rubric 生成与验证设计 v1

> 日期: 2026-06-14 | 目标: DeepSeek 生成 tau-bench 零售 rubric → 驱动 4B GRPO 训练 → 自进化闭环

## 0. 借鉴工作

| 论文 | 核心发现 | 对本项目的指导 |
|------|---------|---------------|
| **MT-GRPO + IRC on Tau-Bench** (arxiv 2604.02869) | Qwen3.5-4B + MT-GRPO, airline 63.8→66.7%; read-only 工具给零 reward; reward 必须经验校准 | 同模型同任务，IRC 校准方法直接借用 |
| **RaR: Rubrics as Rewards** (Scale AI, arxiv 2507.17746) | checklist rubric 分 Essential/Important/Optional/Pitfall 四档做 GRPO reward, +28% | 验证了 rubric→GRPO reward 路线的可行性 |
| **RuscaRL** (arxiv 2508.16949) | hybrid rule-based + LLM judging; batch rubric grading 所有维度一次 API call | 对应我们 L1规则免费+L3 LLM低频 架构 |
| **RC-GRPO** (arxiv 2602.03025) | GRPO 组内方差退化时注入 reward-conditioned token 维持方差 | 遇到 reward 一致性问题时的解法 |
| **SPARK** (arxiv 2509.22624) | policy-reward co-evolving, 用 verifiable reward 同时训练策略和 RM | 与我们的 rubric evolver 共进化思路一致 |

## 1. 训练集分析

84 条零售训练任务（`train_prompts_augmented.jsonl`），23 条开发集，37 条测试集。按操作类型分布：

| 类型 | 预估占比 | 典型任务 | 关键工具 |
|------|---------|---------|---------|
| 查询 (lookup) | ~25% | 查订单状态、查产品信息、查用户信息 | find_user, get_order_details, product_search |
| 修改 (modify) | ~30% | 改地址、改商品规格、改支付方式 | modify_order, modify_user, update_payment |
| 退货退款 (return) | ~15% | 发起退货、查退款状态 | return_order, refund_order |
| 取消 (cancel) | ~10% | 取消订单 | cancel_order |
| 多步骤 (multi) | ~20% | 同时改地址+改商品+查新品 | 多种工具组合 |

> **首次实验**: 84 条全用，不再砍。量不大，16p×4r=64 几小时跑完，砍了反而损失类型多样性。
> 23 条开发集 (test_prompts.jsonl) 用于训练中间评测，37 条测试集用于最终对比 deepseek_retail_37 baseline。
> **不可用测试集训练**——会污染基准。

37 条测试任务用于评测，不参与 rubric 生成输入。

## 2. Rubric 生成流程

### 第一步: 任务聚类与采样

```
320 零售任务
  → 按所需工具/目标聚类为 5 类 (lookup/modify/return/cancel/multi)
    → 每类采样 3-4 条代表任务 (共 ~18 条)
      → 作为 DeepSeek 分析的输入
```

### 第二步: DeepSeek 分析任务类别

输入: 每类 3-4 条采样任务的 prompt + 工具列表 + 评估标准

分析维度 (agent tool-use 专用):

| 维度 | 要分析的 |
|------|---------|
| 工具选择 | 该任务必须用哪些工具、不该用哪些（乱用额外工具浪费回合） |
| 信息充分性 | agent 必须在回答前获取哪些信息（user_id、order_id、product 信息等） |
| 步骤效率 | 最少需要几步工具调用、哪些顺序是冗余的 |
| 错误恢复 | 工具返回 error 后是否重试、重试策略是否正确 |
| 任务完成确认 | agent 是否确认了用户所有要求、是否遗漏子任务 |
| 沟通质量 | 是否主动确认需求、是否给出具体结果而非泛泛而谈 |

输出格式 (每类):
```json
{
  "类别": "modify",
  "核心需求": ["必须先查user_id", "必须确认待修改订单", "修改后需验证"],
  "常见陷阱": ["忘记查zip_code直接搜用户名", "修改所有pending订单而非用户指定的那个"],
  "关键检查点": ["查了user_id?", "用对订单筛选条件?", "验证修改结果?"]
}
```

### 第三步: 生成 Rubric

输入: 5 类的分析结果 + 采样任务

输出: 15-20 条量化 rubric，每条格式:
```json
{
  "名称": "工具选择正确性",
  "评分提示词": "你是tau-bench零售客服评分专家。请评估agent的工具选择...\n\n评分规则:\n- 使用了所有必要工具: 不扣分\n- 遗漏1个必要工具: -3分\n- 使用了无关工具浪费回合: -1分/个\n- 完全不查用户信息直接回答: -5分\n\n输出JSON: {\"分数\": <0-10>, \"扣分项\": [...], \"总结\": \"...\"}",
  "适用任务类型": ["all"]
}
```

15-20 条 rubric 按 6 维度分布:

| 维度 | 数量 | 示例 |
|------|------|------|
| 工具选择正确性 | 3-4 | 必要工具覆盖、无关工具惩罚、工具调用顺序 |
| 信息充分性 | 3-4 | 用户身份确认、订单信息完整、产品信息完整 |
| 步骤效率 | 2-3 | 最少步骤完成率、是否有冗余查询 |
| 错误恢复 | 2-3 | 工具error重试、重试策略正确性 |
| 任务完成度 | 3-4 | 主任务完成、子任务覆盖、结果验证 |
| 沟通质量 | 2-3 | 主动确认需求、结果具体性、后续引导 |

## 3. 验证方案

### 3.1 校准数据准备

不需要额外花钱跑轨迹——直接用已有数据:

| 数据 | 轨迹数 | 用途 |
|------|--------|------|
| DeepSeek 通过轨迹 (retail 37条中 completed=true 的) | 10 条 | "好轨迹"参考 |
| DeepSeek 失败轨迹 (retail 37条中 give_up/timeout) | 27 条 | "差轨迹"参考 |

### 3.2 验证两步走

**验证 A: 区分度测试**

用 rubric 分别对 10 条好轨迹和 10 条差轨迹打分:
- 好轨迹平均分应显著高于差轨迹 (p < 0.05 via Mann-Whitney U)
- 好轨迹平均分 > 6.0, 差轨迹平均分 < 4.0
- score_spread > 0.1 (好/差组间差异)

**验证 B: 与 SimulatedUser satisfaction 正相关**

对 37 条 deepseek_retail_37.json 轨迹用 rubric 评分，计算:
- Pearson r(rubric_scores, satisfaction) > 0.5
- Spearman ρ > 0.5

如果 A 和 B 都通过 → rubric 有效，进入训练。
如果未通过 → 调整 rubric (增加/修改维度) 重新验证。

### 3.3 验证成本

37 条轨迹 × 20 条 rubric = 740 次 LLM 评分调用。但单次评分是简单打分任务，prompt 短、输出短。

改为合并评分: 1 次 API 调用用 5 条 rubric 合并评分，37 条 × 4 次 = 148 次调用。
每次 ~800 input tokens + ~200 output tokens:
- input: 148 × 800 = 118K tokens
- output: 148 × 200 = 30K tokens
- 成本: 118K × $0.14/M + 30K × $0.28/M ≈ $0.017 + $0.008 = **$0.025**

## 4. API 成本估算

### 4.1 定价

DeepSeek-chat (v4-flash): ~¥0.002/1K tokens ≈ $0.00027/1K

但按公开价格: 
- Input: $0.14 / 1M tokens
- Output: $0.28 / 1M tokens

### 4.2 首次生成 Rubric (一次性)

| 步骤 | API 调用 | Input tokens | Output tokens | 成本 |
|------|---------|-------------|---------------|------|
| 任务分析 (step 2) | 1 次 (合并 5 类) | ~4,000 | ~1,500 | ~$0.001 |
| 生成 rubric (step 3) | 1 次 | ~6,000 | ~3,000 | ~$0.0017 |
| 合并 (step 4) | 必要时 1 次 | ~5,000 | ~2,000 | ~$0.0013 |
| 精炼 (step 5) | 1 次 | ~5,000 | ~2,000 | ~$0.0013 |
| 验证 (step 6) | 148 次 | ~118,000 | ~30,000 | ~$0.025 |
| **合计** | | | | **~$0.03** |

### 4.3 进化轮 (每轮)

GRPO 训练后，分析失败轨迹，生成 3-5 条新 rubric:

| 步骤 | 成本 |
|------|------|
| DeepSeek 分析 ~10 条低分轨迹 | ~$0.003 |
| 生成 3-5 条新 rubric | ~$0.002 |
| 验证 (50 次评分) | ~$0.008 |
| **合计** | **~$0.013/轮** |

### 4.4 完整训练周期 (5 轮 GRPO)

| 项目 | 成本 |
|------|------|
| 首次 rubric 生成 + 验证 | $0.03 |
| 5 轮进化 × $0.013 | $0.065 |
| **总计** | **~$0.10** |

对比 DeepSeek 轨迹生成方案:
- 如果每轮需要 10 条 DeepSeek 轨迹: 10 × $2-5 = $20-50/轮
- 5 轮: $100-250
- **rubric 方案便宜 1000-2500 倍**

## 5. 训练时 Reward 桥接

训练时 4B rollout 评分 —— 零 API 成本:

```
4B rollout (tau-bench 零售, 320 prompts)
  → 解析轨迹中的工具调用和对话
    → 对每条轨迹运行 20 条 rubric (本地规则引擎，不再调 API)
      → reward = rubric 分数的加权平均
```

关键优化: 训练时 rubric 评分不调 LLM。而是将 rubric 转化为**规则检查**:
- "用了几种工具?" → 数 tool_calls 种类
- "是否先查了用户ID?" → 检查第一个 tool_call 是否是 find_user_id
- "是否有冗余查询?" → 检查是否重复查询相同信息
- "是否确认了所有子任务?" → 检查对话中是否覆盖所有 nl_assertions

**规则引擎评分 vs LLM 评分的权衡:**

| 方案 | 优点 | 缺点 |
|------|------|------|
| 规则引擎 | 零成本、确定性、快速 (毫秒级) | 部分维度无法规则化 (沟通质量) |
| LLM Judge | 全面覆盖所有维度 | 成本高、有随机性、慢 (秒级) |

**推荐混合方案:** 
- Layer 1 (工具格式/执行结果): deterministic_verifier (已有，零成本)
- Layer 2 (步骤效率/信息充分性): rubric 规则引擎 (零成本)
- Layer 3 (沟通质量): 可量化 rubric 规则化，必要时 LLM Judge (低频)

这样 320 条 rollout 评分零 API 成本 —— 只有进化时才花钱。

## 6. 与现有组件的对接

| 组件 | 角色 | 改动 |
|------|------|------|
| `rubric_engine.py` | 核心: extract→analyze→generate→merge→refine | 适配 agent trajectory 输入格式 |
| `judge.py` | 验证时 LLM 评分 | 不变 |
| `rubric_evolver.py` | 低分触发 DeepSeek 生成新 rubric | 适配 agent trajectory 失败模式 |
| `deterministic_verifier.py` | Layer 1 格式验证 | 不变 |
| `signal_extractor.py` | Layer 2 SimulatedUser 反馈提取 | 不变 |
| `reward_combiner.py` | 融合 L1+L2+L3 | 增加 rubric 规则评分权重 |
| `batch_eval_runner.py` | 标准 37 条评测 | 不变 |
| `pipeline.py` | 训练主循环 | 加上 rubric evolution 触发 |

## 7. Reward 校准 (来自 IRC 论文的关键教训)

IRC 论文发现: reward 值凭直觉设定很危险，必须按"与实际成功率的经验相关性"校准。

### 关键规则

1. **Read-only 工具调用给零 reward**——不能给正分。否则 agent 学会乱查数据库刷分（查了一堆信息但不完成任务）。
2. **State-changing 工具调用错误则惩罚**——错误修改比什么都不做更差。
3. **首次 GRPO 后做 reward 校准**: 跑完 R1，分析各维度分数分布 → 如果某维度几乎所有 rollout 都得高分 (无区分度)，调整该维度的扣分阈值。

### 校准流程

```
R1 训练后:
  → 收集 84 条 rollout 的 20 维 rubric 分数
    → 计算每维度的 mean/std
      → 标记区分度差的维度 (std < 0.15 或 mean > 8.5)
        → 调整该维度扣分门槛 → R2 验证改善
```

## 8. 执行顺序

```
Step 1: 生成 rubric (离线一次, $0.03)
  ├─ 聚类 320 零售任务 → 5 类
  ├─ DeepSeek 分析每类 → 6 维度
  └─ 生成 15-20 条 rubric → RubricStore

Step 2: 冷启动验证 (确认有效)
  ├─ 用 rubric 评 deepseek_retail_37.json 的 37 条轨迹
  ├─ 检查: 好轨迹 > 差轨迹 (p<0.05) + 与 satisfaction 正相关
  └─ 不通过 → 调整 → 重新验证

Step 3: GRPO 训练 R1
  ├─ 基座: Qwen3.5-4B + LoRA rank=16
  ├─ Prompt: 320 retail train prompts
  ├─ Reward: rubric 规则评分 (零 API) + deterministic_verifier
  ├─ 配置: 16p×4r=64, lr=1e-5, max_turns=10
  └─ 期望: completion rate 0.19 → 0.25+

Step 4: 评测 + Rubric 进化 + 再训练
  ├─ 37 条评测 → 对比 baseline
  ├─ DeepSeek 分析失败轨迹 → 生成 3-5 条新 rubric ($0.013)
  └─ 循环 5 轮

期望: baseline (0.191) → R1 (~0.25) → R2 (~0.30) → R3 (~0.35+)
```

## 9. 成功标准

- [ ] 15-20 条 tau-bench 零售 rubric 生成到 RubricStore
- [ ] 好/差轨迹区分度显著性 p < 0.05
- [ ] Rubric 分数与 SimulatedUser satisfaction Pearson r > 0.5
- [ ] GRPO R1 后 completion rate > baseline (27% → 30%+)
- [ ] Rubric evolver 正确触发: 低分模式 → 新 rubric 生成
- [ ] 2+ 轮完整训练闭环跑通
