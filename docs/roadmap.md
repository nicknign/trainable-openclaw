# 研发路线图

## 原则

- **单线程推进** — 一个人开发，Phase 1→2→3→4 严格串行
- **调研与开发重叠** — 已有初步想法和论文积累，调研在背景持续进行，不阻塞开发
- **直接改veRL源码，不写mock** — 预计深度改造，mock没有意义
- **每步可独立验证** — 写完 → 在Linux环境验证通过 → 再下一步
- **先简单实现，再复杂优化** — 每步只做最小可用版本

---

# 总体规划

## 项目目标

打造一个**生产级自进化AI助手引擎**。核心闭环：

```
用户请求 → veRL引擎推理 → 回复用户
                ↓
         对话日志存储
                ↓
     LLM分析用户反馈模式 → LLM自主生成Rubrics
                ↓
      Rubrics对GRPO多答案打分 → 构造训练样本
                ↓
      空闲检测 → 触发LoRA训练 → 权重同步 → 恢复推理
                ↓
      新反馈 → Rubrics持续演进 → 模型持续进化
```

## 阶段划分

| 阶段 | 内容 | 依赖 | 性质 |
|------|------|------|------|
| Phase 0 | 论文调研与算法确定 | — | 背景持续，与开发重叠 |
| Phase 1 | veRL双模引擎改造 | 0 | 核心难点，先做 |
| Phase 1.5 | 数据工程与模拟测试环境 | 1 | 效果验证必备，Phase 2 前置 |
| Phase 2 | 自进化评判系统 | 1.5 | 核心难点，后做 |
| Phase 3 | 集成与Dashboard | 1.5+2 | 串联 |
| Phase 4 | 生产环境评估 | 3 | 框架完成后 |

## 时间线（单线程，目标 5/16 → 6/16）

```
Week 1 (5/16-5/22):
  A1: Rollout API Server ───────────────── 最难点，集中攻克
  Phase 0: 论文调研 ────────────────────── 背景进行

Week 2 (5/23-5/29):
  A2: 空闲检测 + 训练触发
  A3: 权重同步 + 恢复推理
  Phase 0: 算法方向确定 ────────────────── 背景进行
  B0: 对话日志系统 (SQLite)

Week 3 (5/30-6/5):
  Phase 1.5: 数据工程 ──────────────────── S1-S5 模拟测试环境

Week 4 (6/6-6/12):
  B1: 用户反馈收集与分析
  B2: LLM自主生成Rubrics

Week 5 (6/13-6/19):
  B3: Rubric执行器 (LLM Judge)

Week 6 (6/20-6/26):
  C1: 主循环串联 ───────────────────────── 关键里程碑

缓冲 (6/27-6/30):
  B4: Rubric演进 (基本版)
  C2: Dashboard (简化版)
  联调、修bug、写文档
```

### MVP交付物（6/30）

- veRL 常驻推理 API 服务可运行
- 空闲时自动触发训练，训练完恢复推理
- 模拟测试环境：5 个用户画像，可控反馈闭环
- LLM 从用户反馈中生成 Rubrics 并执行打分
- Judge 校准数据 + 评估指标体系
- 核心闭环走通：请求 → 日志 → 评判 → 训练 → 恢复
- 简易 Dashboard

### 推迟到一个月后

- Phase 4 (D1/D2/D3)：生产环境评估、A/B 测试
- Rubric 精细化调优
- 生产级错误处理、监控告警

### 风险

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| veRL改造比预期复杂 | 高 | 阻塞后续 | W1集中攻克，如太复杂降级为简单wrapper |
| GRPO多答案评判效果差 | 中 | B2/B3返工 | 先做pointwise，pairwise作为优化 |
| Linux环境部署踩坑 | 中 | 消耗时间 | 提前准备Docker镜像 |
| 单线程时间不够 | 中 | MVP延期1-2周 | 砍B4/C2保C1核心闭环 |

---

## 两大核心难点

| # | 难点 | 说明 |
|---|------|------|
| 1 | **veRL双模引擎改造** | 把veRL的rollout阶段抽出来做常驻API服务，空闲时触发训练，训练完切回推理 |
| 2 | **自进化评判系统** | LLM从用户反馈中自主归纳生成Rubrics（严格量化评分任务），用这些Rubrics对GRPO多答案打分排名 |

---

# 详细展开

## Phase 0: 论文调研与算法确定（背景持续，与开发重叠）

> 已有初步想法和论文积累，调研在开发过程中持续进行，最终收敛到算法设计文档。

### Step 0.1 — 自进化Agent / Self-Improving LLM

待研究的问题：
- 现有自进化方案有哪些？（Self-Rewarding, SPIN, Self-Play, Constitutional AI 等）
- 训练数据构造方法：如何从交互中提取有效训练信号？
- 如何避免"自我强化错误"（model collapse）？多样性如何保持？

**产出**: `papers/` 目录下相关论文 + `docs/research/self_improving.md` 调研笔记

---

### Step 0.2 — LLM-as-Judge / 自动评估

待研究的问题：
- LLM打分可靠性研究（position bias, verbosity bias, self-enhancement bias）
- 多答案比较方法：pairwise vs pointwise vs listwise
- Rubric自动生成：如何让LLM自主归纳评估标准？
- 与人类评估的对齐程度

**产出**: `papers/` 目录下相关论文 + `docs/research/llm_judge.md` 调研笔记

---

### Step 0.3 — GRPO / RL for LLM 算法细节

待研究的问题：
- GRPO vs PPO vs DPO vs KTO：各自适用场景
- GRPO 中的 reward 设计：outcome reward vs process reward
- 多答案的 advantage 计算方式（已有 group normalization，是否需要改进？）
- 离线 vs 在线训练的取舍

**产出**: `papers/` 目录下相关论文 + `docs/research/grpo_rl.md` 调研笔记

---

### Step 0.4 — 算法方向确定

基于调研结论，确定本项目的：
- 训练算法（GRPO / 变体 / 其他？）
- 评估方案（pairwise rubric / pointwise / 混合？）
- Rubric生成策略（从反馈中自动归纳的具体方案）
- 奖励函数设计（多维度 rubric 分数如何融合为单一 reward？）

**产出**: `docs/design/algorithm.md` — 算法设计文档

---

## Phase 1: veRL双模引擎改造

> 目标：基于veRL现有机制（sleep/wake + CheckpointEngine weight sync），
> 改造成"常驻推理服务 + 空闲训练"双模引擎。
>
> 预估：~2.5 周（W1-W2）

### Step A1 — 抽取 Rollout API Server（最难点）

**文件**: `verl/trainer/serve_ppo.py`（新建，参考 `main_ppo.py`）

**要做的事：**

1. 分析 `verl/trainer/ppo/ray_trainer.py` 中 `fit()` 的 rollout 阶段（~1376行）
2. 将 rollout 从训练循环中解耦，做成常驻 API 服务：
   - 初始化 ActorRolloutRefWorker，让 rollout engine 保持在 wake 状态
   - 暴露 OpenAI-compatible `POST /v1/chat/completions` 接口
   - 复用 `LLMServerClient.generate()` 做推理
   - 跳过 dataloader/reward/critic 等训练组件

**核心改造点：**
- `RayPPOTrainer.fit()` 的 gen→sleep→train→wake 循环 → 改成 gen→gen→... (常驻)
- 用 FastAPI 包裹 `async_rollout_manager.generate_sequences()`

**验证方式：**
- 启动服务 → `curl POST /v1/chat/completions` → 返回模型生成结果
- 多次请求，每次独立返回不同采样结果
- 服务持续运行不退出

---

### Step A2 — 空闲检测 + 训练触发

**文件**: `trainable_openclaw/training/orchestrator.py`

**要做的事：**

1. **空闲检测器** — 记录最后请求时间，超过 N 秒判定空闲
2. **样本累积队列** — 推理过程中的对话自动入队
3. **训练触发条件** — 空闲 + 样本数 >= min_samples
4. **触发训练** — 调用 veRL 的训练逻辑（复用 `RayPPOTrainer.fit()` 的单步训练）

**核心改造点：**
- 推理模式：请求到达 → generate → 记录对话到 buffer
- 空闲检测到 → sleep_replicas() → 执行训练 → update_weights() → resume replicas
- 训练期间到达的请求 → 返回 503（或排队）

**验证方式：**
- 发几轮请求 → 停一段时间 → 观察日志确认训练被触发
- 训练完成后自动恢复推理
- 训练期间发请求返回 503

---

### Step A3 — 权重同步 + 恢复推理

**要做的事：**

1. 复用 veRL 的 `CheckpointEngineManager.update_weights()` 流程
2. 训练完成 → merge LoRA → sync weights → resume
3. 处理训练失败回滚（保留旧权重继续推理）

**验证方式：**
- 训练前后对同一 prompt 采样 → 输出分布应有变化
- 训练失败后服务不崩溃，继续用旧权重推理

---

## Phase 1.5: 数据工程与模拟测试环境

> 目标：在开发自进化评判系统之前，先搭建可控的模拟测试环境。
> 没有可靠的测试环境和评估手段，就无法判断 Phase 2 的 B1/B2/B3 是否做对了。

### 设计出发点：用强模型做老师，纠错对话就是训练数据

核心思路很简单——用一个强大的模型（DeepSeek-v4-flash）扮演"用户"，主动发现并纠正 Qwen3-4B 的错误，逐步引导它做出更好的回答。

```
一条 LMSYS 种子 prompt: "Write a sorting function in Python"

   User Sim Agent                          Qwen3-4B 推理服务
   (DeepSeek-v4-flash, 扮演挑剔的用户)       (被训练的模型)
   ─────────────────────                   ──────────────────
   
   ① 提出任务:
   "帮我写个排序函数"           ──────→     "def sort_list(a, b): ..."
                                          ↑ 变量名糟糕、无类型注解
   ② 发现错误，给出纠错:
   "a和b是什么意思？             ←──────   
   能不能用有意义的变量名？
   还有，加上类型注解"

   ③ Qwen3-4B 修正后:           ──────→     "def merge_sort(
                                              seq: list[int]
                                          ) -> list[int]: ..."
                                          ↑ 有进步，但没处理边缘情况
   ④ 继续纠错:
   "空列表传入会怎样？           ←──────   
   你考虑过吗？加上边界处理"

   ⑤ 最终版本:                  ──────→     正确处理空列表、类型完整、
                                          命名清晰、有文档字符串 ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这条 3 轮纠错对话包含:
  ✅ 训练数据: (bad_answer, correction, better_answer) × 3 轮
  ✅ 测试数据: 同类 prompt → 看模型是否不需要纠正就能答对
  ✅ Rubric 种子: "变量命名规范" "类型注解完整性" "边界条件处理"
  
  1 条 LMSYS prompt × User Sim 纠错能力 = 1 条高质量训练轨迹
  N 条 LMSYS prompts × 多种 persona = N 条不同角度的训练轨迹
```

**LMSYS 在这里的作用：**

```
LMSYS 提供:                                 LMSYS 不提供（也不需要）:
  ✅ 真实用户 prompt（15个类别 × N条）          ✗ quality_score（不需要）
  ✅ 话题分布统计（确保扩展不偏）                ✗ assistant 回答（不需要）
  ✅ 用户意图多样性（提问/指令/创作/...）        ✗ flaw 标注（User Sim 自己判断）
  
  → 从 LMSYS 抽 prompts 作为"种子问题"
  → User Sim 自己生成"正确答案"和"纠错路径"
  → 不需要 LMSYS 告诉我们"什么是好答案"
```

### 种子扩展流水线

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    种子扩展引擎（参考 AReaL-SEA 多 Agent 架构）             │
│                                                                          │
│  输入: LMSYS 中的 prompt（如 "Write a sorting function in Python"）       │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Step 1: 种子抽取 (Seed Extractor)                                │    │
│  │   从 LMSYS 按分层抽样抽取 prompts:                                │    │
│  │     • 按 15 个类别分层（coding / math / writing / ...）           │    │
│  │     • 按难度分层（问题复杂度 / 多轮程度）                          │    │
│  │     • 去重、过滤过短/无意义 prompt                                │    │
│  │   输出: 种子 prompt 池 (~5000 prompts, 覆盖 15 类别)              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Step 2: 场景构建 (Meta-Plan Agent)                                │    │
│  │   输入: 种子 prompt + LMSYS 话题分布                               │    │
│  │   为每条 prompt 生成 (persona, expected_correction_areas):         │    │
│  │     • persona: 用户角色 + 关注维度 + 严格程度                     │    │
│  │     • expected_areas: 该 prompt 类型常见的错误模式                │    │
│  │       （User Sim 不会"预设"错误，但知道从哪些维度检查）             │    │
│  │   输出: (prompt, persona, check_dimensions) 三元组                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Step 3: 交互生成 (Interaction Engine)                             │    │
│  │                                                                   │    │
│  │   ┌─────────────────────┐      ┌─────────────────────┐           │    │
│  │   │  User Sim Agent      │      │  Qwen3-4B 推理服务    │           │    │
│  │   │  (DeepSeek-v4-flash)  │ ←──→ │  (被训练对象)         │           │    │
│  │   │                      │      │                      │           │    │
│  │   │  ① 提出任务           │      │  ② 生成回答           │           │    │
│  │   │  ③ 按 persona 审查:   │      │  ④ 接收纠错 → 修改    │           │    │
│  │   │     发现错误 → 指正   │      │     继续完善           │           │    │
│  │   │     没有错误 → 通过   │      │                      │           │    │
│  │   │  ⑤ 最终确认 ✅        │      │                      │           │    │
│  │   └─────────────────────┘      └─────────────────────┘           │    │
│  │                                                                   │    │
│  │   输出: 完整纠错轨迹 (prompt → answer_v1 → correction_1 →        │    │
│  │          answer_v2 → correction_2 → ... → answer_final ✅)        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Step 4: 轨迹评估 (Trajectory Judge)                               │    │
│  │   评估维度:                                                        │    │
│  │     • 最终回答是否正确？(User Sim 是否最终满意)                   │    │
│  │     • 纠错效率: 0轮直接通过 / 1-2轮 / 3轮+                        │    │
│  │     • 每次纠错是否合理？（有具体问题，不是泛泛说"不好"）            │    │
│  │     • 改进是否定向？（不是胡乱改，而是针对指出的问题）              │    │
│  │   输出: SUCCESS/FAIL + 纠错轮次 + 纠错维度标签                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ├── SUCCESS → 存入 ConversationStore                              │
│       │             导出训练数据 (bad→correction→good 三元组)            │
│       │             提取纠错维度 → 积累为 Rubric 种子                    │
│       │                                                                  │
│       └── FAIL → Reflection Module                                       │
│                 分析: 为什么纠错失败？                                    │
│                 - Qwen3-4B 能力不够？→ 标记为"难"，积累后触发训练        │
│                 - User Sim 纠错不清？→ 改进 User Sim prompt              │
│                 - prompt 本身太难？→ 降低难度或换 prompt                 │
│                 下一轮生成更精准的纠错场景                                 │
└──────────────────────────────────────────────────────────────────────────┘
```

### 评估标准：纠错率取代质量分

不追求 LMSYS 的连续质量分（0-1），而是更简单、更可验证的指标：

```
传统方案:                      本方案:
  Judge 给回答打分 0.85          User Sim 与模型交互，数一数
  问: 0.85 算高还是低？          用户纠正了几次？
  答: 不知道，要对比基线         问: 纠错次数下降了吗？
                                答: 下降了 → 模型在进步 ✓

训练奖励信号:
  旧: reward = quality_score (连续值，需要校准)
  新: reward = 1 - (纠错轮次 / 最大轮次)  
      或者更简单: 0轮纠错=1, ≥3轮=0
      → 二值/序数值，不需要 LMSYS 校准
```

### Step S1 — LMSYS 种子数据抽取

**来源**: LMSYS-Chat-1M（已导入 ConversationStore，24.6 万训练 + 2.7 万测试）

**做什么**: 从 LMSYS 中抽取高质量的 prompt 作为种子，不关心 LMSYS 的 assistant 回答和 quality_score。

**要做的事：**

1. 从 ConversationStore 按分层抽样抽取种子 prompt 池：
   - 按 15 个类别分层（explanation / coding / math / writing / ...）
   - 每个类别抽 200-400 条，总池 ~5000 prompts
   - 过滤：过短的（<10字）、纯闲聊的、内容不完整的
   - 去重：语义相似度 > 0.9 的只保留一条
2. 为每条 prompt 标注：
   - `category`: LMSYS 原始分类
   - `complexity`: simple / moderate / complex（由 LLM 分析 prompt 复杂性）
   - `expected_dimensions`: 该任务的评估维度（如 coding → correctness/naming/types/edge_cases）
3. 建立种子 prompt 索引，供 Meta-Plan Agent 使用

**产物**: `data/seed_prompts.jsonl` — 分层抽样的种子 prompt 池

**验证**: 5000 条种子覆盖 15 个类别，每类 >= 200 条

---

### Step S2 — 用户模拟 + 纠错交互生成

**核心**: User Sim Agent (DeepSeek-v4-flash) 扮演挑剔的用户，主动发现并纠正 Qwen3-4B 的错误。

**要做的事：**

1. Meta-Plan Agent 为每条种子 prompt 生成交互方案：
   - 选择合适的 persona（代码审查者 / 数学老师 / 编辑 / 安全专家 / ...）
   - 设定该 persona 的检查维度（e.g., 代码审查者 → 正确性 + 命名 + 类型 + 边界）
   - 生成 User Sim 的 system prompt（准确描述角色和纠错标准）
2. Interaction Engine 执行多轮纠错交互：
   - User Sim 提出任务 → Qwen3-4B 生成 → User Sim 审查 → 纠错 → Qwen3-4B 修改 → ... → User Sim 满意或放弃
   - 每轮记录：(turn, speaker, content, correction_type_if_any)
3. 纠错规则：
   - User Sim 不能"预设错误"——必须基于 Qwen3-4B 的实际输出指出问题
   - 纠错必须具体，不能泛泛说"不够好"
   - 同一维度不重复纠错（3 轮后还不改 → 标记 FAIL，不再纠）

**产物**:
- `trainable_openclaw/simulation/user_sim.py` — User Sim Agent
- `trainable_openclaw/simulation/engine.py` — 多轮交互引擎
- `data/correction_trajectories.jsonl` — 纠错轨迹数据集

**验证**:
- 100 条种子 prompt × 3 种 persona = 300 条纠错轨迹
- 每条轨迹包含 ≥ 1 轮交互
- User Sim 纠错抽查 50 条，合理率 > 80%

---

### Step S3 — 轨迹评估与数据导出

**要做的事：**

1. Trajectory Judge 评估每条纠错轨迹：
   - 分级: `direct_pass`（无需纠正）/ `corrected`（纠正后通过）/ `partial`（部分纠正）/ `failed`（纠正失败）
   - 提取纠错维度标签（该轨迹涉及了哪些纠错点）
2. 数据分流：
   - `direct_pass + corrected` → 训练集（正例：模型最终做对了）
   - `partial` → 训练集（带纠错标签的部分正确样本）
   - `failed` → 分析集（留给 Reflection 分析根因）
3. 格式化导出：
   - 训练数据: `(prompt, bad_answer, correction, good_answer)` 四元组
   - Rubric 种子: 从纠错维度中聚合高频问题（如 "变量命名不规范" 出现 50 次 → 生成 rubric）

**产物**:
- `data/train_correction_pairs.jsonl` — 训练数据
- `data/test_prompts.jsonl` — 测试用 prompt（不含 LMSYS 参考回答）
- `data/rubric_seeds.json` — 从纠错中聚合的 rubric 种子

**验证**:
- 训练集 ≥ 200 条（direct_pass + corrected）
- 每个类别有 ≥ 5 条纠错维度标签

---

### Step S4 — 反思与持续优化

**参考**: AReaL-SEA 的 Reflection Module + Closed-Loop Evolution

**要做的事：**

1. Reflection Agent 分析 FAIL 轨迹：
   - 根因分类：模型能力不足 / User Sim 纠错不合理 / prompt 歧义
   - 模型能力不足 → 增加该类型 prompt 的训练权重
   - User Sim 不合理 → 更新 User Sim prompt，改进纠错策略
2. 迭代优化：
   - 每轮 Reflection 后更新 Meta-Plan 和 User Sim 的 prompt
   - 下一轮生成的纠错轨迹质量提升（FAIL 率下降）
3. 持续积累：
   - 纠错维度标签 → B2 Rubric 生成器的输入
   - 训练数据积累 → 触发 GRPO 训练

**产物**:
- `trainable_openclaw/simulation/reflection.py` — 反思模块
- `configs/simulation/` — User Sim / Meta-Plan / Judge 的 prompt 模板（持续迭代）

**验证**:
- 3 轮迭代后 FAIL 率下降 ≥ 10%
- Reflection 输出有可操作的改进建议（非泛泛分析）

---

### Step S5 — 评估指标体系

**核心指标: 纠错率 (Correction Rate)**

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| **纠错率** | 需要纠正的交互 / 总交互 | 越低越好 → 模型在进步 |
| **平均纠错轮次** | Σ纠错轮次 / 需纠正的交互数 | 越低越好 → 模型改得快 |
| **一次通过率** | 无需纠正直接满意的交互 / 总交互 | 越高越好 |
| **纠错闭环率** | 纠正后最终满意的交互 / 需纠正的交互 | 越高越好 → 模型能学会 |

**辅助指标:**

| 指标 | 用途 |
|------|------|
| 种子覆盖率 | 扩展场景覆盖了多少种子的纠错模式 | 确保扩展不偏离真实需求 |
| LMSYS 分布对齐度 | 扩展数据的类别/难度分布 vs LMSYS 真实分布 | 防止分布偏移 |
| User Sim 合理率 | 用户模拟的纠错质量（人工抽查） | 论文关键发现：差模拟器腐蚀训练 |
| 训练前后 Δ纠错率 | 训练后纠错率的变化 | 衡量自进化效果 |

**产出**:
- `trainable_openclaw/evaluation/metrics.py` — 指标计算模块

**验证方式**:
- 每个指标实现为独立函数
- 用 mock 数据验证计算正确性

---

## Phase 2: 自进化评判系统

> 核心理念：**Rubrics不是人工预设的，而是LLM从用户反馈中自主归纳生成的。**
>
> 流程：用户多轮反馈 → LLM分析反馈模式 → 自动生成严格评分任务 → 执行打分 → 随反馈积累演进
>
> 预估：~2 周（W4-W5）

### Step B1 — 用户反馈收集与分析

**文件**: `trainable_openclaw/evaluation/feedback.py`

**要做的事：**

1. 从对话日志中提取"用户修正/批评"的片段
2. LLM 分析模块：
   - 输入：多次反馈片段
   - 输出：反馈模式总结
   - 例如：`{"pattern": "变量命名不规范", "frequency": 5, "severity": "high"}`
3. 累积反馈模式库，按频率和严重程度排序

**验证方式：**
- 构造5轮带反馈的对话 → LLM分析 → 正确识别至少2个反馈模式

---

### Step B2 — LLM自主生成Rubrics

**文件**: `trainable_openclaw/evaluation/rubric.py`

**要做的事：**

1. **RubricGenerator** — 输入反馈模式 → LLM生成严格评分任务
2. 每条 Rubric 结构：
   - `id`, `title`, `prompt`（严格量化评分规则）, `source_feedback`, `version`
3. 关键约束：
   - prompt 必须严格、精确、可执行（LLM 打分者只需照做）
   - 评分标准量化（扣几分、什么条件），避免主观
   - 输出格式强制 JSON

**验证方式：**
- 输入反馈模式 → LLM生成 rubric → prompt 包含可量化规则
- N 个不同质量的答案分别打分 → 能排出名次

---

### Step B3 — Rubric执行器（LLM Judge）

**文件**: `trainable_openclaw/evaluation/judge.py`

**要做的事：**

1. `RubricExecutor` — 接收 rubric 列表 + 待评估回答 → 逐条执行打分
2. 每条 rubric 独立调用 deepseek-flash，严格按 rubric.prompt 执行
3. **GRPO多答案模式**：N 个答案 × M 条 rubrics → 每个答案得到 M 维分数向量
4. 随机打乱顺序消除 position bias

**验证方式：**
- 3条rubrics × 2个答案 → 每个答案得到3维分数 → 优者分高
- 打乱顺序后结果一致

---

### Step B4 — Rubric持续演进

**要做的事：**

1. 每积累 N 条新反馈触发 rubric 更新
2. 生命周期：匹配已有 → 更新 / 不匹配 → 新增 / 长期无用 → 归档
3. `RubricStore` — 持久化 + 版本管理

**验证方式：**
- 新反馈 → rubric 自动更新或新增
- 版本号正确递增

---

## Phase 3: 集成

> 预估：~1 周（W4-缓冲）

### Step C1 — 主循环串联

**文件**: `trainable_openclaw/pipeline.py`

实现完整闭环：

```
请求 → API Server(veRL rollout) → 回复用户
  ↓
日志记录 (session/对话)
  ↓
用户反馈收集 → LLM分析反馈模式
  ↓
LLM生成/更新Rubrics
  ↓
Rubric执行器 → GRPO多答案打分 → 排名 → 训练样本
  ↓
空闲检测 → 触发训练 → 权重同步 → 恢复推理
```

**验证方式：**
- 模拟完整流程：发3轮对话 → 评估 → 空闲 → 训练 → 继续推理

---

### Step C2 — Dashboard

**文件**: `trainable_openclaw/dashboard/app.py`

Streamlit 简易面板：当前模式、请求统计、训练记录、评估分数趋势

---

## Phase 4: 生产环境评估（一个月后）

> 目标：上线后在真实用户场景下量化系统效果，证明自进化训练确实提升了模型能力。
> 开发阶段的模拟验证已在 Phase 1.5 完成，Phase 4 关注生产环境表现。

### Step D1 — 测试集构建

1. 从真实使用日志中筛选高质量测试用例
2. 覆盖不同任务类型、难度、用户
3. 每个用例含 prompt + 参考标准 + 评估维度

### Step D2 — 效果评估体系

1. **离线评估** — 固定测试集，训练前后对比，Rubric 打分量化提升
2. **在线评估** — A/B 对比新旧模型，对比正/负反馈率
3. **退化检测** — 监控各维度分数，设置告警阈值

### Step D3 — 持续改进闭环

效果报告 → 分析薄弱维度 → 调整策略 → 再训练

---

## 进度记录

| Step | 内容 | 状态 | 日期 | 备注 |
|------|------|------|------|------|
| 0.1 | 自进化Agent调研 | ⬜ | | 背景进行 |
| 0.2 | LLM-as-Judge调研 | ⬜ | | 背景进行 |
| 0.3 | GRPO/RL算法调研 | ⬜ | | 背景进行 |
| 0.4 | 算法方向确定 | ⬜ | | 背景进行 |
| A1 | Rollout API Server | ✅ 已完成 | 2026-05-22 | FastAPI + vLLM Qwen3-4B, OpenAI-compatible |
| A2 | 空闲检测+训练触发 | ✅ 已完成 | 2026-05-22 | orchestrator + idle_timeout + min_samples |
| A3 | 权重同步+恢复推理+GRPO | ✅ 已完成 | 2026-05-25 | CheckpointEngineManager weight sync + GRPO训练闭环 |
| B0 | 对话日志系统 | ✅ 已完成 | 2026-05-29 | SQLite + WAL, sessions/messages 双表, CLI viewer, 34 测试 |
| **S1** | **LMSYS 种子数据抽取** | ✅ 已完成 | 2026-05-29 | 3200 prompts, 32 类别, 均衡分布 |
| **S2** | **用户模拟 + 纠错交互生成** | ✅ 已完成 | 2026-06-02 | 500 seeds → 557训练对 + 80测试对, 11类别 |
| **S3** | **轨迹评估与数据导出** | ✅ 已完成 | 2026-05-30 | 分级+格式化→训练对/rubric种子, 48训练对 |
| **S4** | **反思与持续优化** | ⬜ | | Reflection → 更新 prompts → FAIL率下降 |
| **S5** | **评估指标体系** | 🟡 进行中 | 2026-06-02 | 训练前基线已建立 (baseline_eval.json) |
| B1 | 用户反馈收集与分析 | ✅ 已完成 | 2026-05-30 | LLM分析 7模式 + simple模式 14维度 |
| B2 | LLM自主生成Rubrics | ✅ 已完成 | 2026-05-31 | 优化至8条动态rubric (rubrics_dynamic, category-aware) |
| B3 | Rubric执行器(Judge) | ✅ 已完成 | 2026-06-02 | Sync API + merged scoring, 5 bug修复, 正式训练运行中 |
| B4 | Rubric持续演进 | ⬜ | | 缓冲 |
| C1 | 主循环串联 | 🟡 进行中 | 2026-06-02 | 动态rubric + checkpoint + 496 prompts全闭环, 5轮训练中 |
| C2 | Dashboard | ⬜ | | 缓冲 |
| D1 | 测试集构建 | ✅ 已完成 | 2026-06-02 | 80 test prompts, 11类别, train/test零重叠 |
| D2 | 效果评估体系 | 🟡 进行中 | 2026-06-02 | baseline_eval.json + 训练后Δ纠错率待测 |
| D3 | 持续改进闭环 | ⬜ | | 一个月后 |

### Phase 1 里程碑总结（2026-05-25）

**A1+A2+A3 全部完成**，共 59 个集成测试全部通过（33 mock + 26 GPU）。

**关键成果：**
- veRL 常驻推理 API 服务（FastAPI + vLLM HYBRID 模式，Qwen3-4B + LoRA rank=16）
- 空闲检测 + 训练编排器（idle_timeout → 自动触发 GRPO 训练）
- GRPO 训练闭环（生成 → 奖励 → sleep → old_log_probs → advantage → update → weight sync → wake）
- 与 veRL 原生实现对齐：reward 最后 token 放置、原生 GSM8K reward 函数、config 驱动 mini-batch、Tracking 日志（console + tensorboard）
- End-to-end GSM8K 20-step 训练验证通过（batch=64, 90min, 无 OOM 无崩溃）
- GSM8K 耗尽保护（训练失败不标记耗尽，可重试）

**启动脚本：**
| 脚本 | 用途 |
|------|------|
| `scripts/run_serve_ppo.sh` | 生产启动（GSM8K enabled, idle_timeout=30） |
| `scripts/run_serve_ppo_test.sh` | 测试启动（低阈值, GSM8K disabled） |
| `scripts/run_gsm8k_e2e_test.sh` | GSM8K 端到端测试（20 steps, batch=64） |

### Phase 1.5 S2 进度（2026-05-30）

**S1 种子抽取已完成，S2 仿真流水线搭建完成，训练集 100 通已完成，测试集 50 通生成中。Phase 2 评估模块全部开发完成并验证通过。**

**关键成果：**
- `scripts/run_simulation.py` — 自包含仿真脚本（~800 行），支持真实/模拟/空跑/统计四种模式
- 5 种用户画像均衡覆盖 32 个 LMSYS 类别（之前 53% 单一画像，已修复）
- 多轮纠错对话引擎：User Sim (DeepSeek) ↔ Qwen3-4B，串行逐条执行
- Messages 格式记录：`[{role, content, reasoning}]`，`<think>` 提取到 `reasoning`
- GPU 服务器稳定运行：RTX 4080 SUPER, serve_ppo HYBRID 模式, Qwen3-4B + LoRA

**数据集制作：**
| 文件 | 数量 | 状态 |
|------|------|------|
| `data/seed_prompts.jsonl` | 3200 prompts, 32 类别 | ✅ 已完成 |
| `data/seed_train_100.jsonl` | 100 prompts (打乱) | ✅ 已拆分 |
| `data/seed_test_50.jsonl` | 50 prompts (打乱) | ✅ 已拆分 |
| `data/train_trajectories.jsonl` | 100 通纠错对话 | ✅ 已完成 (闭环率 99%, 平均纠错 0.49) |
| `data/test_trajectories.jsonl` | 50 通纠错对话 | ✅ 已完成 |
| `data/training_pairs.jsonl` | 48 个训练三元组 | ✅ S3 已产出 |
| `data/positive_examples.jsonl` | 67 个直接通过样例 | ✅ S3 已产出 |
| `data/rubric_seeds.json` | 14 个纠错维度 | ✅ S3 已产出 |
| `data/phase3_datasets/train_prompts.jsonl` | **557 pairs / 496 unique** | ✅ 2026-06-02 | 训练集 (去重叠后) |
| `data/phase3_datasets/test_prompts.jsonl` | **80 pairs / 78 unique** | ✅ 2026-06-02 | 测试集 (与训练零重叠) |
| `data/phase3_datasets/baseline_eval.json` | 训练前纠错率基线 | ✅ 2026-06-02 | per_category + per_prompt |
| `data/rubrics_dynamic.json` | **8 条动态 category-aware Rubric** | ✅ 2026-06-02 | 按类别匹配，替换rubrics_v2 |

**Phase 2 评估模块全部完成：**
| 模块 | 文件 | 状态 |
|------|------|------|
| S3 轨迹评估 | `evaluation/trajectory_eval.py` | ✅ 完成并验证 |
| B1 反馈分析 | `evaluation/feedback.py` | ✅ 完成并验证 (simple + LLM 双模式) |
| B2 Rubric生成 | `evaluation/rubric.py` | ✅ 完成并验证 (LLM生成 + 模板回退) |
| B3 Judge执行 | `evaluation/judge.py` | ✅ 完成并验证 (GRPO reward 计算) |
| 流水线脚本 | `scripts/run_evaluation.py` | ✅ 完成并验证 (S3→B1→B2→B3) |

**后续流程：**
- 测试集 50 通生成完成 → 跑 S3 评估 + 完整流水线
- 对比训练集/测试集评估结果
- 修复 B3 Judge 验证中的 JSON 解析边界情况
- 将 Rubric 打分接入 GRPO reward 计算

### Phase 3 C1 集成进度（2026-05-31）

**Rubric Judge → GRPO Reward 闭环已走通。** serve_ppo 训练循环已从 GSM8K 数学奖励切换到通用 Rubric 奖励。

**关键成果：**
- `trainable_openclaw/training/reward_bridge.py` — RewardBridge 同步包装器，将异步 B3 JudgeExecutor 封装为 Ray actor 内可用的同步调用（`asyncio.run()`）
- `serve_ppo.py` 改造:
  - `_async_monitor_loop` 启动时从 `data/training_pairs.jsonl` 加载 48 条训练 prompt，从 `data/rubrics_v2.json` 加载 5 条 rubric
  - `_train_bridge` 每步随机选 8 个 prompt → vLLM 生成 8 个回答（rollout_n=8, 共 64 个答案）
  - RewardBridge 对 64 个答案 × 5 条 rubric 评分 → mean 聚合 → reward
  - `train_step()` 使用 reward 进行 GRPO 训练（替代 GSM8K 数学答案提取）
- **配置**: `+trainer.trajectory.enabled=true` + `data_path` + `rubrics_path` + `api_key` + `max_rubrics` + `reward_mode`

**训练结果对比：**

| 轮次 | Rubric | Steps | Reward 范围 | Mean | 耗时 |
|------|--------|-------|-------------|------|------|
| Round 1 | 20 条旧 | 10 | 0.186 ~ 0.296 | 0.248 | 76 min |
| Round 2 | **5 条新** | 10 | 0.575 ~ **0.698** | **0.638** | 53 min |

**经验总结：**
- Rubric 质量 >> Rubric 数量 — 5 条精炼 rubric 效果远超 20 条
- GRPO advantage 信号仍偏弱（同 prompt 8 回答的 rubric 评分区分度有限），loss 波动在 -0.006~0.014
- lr=5e-6 偏低，10 步无明显收敛趋势 — 下一步增大 lr 到 1e-5，跑 20 步
- 每步选不同 prompt 导致 reward 震荡 — 可能是合理的（模型在多样化 prompt 上泛化），也可能是步间对比困难的噪声

### Phase 3 C1 进度更新（2026-06-02）

**数据规模大幅扩展 + 5 个关键 Bug 修复 + Checkpoint 机制完成 + 正式训练启动。**

**数据扩展：**
- 500 条种子 → 570 训练对 + 80 测试对（11 类别均衡覆盖）
- 去重叠后: 557 训练对 / 496 unique prompts，与测试集零重叠

**动态 Rubric 系统：**
- 8 条 category-aware rubric（`rubrics_dynamic.json`），训练时按 prompt 类别动态匹配
- 替换了固定的 `rubrics_v2.json`（5 条通用 rubric）

**5 个关键 Bug 修复：**

| Bug | 根因 | 修复 |
|-----|------|------|
| Ray event loop 冲突 | `asyncio.run()` 在 Ray uvloop 内冲突 → reward=0 | judge.py 新增完整 sync API (openai.OpenAI + ThreadPoolExecutor) |
| Merged JSON 截断 | 8 rubric 合并超过 max_tokens=800 | 动态缩放: `max(800, len(rubrics)*200)` |
| 数据字段不匹配 | 文件用 `prompt`，代码期望 `种子提示词` | 双字段兼容 |
| Rubric 字段不匹配 | `from_dict()` 收到未知字段 | 过滤已知 dataclass 字段 |
| Train/Test 泄漏 | 13 prompts 重叠 | `fix_leak.py` 从训练集移除 → 零重叠 |

**模型 Checkpoint 保存：**
- `ServeRunner.save_checkpoint()` — 委托 veRL FSDPCheckpointManager 保存
- 配置: 每 10 步保存，保留 3 个，目录 `/data/wangye/trainable-openclaw/checkpoints/`
- 产物: LoRA adapter (~631KB) + HF config/tokenizer
- 测试脚本 `scripts/test_checkpoint.sh` 验证通过

**正式训练 (Round 3)：**
- **配置**: 5 rounds × 10 steps, 48 prompts/step × 4 rollouts = 192 answers/step, lr=5e-6
- **数据**: 496 unique prompts, 动态 8-category rubric, 与测试集零重叠
- **Checkpoint**: 每 10 步保存，保留 3 个
- **远程**: `connect.westc.seetacloud.com:13738`, PID 152756, RTX 4090
- **日志**: `/tmp/phase3_train.log` (服务), `/tmp/serve_ppo_train.log` (训练详情)
- **预估**: ~9 小时 (50 steps)

**下一步：**
- 训练完成后运行纠错率评估 (`baseline_eval.json` vs 训练后) → 计算 Δ纠错率
- 对比 Round 3 (动态 rubric, 496 prompts) vs Round 2 (固定 rubric, 48 prompts)
- 日志驱动自动 rubric 更新 (B4)
