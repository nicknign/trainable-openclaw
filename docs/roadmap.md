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
| Phase 2 | 自进化评判系统 | 1 | 核心难点，后做 |
| Phase 3 | 集成与Dashboard | 1+2 | 串联 |
| Phase 4 | 测试集与效果评估 | 3 | 框架完成后 |

## 时间线（单线程，目标 5/16 → 6/16）

```
Week 1 (5/16-5/22):
  A1: Rollout API Server ───────────────── 最难点，集中攻克
  Phase 0: 论文调研 ────────────────────── 背景进行

Week 2 (5/23-5/29):
  A2: 空闲检测 + 训练触发
  A3: 权重同步 + 恢复推理
  Phase 0: 算法方向确定 ────────────────── 背景进行

Week 3 (5/30-6/5):
  B1: 用户反馈收集与分析
  B2: LLM自主生成Rubrics

Week 4 (6/6-6/12):
  B3: Rubric执行器 (LLM Judge)
  C1: 主循环串联 ───────────────────────── 关键里程碑

缓冲 (6/13-6/16):
  B4: Rubric演进 (基本版)
  C2: Dashboard (简化版)
  联调、修bug、写文档
```

### MVP交付物（6/16）

- veRL 常驻推理 API 服务可运行
- 空闲时自动触发训练，训练完恢复推理
- LLM 从用户反馈中生成 Rubrics 并执行打分
- 核心闭环走通：请求 → 日志 → 评判 → 训练 → 恢复
- 简易 Dashboard

### 推迟到一个月后

- Phase 4 (D1/D2/D3)：测试集构建、效果评估体系
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

## Phase 2: 自进化评判系统

> 核心理念：**Rubrics不是人工预设的，而是LLM从用户反馈中自主归纳生成的。**
>
> 流程：用户多轮反馈 → LLM分析反馈模式 → 自动生成严格评分任务 → 执行打分 → 随反馈积累演进
>
> 预估：~2 周（W3-W4）

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

## Phase 4: 测试集与效果评估（一个月后）

> 目标：量化系统效果，证明自进化训练确实提升了模型能力。

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
| A1 | Rollout API Server | ✅ 已完成 | 2026-05-22 | FastAPI + vLLM Qwen3-4B, OpenAI-compatible, 12 GPU集成测试通过 |
| A2 | 空闲检测+训练触发 | ✅ 已完成 | 2026-05-22 | orchestrator + idle_timeout + min_samples, 5 GPU集成测试通过 |
| A3 | 权重同步+恢复推理+GRPO | ✅ 已完成 | 2026-05-25 | CheckpointEngineManager weight sync + GRPO训练闭环, 9 GPU集成测试通过 |
| B0 | 对话日志系统 | ✅ 已完成 | 2026-05-29 | SQLite + WAL, sessions/messages 双表, CLI viewer, 23 测试通过 |
| B1 | 用户反馈收集与分析 | ⬜ | | W3 |
| B1.2 | OASST2 数据处理 | 🟡 进行中 | 2026-05-29 | 数据集下载/解析/划分, 模拟用户反馈, 基础模型评测 |
| B2 | LLM自主生成Rubrics | ⬜ | | W3 |
| B3 | Rubric执行器(Judge) | ⬜ | | W4 |
| B4 | Rubric持续演进 | ⬜ | | 缓冲 |
| C1 | 主循环串联 | ⬜ | | W4 关键里程碑 |
| C2 | Dashboard | ⬜ | | 缓冲 |
| D1 | 测试集构建 | ⬜ | | 一个月后 |
| D2 | 效果评估体系 | ⬜ | | 一个月后 |
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
