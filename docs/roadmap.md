# 研发路线图

## 项目目标

打造一个**自进化 AI 助手引擎**。核心闭环：

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

## 原则

- **单线程推进** — 一个人开发，严格串行
- **直接改 veRL 源码，不写 mock** — 深度改造，mock 没有意义
- **每步可独立验证** — 写完 → Linux 验证通过 → 再下一步
- **先简单实现，再复杂优化** — 每步只做最小可用版本

---

## 阶段总览

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 0 | 论文调研与算法确定 | 背景持续 |
| Phase 1 | veRL 双模引擎改造 | ✅ 完成 |
| Phase 1.5 | 数据工程与模拟测试环境 | 🟡 S4/S6 待做 |
| Phase 2 | 自进化评判系统 | ✅ 完成 |
| Phase 3 | 集成与 Dashboard | ✅ 完成 |
| Phase 4 | 生产环境评估 | 🟡 D3 远期 |

---

## 进度总览

| Step | 内容 | 状态 | 日期 | 说明 |
|------|------|------|------|------|
| 0.1 | 自进化 Agent 调研 | ⬜ | | 背景持续 |
| 0.2 | LLM-as-Judge 调研 | ⬜ | | 背景持续 |
| 0.3 | GRPO/RL 算法调研 | ⬜ | | 背景持续 |
| 0.4 | 算法方向确定 | ⬜ | | 背景持续 |
| A1 | Rollout API Server | ✅ | 05-22 | FastAPI + vLLM Qwen3-4B, OpenAI 兼容 |
| A2 | 空闲检测 + 训练触发 | ✅ | 05-22 | orchestrator + idle_timeout + min_samples |
| A3 | 权重同步 + GRPO 训练 | ✅ | 05-25 | CheckpointEngineManager + GRPO 闭环 |
| B0 | 对话日志系统 | ✅ | 05-29 | SQLite + WAL, sessions/messages 双表, CLI viewer |
| S1 | LMSYS 种子数据抽取 | ✅ | 05-29 | 3200 prompts, 32 类别, 均衡分布 |
| S2 | 用户模拟 + 纠错交互生成 | ✅ | 06-02 | 500 seeds → 557 训练对 + 80 测试对, 11 类别 |
| S3 | 轨迹评估与数据导出 | ✅ | 05-30 | 分级 + 格式化 → 训练对 / rubric 种子 |
| S4 | 反思与持续优化 | ⬜ | | Reflection → 更新 prompts → FAIL 率下降 |
| S5 | 评估指标体系 | ✅ | 06-03 | metrics.py, 27 tests, 4 个 dataclass |
| S6 | Agent 框架适配调研 | ⬜ | | open-claw / harness 对话格式与日志结构 |
| B1 | 用户反馈收集与分析 | ✅ | 05-30 | LLM 分析 7 模式 + simple 模式 14 维度 |
| B2 | LLM 自主生成 Rubrics | ✅ | 05-31 | 8 条动态 category-aware rubric |
| B3 | Rubric 执行器 (Judge) | ✅ | 06-03 | Sync API + merged scoring + 真实 API 验证 |
| B4 | Rubric 持续演进 | ✅ | 06-03 | rubric_evolver.py, 25 tests, 远程 e2e |
| C1 | 主循环串联 (Pipeline) | ✅ | 06-03 | pipeline.py, 20 tests, GPU e2e, CLI 三种模式 |
| C2 | Dashboard | ✅ | 06-03 | dashboard.py, 6 tests, Streamlit 启动验证 |
| D1 | 测试集构建 | ✅ | 06-02 | 80 test prompts, 11 类别, train/test 零重叠 |
| D2 | 效果评估体系 | ✅ | 06-03 | baseline + post-eval ckpt10 对比完成 |
| D3 | 持续改进闭环 | ⬜ | | 远期 (Phase 4) |

---

## Phase 0 — 论文调研与算法确定

背景持续，与开发重叠。待研究：

- **0.1 自进化 Agent**: Self-Rewarding, SPIN, Self-Play, Constitutional AI 等方案对比
- **0.2 LLM-as-Judge**: position bias, verbosity bias, pairwise vs pointwise, 与人工评估对齐
- **0.3 GRPO/RL 算法**: GRPO vs PPO vs DPO, outcome reward vs process reward
- **0.4 算法方向确定**: 综合调研结论 → `docs/design/algorithm.md`

---

## Phase 1 — veRL 双模引擎改造 ✅

### A1 — Rollout API Server
将 veRL 的 rollout 阶段抽取为常驻 API 服务。FastAPI + vLLM HYBRID 模式，暴露 OpenAI 兼容 `/v1/chat/completions` 接口。

**实现**: `verl-main-0516/verl/trainer/serve_ppo.py`

### A2 — 空闲检测 + 训练触发
后台监控线程，空闲超时 + 样本累计 → 自动触发训练。训练期间返回 503。

**实现**: `trainable_openclaw/training/orchestrator.py` (24 tests)

### A3 — 权重同步 + GRPO 训练
训练完成后 weight sync → 恢复推理。支持 GSM8K 数学奖励和通用 Rubric 奖励两种模式。

**实现**: serve_ppo 内 `train_step()` + `_train_bridge` + `CheckpointEngineManager`

---

## Phase 1.5 — 数据工程与模拟测试环境

核心理念：用强模型 (DeepSeek-v4-flash) 扮演挑剔用户，主动发现并纠正 Qwen3-4B 的错误，逐步引导它做出更好的回答。纠错次数作为训练信号——纠错次数下降 = 模型在进步。

### S1 — LMSYS 种子数据抽取 ✅
从 LMSYS-Chat-1M 分层抽样 3200 条 prompt，32 个类别均衡覆盖。

**产物**: `data/seed_prompts.jsonl`

### S2 — 用户模拟 + 纠错交互生成 ✅
5 种用户画像，User Sim ↔ Qwen3-4B 多轮纠错对话。500 条种子跑出 557 训练对 + 80 测试对，11 类别。

**产物**: `scripts/run_simulation.py`, `data/phase3_datasets/`

### S3 — 轨迹评估与数据导出 ✅
纠错轨迹分级 (direct_pass / corrected / partial / failed)，提取训练三元组和 rubric 种子维度。

**产物**: `trainable_openclaw/evaluation/trajectory_eval.py`

### S4 — 反思与持续优化 ⬜
Reflection Agent 分析 FAIL 轨迹根因 → 改进 User Sim prompt → 降低 FAIL 率。

**产物** (待): `trainable_openclaw/simulation/reflection.py`

### S5 — 评估指标体系 ✅
4 个 dataclass (JudgeQuality / ModelImprovement / RubricQuality / Convergence)，Spearman 相关性、accuracy@k、覆盖率、收敛检测。

**产物**: `trainable_openclaw/evaluation/metrics.py` (27 tests)

### S6 — Agent 框架适配调研 ⬜
调研 open-claw、harness 等 Agent 框架的对话格式和日志结构，对比当前仿真格式的差异，设计适配方案。

**产物** (待): `docs/research/agent_frameworks.md`, `docs/design/agent_adapter.md`

---

## Phase 2 — 自进化评判系统 ✅

### B0 — 对话日志系统 ✅
SQLite + WAL 模式，sessions/messages 双表，CLI 查看器。

**产物**: `trainable_openclaw/logging/conversation_store.py` (23 tests)

### B1 — 用户反馈收集与分析 ✅
从对话日志提取反馈片段 → LLM 分析 → 识别高频反馈模式。

**产物**: `trainable_openclaw/evaluation/feedback.py`

### B2 — LLM 自主生成 Rubrics ✅
反馈模式 → LLM 生成严格量化评分 prompt → RubricStore 持久化 + 版本管理。

**产物**: `trainable_openclaw/evaluation/rubric.py`, `rubric_engine.py`

### B3 — Rubric 执行器 (LLM Judge) ✅
多 Rubric 合并评分 (省 5x API 调用)，sync API 兼容 Ray actor，动态 max_tokens 防截断。

**产物**: `trainable_openclaw/evaluation/judge.py` (真实 API 验证通过)

### B4 — Rubric 持续演进 ✅
低分样本触发演进，生命周期管理 (匹配更新 / 新增 / 归档)。

**产物**: `trainable_openclaw/evaluation/rubric_evolver.py` (25 tests + 远程 e2e)

---

## Phase 3 — 集成 ✅

### C1 — Pipeline 主循环 ✅
串联 pre-eval → 训练 → post-eval → rubric 演进。CLI 三种模式 (`--eval-only` / `--gen-config` / `--evolve-rubrics`)。

**产物**: `trainable_openclaw/pipeline.py` (20 tests + GPU e2e)

### C2 — Dashboard ✅
Streamlit 面板：模型状态、训练进度、评估分数、Rubric 统计。

**产物**: `scripts/dashboard.py` (6 tests)

---

## Phase 4 — 生产环境评估

### D1 — 测试集构建 ✅
80 test prompts，11 类别，与训练集零重叠。

### D2 — 效果评估体系 ✅
纠错率 (Correction Rate) 为核心指标。基线评测 + 训练后对比完成。

**产物**: `trainable_openclaw/evaluation/correction_rate.py`

### D3 — 持续改进闭环 ⬜
效果报告 → 分析薄弱维度 → 调整策略 → 再训练。远期规划。

---

## 关键数字

| 指标 | 数值 |
|------|------|
| 总 commits | 49 |
| 单测通过 | 154 (6 个文件) |
| 代码行数 (trainable_openclaw/) | ~3,600 |
| 仿真训练对 | 557 (496 unique prompts, 11 类别) |
| 测试集 | 80 prompts, 与训练零重叠 |
| 动态 Rubric | 8 条 category-aware |

---

## 训练实验记录

### Round 1 — 20 条旧 Rubric (基线)
- 8p×8r=64 answers/step, lr=3e-6, 10 steps
- reward: 0.186~0.296, mean=0.248, 76 min
- 问题: rubric 太多太泛，区分度低

### Round 2 — 5 条优化 Rubric
- 8p×8r=64, lr=5e-6, 10 steps
- reward: 0.575~0.698, mean=0.638, 53 min
- Rubric 质量 >> 数量

### Round 3 — 正式训练 (动态 Rubric, 496 prompts)
- 48p×4r=192 answers/step, lr=5e-6, 42 steps
- Phase 1 (steps 1-15): 小 batch debug, reward 0~0.35
- Phase 2 (steps 16-42): 全 batch, mean reward=0.184, 轻微下行
- **Checkpoint 问题**: 仅 step_10 保存成功，Phase 2/3 检查点丢失

### Checkpoint step_10 评测

| 指标 | 基线 (Qwen3-4B) | 训练后 (ckpt10) | Delta |
|------|-----------------|-----------------|-------|
| 纠错率 | 0.60 | 0.88 | **+0.28 (恶化)** |
| 直接通过 | 32 (40%) | 9 (11.5%) | -23 |
| 失败 | 12 (15%) | 35 (44.9%) | +23 |

**结论**: Qwen3-4B 容量不足，reward 信号无上升趋势。10/11 类别退化。

---

## 当前状态 & 遗留问题

**框架代码全部完成** (18/22 steps)，154 单测通过，远程 GPU 真实 API 验证通过。

**3 个遗留问题：**
1. **训练不收敛** — Qwen3-4B + GRPO，需要换 7B+ 模型、调整超参
2. **Checkpoint 跨重启丢失** — serve_ppo 重启后 save_ckpt_interval 未能继续触发
3. **S4/S6 待开发** — 反思模块 + Agent 框架适配调研

**下阶段方向：**
1. S6: Agent 框架适配调研 (open-claw / harness)
2. 换 7B+ 模型重跑训练，修复 checkpoint 机制
3. 训练收敛后补 S4 反思模块
