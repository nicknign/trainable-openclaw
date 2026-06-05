# 研发路线图

## 项目目标

打造一个**自进化 AI 助手引擎**。让 AI 助手在使用中持续改进——更好的工具选择、更少的无效步骤、更准确的任务完成。

```
用户请求 → Agent (nanobot/open-claw) → 多轮交互 → 完成任务
                                                    ↓
                                              对话日志存储
                                                    ↓
                                      LLM 分析 Agent 行为模式
                                                    ↓
                                    LLM 自主生成 Agent Rubrics
                                       (工具选择 / 步骤效率 /
                                        错误恢复 / 信息充分性)
                                                    ↓
                                    Rubrics 对 rollout 轨迹打分
                                                    ↓
                                    空闲检测 → GRPO 训练
                                                    ↓
                                    权重同步 → 恢复服务
                                                    ↓
                          Agent 能力持续进化 (skill improvement)
```

---

## 原则

- **单线程推进** — 一个人开发，严格串行
- **先简单实现，再复杂优化** — nanobot (轻量) → open-claw (生产级)
- **每步可独立验证** — 写完 → Linux 验证通过 → 再下一步

---

## 阶段总览

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 0 | 论文调研与算法确定 | 背景持续 |
| Phase 1 | veRL 双模引擎改造 | ✅ 完成 |
| Phase 1.5 | 数据工程与模拟测试环境 | 🟡 S4 待做 |
| Phase 2 | 自进化评判系统 | ✅ 完成 |
| Phase 3 | 集成与 Dashboard | ✅ 完成 |
| **Phase 4** | **Agent 引擎集成** | **⬜ 新阶段** |
| Phase 5 | 生产环境评估 | 🟡 D3 远期 |

---

## 进度总览

| Step | 内容 | 状态 | 日期 | 说明 |
|------|------|------|------|------|
| 0.1 | 自进化 Agent 调研 | ⬜ | | 背景持续 |
| 0.2 | LLM-as-Judge 调研 | ⬜ | | 背景持续 |
| 0.3 | GRPO/RL 算法调研 | ⬜ | | 背景持续 |
| 0.4 | 算法方向确定 | ⬜ | | 背景持续 |
| A1 | Rollout API Server | ✅ | 05-22 | FastAPI + vLLM, OpenAI 兼容 |
| A2 | 空闲检测 + 训练触发 | ✅ | 05-22 | orchestrator |
| A3 | 权重同步 + GRPO 训练 | ✅ | 05-25 | CheckpointEngineManager |
| B0 | 对话日志系统 | ✅ | 05-29 | SQLite + WAL |
| S1 | LMSYS 种子数据抽取 | ✅ | 05-29 | 3200 prompts |
| S2 | 用户模拟 + 纠错交互生成 | ✅ | 06-02 | 557 训练对 |
| S3 | 轨迹评估与数据导出 | ✅ | 05-30 | 分级 + 格式化 |
| S4 | 反思与持续优化 | ⬜ | | Reflection → FAIL 率下降 |
| S5 | 评估指标体系 | ✅ | 06-03 | metrics.py (27 tests) |
| B1 | 用户反馈收集与分析 | ✅ | 05-30 | 7 模式 + 14 维度 |
| B2 | LLM 自主生成 Rubrics | ✅ | 05-31 | 8 条动态 rubric |
| B3 | Rubric 执行器 (Judge) | ✅ | 06-03 | Sync + merged scoring |
| B4 | Rubric 持续演进 | ✅ | 06-03 | evolver (25 tests) |
| C1 | 主循环串联 (Pipeline) | ✅ | 06-03 | pipeline.py (20 tests) |
| C2 | Dashboard | ✅ | 06-03 | Streamlit (6 tests) |
| **T1** | **nanobot 调研与集成** | ⬜ | | 轻量 Agent 框架跑通 |
| **T2** | **Agent rollout 训练适配** | ⬜ | | veRL 多轮 Agent 轨迹生成 |
| **T3** | **Agent 场景 Judge 扩展** | ⬜ | | 工具选择/步骤效率/错误恢复评判 |
| **T4** | **open-claw 迁移** | ⬜ | | 生产级 Agent 框架切换 |
| D1 | 测试集构建 | ✅ | 06-02 | 80 test prompts |
| D2 | 效果评估体系 | ✅ | 06-03 | baseline + post-eval |
| D3 | 持续改进闭环 | ⬜ | | 远期 |

---

## Phase 0 — 论文调研与算法确定

背景持续，与开发重叠。

- **0.1 自进化 Agent**: Self-Rewarding, SPIN, Self-Play, Constitutional AI
- **0.2 LLM-as-Judge**: bias 分析, pairwise vs pointwise, 与人工评估对齐
- **0.3 GRPO/RL 算法**: GRPO vs PPO vs DPO, reward 设计
- **0.4 算法方向确定**: 综合结论 → `docs/design/algorithm.md`

---

## Phase 1 — veRL 双模引擎改造 ✅

### A1 — Rollout API Server
veRL rollout 阶段抽取为常驻 API 服务。FastAPI + vLLM HYBRID 模式，OpenAI 兼容接口。

**实现**: `verl-main-0516/verl/trainer/serve_ppo.py`

### A2 — 空闲检测 + 训练触发
空闲超时 + 样本累计 → 自动触发训练，训练期间返回 503。

**实现**: `trainable_openclaw/training/orchestrator.py` (24 tests)

### A3 — 权重同步 + GRPO 训练
train_step → weight sync → wake。支持 GSM8K 和 Rubric 两种奖励模式。

**实现**: serve_ppo 内 `train_step()` + `_train_bridge` + `CheckpointEngineManager`

---

## Phase 1.5 — 数据工程与模拟测试环境

> **已完成的工作基于单轮对话。** 用 DeepSeek-v4-flash 扮演挑剔用户，与 Qwen3-4B 做多轮纠错对话，生成训练数据。

### S1 — LMSYS 种子数据抽取 ✅
3200 条 prompt，32 类别均衡覆盖。

### S2 — 用户模拟 + 纠错交互生成 ✅
5 种用户画像，500 条种子 → 557 训练对 + 80 测试对，11 类别。

**产物**: `scripts/run_simulation.py`, `data/phase3_datasets/`

### S3 — 轨迹评估与数据导出 ✅
分级 (direct_pass / corrected / partial / failed)，提取训练三元组。

**产物**: `trainable_openclaw/evaluation/trajectory_eval.py`

### S4 — 反思与持续优化 ⬜
Reflection Agent 分析 FAIL 轨迹 → 改进 User Sim prompt → FAIL 率下降。

**产物** (待): `trainable_openclaw/simulation/reflection.py`

### S5 — 评估指标体系 ✅
JudgeQuality / ModelImprovement / RubricQuality / Convergence 四个 dataclass + Spearman / accuracy@k。

**产物**: `trainable_openclaw/evaluation/metrics.py` (27 tests)

---

## Phase 2 — 自进化评判系统 ✅

### B0 — 对话日志系统 ✅
SQLite + WAL，sessions/messages 双表，CLI viewer。`trainable_openclaw/logging/` (23 tests)

### B1 — 用户反馈收集与分析 ✅
LLM 从对话日志识别反馈模式。`trainable_openclaw/evaluation/feedback.py`

### B2 — LLM 自主生成 Rubrics ✅
反馈模式 → 量化评分 prompt → RubricStore 持久化。`evaluation/rubric.py`, `rubric_engine.py`

### B3 — Rubric 执行器 (Judge) ✅
合并评分 (省 5x API)，sync API 兼容 Ray。`evaluation/judge.py`

### B4 — Rubric 持续演进 ✅
低分触发演进，生命周期管理。`evaluation/rubric_evolver.py` (25 tests + e2e)

---

## Phase 3 — 集成 ✅

### C1 — Pipeline 主循环 ✅
pre-eval → 训练 → post-eval → rubric 演进。CLI 三种模式。`pipeline.py` (20 tests + GPU e2e)

### C2 — Dashboard ✅
Streamlit 面板。`scripts/dashboard.py` (6 tests)

---

## Phase 4 — Agent 引擎集成 ⬜

> **核心转变：从单轮纠错对话 → 多轮 Agent 交互。**
>
> 当前系统基于简单的 chat 格式 (`[{role, content}]`) 运作。Agent 场景完全不同：
> - 一轮交互包含多步：思考 → 工具调用 → 获取结果 → 回复用户
> - 评判维度扩展：工具选择合理性、步骤效率、错误恢复能力
> - veRL rollout 需要生成 Agent 轨迹，而非单次回答
>
> **策略：先用 nanobot 快速验证闭环，再迁移到 open-claw 做生产部署。**

### T1 — nanobot 调研与集成

**nanobot** 是一个轻量级 Agent 框架，适合快速集成和验证。

**要做的事：**

1. **nanobot 框架分析**
   - Agent 循环结构：消息流、工具注册与调用、记忆管理
   - 对话日志格式：一轮完整 Agent 交互包含哪些字段（thought / tool_call / tool_result / response）
   - 与当前 `ConversationStore` schema 的差异对比
2. **nanobot 实例搭建**
   - 在本项目内集成 nanobot，对接 veRL 推理后端
   - 配置基础工具集（文件操作、代码执行、网络搜索等）
   - 跑通一个完整的 Agent 交互流程，抓取日志样本
3. **日志格式对齐**
   - 确定 ConversationStore schema 扩展方案（新增 role 类型：`tool_call` / `tool_result` / `thinking`）
   - 写 adapter 将 nanobot 日志转为统一格式
4. **仿真管线适配**
   - User Sim 从"纠错对话"升级为"Agent 任务审查"
   - 模拟用户给 Agent 布置任务 → Agent 多步执行 → 用户审查结果 → 纠错

**产物**:
- `trainable_openclaw/agent/nanobot_adapter.py` — nanobot ↔ veRL 适配器
- `data/samples/nanobot_logs/` — 典型 Agent 交互日志样本
- `docs/research/nanobot_analysis.md` — nanobot 框架分析笔记

**验证**:
- nanobot + veRL 推理后端跑通，完成至少 1 个完整 Agent 任务
- Agent 日志成功写入 ConversationStore，字段无丢失

---

### T2 — Agent rollout 训练适配

**这是改造量最大的步骤。** 当前 veRL rollout 生成单轮回答 (`prompt → answer`)。Agent 训练需要生成多轮轨迹。

**当前模式 (单轮):**
```
veRL receives: "写一个排序函数"
veRL generates: "def sort(arr): ..."
Judge scores this single answer
```

**Agent 模式 (多轮):**
```
veRL receives: "帮我整理 Downloads 文件夹"
veRL generates a trajectory:
  ① thinking: "需要先列出文件，按类型分类"
  ② tool_call: list_files("~/Downloads")
  ③ tool_result: [file1.pdf, file2.txt, file3.jpg, ...]
  ④ thinking: "有 3 种类型，分别移到对应文件夹"
  ⑤ tool_call: move_file("file1.pdf", "~/Documents/PDFs")
  ⑥ tool_call: move_file("file2.txt", "~/Documents/Texts")
  ⑦ tool_call: move_file("file3.jpg", "~/Pictures")
  ⑧ response: "已整理完成：3 个文件已按类型移动到对应文件夹"

Judge scores the ENTIRE trajectory:
  - 工具选择: ✅ list_files + move_file 合理
  - 步骤效率: ⚠️ 逐个 move_file，应该批量
  - 错误处理: ⚠️ 没有检查文件是否已存在
```

**要做的事：**

1. **Agent 训练数据构造**
   - 用 nanobot + User Sim 生成 Agent 训练轨迹
   - 数据格式：`(task, trajectory, user_feedback)` 三元组
   - 轨迹包含多步 thought/tool_call/tool_result/response
2. **veRL rollout 改造**
   - serve_ppo 的 `_train_bridge` 支持 multi-turn rollout
   - 每步训练时，让 veRL 生成完整的 Agent 轨迹（而非单次回答）
   - 轨迹生成需要 tool execution 环境（sandbox / mock tools）
3. **奖励信号设计**
   - trajectory-level reward：整条轨迹的质量评分
   - step-level reward：每一步的工具选择/推理质量
   - 组合方式：mean / weighted / outcome-only
4. **训练数据格式标准化**
   - 定义 Agent 训练数据的标准 schema
   - 兼容 nanobot 和后续 open-claw 的日志格式

**产物**:
- `trainable_openclaw/agent/rollout.py` — Agent 多轮 rollout 引擎
- `trainable_openclaw/agent/training_data.py` — Agent 训练数据构造
- `docs/design/agent_training.md` — Agent 训练方案设计文档

**验证**:
- nanobot + veRL 完成 10 条 Agent 任务，生成完整轨迹
- rollout 引擎能稳定生成 3+ 步的 Agent 轨迹
- 轨迹日志可被 ConversationStore 完整记录

---

### T3 — Agent 场景 Judge 扩展

Agent 的评判维度与单轮对话完全不同。需要扩展 Rubric 体系和 Judge 能力。

**当前 Judge 评判维度 (单轮对话):**
- 事实准确性、逻辑正确性、完整性、格式规范、语言表达

**Agent Judge 新增维度:**

| 维度 | 说明 | 示例 |
|------|------|------|
| 工具选择合理性 | 是否选择了最合适的工具？有无更好替代？ | 用 `list_files` 而非 `ls` shell 命令 |
| 步骤效率 | 是否有多余步骤？能否合并？ | 逐个移动文件 vs 批量移动 |
| 错误恢复 | 工具失败后如何处理？ | 文件不存在 → 创建目录后重试 |
| 信息充分性 | 执行前是否收集了足够信息？ | 移动前未检查目标路径是否存在 |
| 安全边界 | 是否拒绝了危险操作？ | 拒绝 `rm -rf /` 并给出警告 |
| 用户交互 | 不确定时是否询问用户？ | 文件重名时让用户选择覆盖/跳过 |

**要做的事：**

1. **Agent Rubric 生成**
   - 从 Agent 轨迹反馈中提取 Agent 特有的错误模式
   - B2 RubricGenerator 扩展：支持 Agent 场景的评分 prompt 模板
   - 示例 Rubric: "工具选择合理性 — 检查 Agent 是否选用了最直接的工具完成任务，每处不当选择扣 2 分"
2. **Agent User Sim 升级**
   - User Sim 能审查 Agent 轨迹，指出具体的工具选择/步骤问题
   - 模拟用户反馈："你为什么要逐个移动文件？用通配符一次搞定不就行了？"
3. **Agent Judge 执行**
   - B3 JudgeExecutor 扩展：输入从单条 answer → 整条 trajectory
   - 轨迹评分 prompt 设计：如何把多步轨迹 + 工具调用上下文塞进评分 prompt
4. **Agent Rubric 演进**
   - B4 RubricEvolver 适配 Agent 场景的低分样本检测

**产物**:
- `trainable_openclaw/evaluation/agent_rubric.py` — Agent 专用 Rubric 模板
- `trainable_openclaw/evaluation/agent_judge.py` — Agent 轨迹评分器
- `docs/design/agent_judge.md` — Agent Judge 设计文档

**验证**:
- 3 条 Agent Rubric 可正确评估 nanobot 轨迹
- 优质轨迹得分 > 劣质轨迹得分 (区分度验证)

---

### T4 — open-claw 迁移

nanobot 闭环跑通后，迁移到功能更完整的 open-claw 框架。

**与 nanobot 的关键差异：**
- 更复杂的工具生态 (更多的内置工具、工具组合)
- 更长的上下文窗口 (可能需要上下文压缩)
- 多模态支持 (图片/文件输入)
- 更完善的安全机制

**要做的事：**

1. **open-claw 框架分析** — 架构、消息格式、工具系统、日志格式
2. **adapter 开发** — 将 T1 的适配层从 nanobot 切换到 open-claw，接口保持不变
3. **工具集映射** — nanobot 工具 → open-claw 工具的语义对应
4. **仿真管线适配** — User Sim 适配 open-claw 的 Agent 交互模式
5. **训练数据迁移** — nanobot 阶段积累的训练数据可在 open-claw 上复用

**产物**:
- `trainable_openclaw/agent/openclaw_adapter.py` — open-claw ↔ veRL 适配器
- `docs/research/openclaw_analysis.md` — open-claw 框架分析笔记

**验证**:
- open-claw + veRL 跑通与 nanobot 阶段相同的 Agent 任务
- adapter 接口无需修改即可切换后端

---

## Phase 5 — 生产环境评估

### D1 — 测试集构建 ✅
80 test prompts，11 类别，与训练集零重叠。

### D2 — 效果评估体系 ✅
纠错率为核心指标。baseline + post-eval 对比完成。`evaluation/correction_rate.py`

### D3 — 持续改进闭环 ⬜
效果报告 → 分析薄弱维度 → 调整策略 → 再训练。远期规划。

---

## 关键数字

| 指标 | 数值 |
|------|------|
| 总 commits | 49 |
| 单测通过 | 154 (6 个文件) |
| Phase 1-3 代码行数 | ~3,600 |
| 仿真训练对 | 557 (496 unique, 11 类别) |
| 动态 Rubric | 8 条 category-aware |

---

## 训练实验记录

| Round | Rubric | 配置 | Reward Mean | 结果 |
|-------|--------|------|-------------|------|
| 1 | 20 条旧 | 8p×8r=64, lr=3e-6 | 0.248 | 区分度低 |
| 2 | 5 条优化 | 8p×8r=64, lr=5e-6 | 0.638 | Rubric 质量 > 数量 |
| 3 | 8 条动态 | 48p×4r=192, lr=5e-6, 42 steps | 0.184 | 不收敛, ckpt10 退化 |
| 4 | 4 条 coding | 16p×4r=64, lr=1e-5, 104 steps | 0.588 | **response_length 是关键** |

**Round 4 详细 (2026/06/02-06/05, ~16h GPU)**:
- 前 73 步 rlen≤512（截断），reward 0.03-0.18，无效训练
- response_length 512→4096 后（Phase 5, 31 steps），reward 跳至 0.588
- 3 步移动平均 0.52-0.64 震荡，无明显单调趋势
- **根因**: rollout.yaml 默认 response_length=512 截断了 75% 的代码
- **教训**: 训练前必须检查 `rlen` 指标，确保回复完整
- Checkpoint step_20 在 Phase 5 保存（有效训练），LoRA 已提取

**当前结论**：Qwen3-4B + LoRA rank=16 在 coding 任务上有学习信号（loss 非零），但 31 步不足收敛。当前阶段聚焦功能开发，大规模训练调优留到后续。

---

## 当前状态

**Phase 1-3 完成** (18/21 steps)，154 单测通过，框架代码可运行。4 轮训练实验完成。

**Phase 4 启动中** — Agent 引擎集成是当前核心工作：

1. T1 nanobot 集成 — 快速跑通 Agent 闭环
2. T2 rollout 改造 — Agent 多轮轨迹生成
3. T3 Judge 扩展 — Agent 评判维度
4. T4 open-claw 迁移 — 生产级框架

**训练收敛问题** — 当前阶段聚焦功能开发，后续阶段解决：7B+ 模型、更多步数、pairwise reward。
