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
| **Phase 4** | **Agent 引擎集成** | **🟡 T1 完成, T2 详细计划已出** |
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
| **R1** | **MUA-RL 调研** | ✅ | 06-12 | loss masking + 稀疏奖励, 互补不重复 |
| **T1** | **nanobot 调研与集成** | ✅ | 06-10 | WebUI 修复 + strip_think patch |
| **T2.1** | **Loss Masking 移植** | ⬜ | | ~200行, 防 reward hacking |
| **T2.2** | **稀疏二元奖励** | ⬜ | | ~100行, 可验证任务 0/1 |
| **T2.3** | **场景+冷启动数据生成** | ⬜ | | 5场景, 500-1000条轨迹 |
| **T2.4** | **SFT 冷启动训练** | ⬜ | | 混合数据源, 3 epoch |
| **T2.5** | **Agent GRPO 训练** | ⬜ | | 16p×4r, loss masking |
| **T2.6** | **Agent 评测** | ⬜ | | BFCL v4 + tau-bench |
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

**产出**: `scripts/run_simulation.py`, `data/phase3_datasets/`

### S3 — 轨迹评估与数据导出 ✅
分级 (direct_pass / corrected / partial / failed)，提取训练三元组。

**产出**: `trainable_openclaw/evaluation/trajectory_eval.py`

### S4 — 反思与持续优化 ⬜
Reflection Agent 分析 FAIL 轨迹 → 改进 User Sim prompt → FAIL 率下降。

**产出** (待): `trainable_openclaw/simulation/reflection.py`

### S5 — 评估指标体系 ✅
JudgeQuality / ModelImprovement / RubricQuality / Convergence 四个 dataclass + Spearman / accuracy@k。

**产出**: `trainable_openclaw/evaluation/metrics.py` (27 tests)

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

## Phase 4 — Agent 引擎集成 🟡

> **核心转变：从单轮纠错对话 → 多轮 Agent 交互。**
>
> 当前系统基于简单的 chat 格式 (`[{role, content}]`) 运作。Agent 场景完全不同：
> - 一轮交互包含多步：思考 → 工具调用 → 获取结果 → 回复用户
> - 评判维度扩展：工具选择合理性、步骤效率、错误恢复能力
> - veRL rollout 需要生成 Agent 轨迹，而非单次回答
>
> **策略：先用 nanobot 快速验证闭环，再迁移到 open-claw 做生产部署。**

### T1 — nanobot 调研与集成 ✅

**nanobot** 是一个轻量级 Agent 框架，适合快速集成和验证。

**已完成的工作：**

1. **nanobot 框架分析** ✅
   - Agent 循环结构：消息流 (bus.publish_inbound) → agent loop → consume_outbound
   - 工具注册与调用：filesystem, shell, sandbox 等内置工具
   - 记忆管理：SessionManager (JSONL) + SessionStore
   - CLI vs WebUI vs API 三种交互模式
2. **nanobot 实例搭建** ✅
   - nanobot 对接 veRL serve_ppo 推理后端 (provider=custom, apiBase=http://localhost:8000/v1)
   - 配置基础工具集，跑通完整 Agent 交互流程
   - `scripts/start_experience.sh`: 一键启动 serve_ppo + nanobot serve + nanobot gateway
3. **关键 Bug 修复** ✅
   - CLI `_wants_stream` 导致无回复 → patch 为 False
   - nanobot 启动需 PYTHONPATH (pip install -e 不可用)
   - Qwen3-4B thinking 不可关闭 → maxTokens=4096 + strip_think() 应用层处理
4. **适配层代码** ✅
   - `nanobot_adapter.py`: config 生成、serve_ppo 健康检查、连通性验证
   - `rollout.py`: Agent rollout 生成器（nanobot + DeepSeek）
   - `log_bridge.py`: nanobot JSONL ↔ ConversationStore SQLite 双写桥

**当前服务架构：**
```
serve_ppo :8000 (Qwen3-4B) ← nanobot serve :8900 ← nanobot GW :18790
                                                    ├─ health check
                                                    └─ WebSocket :18791
```

**产出**:
- `trainable_openclaw/agent/nanobot_adapter.py` — nanobot ↔ veRL 适配器
- `trainable_openclaw/agent/rollout.py` — Agent rollout 生成器
- `trainable_openclaw/agent/log_bridge.py` — 日志桥接
- `scripts/start_experience.sh` — 一键启动脚本
- `scripts/test_phase4.py` — 6 项集成测试 (6/6 通过)

**已解决的问题：**
- WebUI 回复正常: strip_think() 正则从 `r"^\s*<think>[\s\S]*$"` 改为 `r"^\s*<think>"` (只删除开标签，保留内容)
- kill_and_restart.py: 强制杀进程 + 清 __pycache__ + 重打补丁 + 重启全部服务
- Gateway 两端口架构确认: 18790 health check (TCP) + 18791 WebSocket + WebUI 静态文件

**已知限制：**
- Qwen3.5-0.8B 模型存在但 transformers 版本冲突暂不可用
- DB 1.9GB (仿真流水线数据), 需定期清理

**验证**:
- nanobot + veRL 推理后端 WebUI + CLI 交互正常
- 156 单元测试 0 失败 (154 已有 + Phase 4 集成测试)
- Gateway 空回复计数: 0 (strip_think patch 生效)

---

### T2 — Agent Tool-Use 训练闭环 🟡

> **参考**: MUA-RL (美团+中科院, arXiv 2508.18669, Apache 2.0) — 首个将模拟用户嵌入 GRPO 训练循环的工具使用框架。
> 调研文档: `docs/mua-rl_research.md`

**策略：不追求一步到位做多轮 vLLM rollout（移植成本太高），分两步走——先用外部管线生成冷启动 SFT 数据 + 单轮 GRPO，再逐步引入 loss masking 和稀疏奖励。**

#### T2.1 — MUA-RL Loss Masking 移植

MUA-RL 已证明 loss masking 能防止 reward hacking（背工具输出、抄袭用户语言、刷长度）。迁移到 serve_ppo：

- [ ] 在 serve_ppo rollout 阶段追踪 token 来源（agent 生成 / 工具返回 / 用户消息）
- [ ] 生成时维护 `loss_mask` 列表：agent token=1，工具 token=0，用户 token=0
- [ ] 批处理时拼接 `prompt_loss_mask` + `response_loss_mask`
- [ ] `ray_trainer.py` 添加多轮模式切换：`response_mask = loss_mask`（替代 attention_mask）
- [ ] `dp_actor.py` policy gradient loss 仅对 `loss_mask=1` 的 token 计算
- [ ] 改动量: ~200 行 Python，风险低

**产出**: serve_ppo.py 内 loss_mask 追踪 + 训练时切换

**验证**: 检查 loss_mask=0 的 token 在反向传播中贡献零梯度

#### T2.2 — 稀疏二元奖励

参考 MUA-RL 的奖励设计，在可验证任务上给 0/1 奖励：

- [ ] `reward_bridge.py` 新增 `compute_binary_reward()` 函数
- [ ] 可验证任务类型：代码执行（exit code=0）、文件状态（diff hash 匹配）、搜索准确性（URL/内容匹配）
- [ ] 不可验证任务（写作、头脑风暴）保持 rubric 连续评分
- [ ] 组合方式: `final = α * binary + (1-α) * rubric_mean`（默认 α=0.5）
- [ ] 改动量: ~100 行 Python

**产出**: `trainable_openclaw/training/reward_bridge.py` 内 binary reward 函数

**验证**: 代码执行正确→reward=1, 执行错误→reward=0, rubric 正常降级

#### T2.3 — Agent 场景设计与冷启动数据生成

参考 MUA-RL 的 9 场景设计方法，为 nanobot 的 17 个工具设计任务场景：

**5 个场景类别：**

| 场景 | 覆盖工具 | 示例任务 | 占比 |
|------|---------|---------|------|
| 文件管理 | read_file, write_file, edit_file, grep, find_files, list_dir, exec | "把 Downloads 里所有 PDF 移到 Documents/PDFs 并生成文件清单" | 30% |
| 信息检索 | web_search, web_fetch, grep, read_file | "查一下 PyTorch 2.6 的新特性，和当前安装版本做对比" | 25% |
| 代码任务 | exec, write_file, edit_file, apply_patch, run_cli_app | "写一个 Python 脚本分析 access.log 找出 404 最多的 URL" | 25% |
| 多步骤编排 | spawn, message, complete_goal, write_stdin | "同时搜索三个关键词，汇总结果后写报告" | 15% |
| 系统管理 | cron, long_task, exec, list_dir | "设置一个定时任务每天备份 config 文件" | 5% |

**冷启动数据生成流程：**
1. 每个场景生成 100-200 个具体任务（DeepSeek-v4-flash 生成，人工抽查 20%）
2. 用 DeepSeek-v4-flash 作为 expert agent + **真实 nanobot 工具** 执行每个任务
3. 记录完整轨迹：`[{role, content, tool_calls?, tool_call_id?, name?}]`
4. Rejection sampling: 只保留任务成功完成的轨迹（`complete_goal` 被调用且结果正确）
5. 质量过滤: 用已有 rubric judge 评分，丢弃 < 0.7 的轨迹
6. 目标: 500-1000 条高质量冷启动轨迹

**工具 mock/真实执行策略：**
- `exec`: 真实 shell 执行（sandbox 限制 `rm -rf /` 等危险操作）
- `read_file/write_file/edit_file`: 真实文件系统操作（临时目录）
- `web_search/web_fetch`: DeepSeek 模拟返回（避免真实网络调用耗时长）
- `spawn/message/cron`: mock 返回（预定义响应）

**产出**:
- `data/agent_scenarios/` — 5 个场景的任务定义 (JSON)
- `scripts/generate_agent_data.py` — 冷启动数据生成脚本
- `data/agent_trajectories/train.jsonl` — 500-1000 条训练轨迹
- `data/agent_trajectories/test.jsonl` — 50-100 条测试轨迹

**验证**:
- 每个场景生成 20 条 pilot 数据，确认工具调用格式正确
- 轨迹平均 3+ 步工具调用
- rejection sampling 通过率 > 60%

#### T2.4 — SFT 冷启动训练

用外部数据 + 自生成数据做 SFT warm-start：

- [ ] 数据源混合：自生成轨迹 (70%) + SWE-smith 代码轨迹 (15%) + ToolACE 函数调用 (15%)
- [ ] 格式统一: 转为 nanobot 消息格式 `[{role, content, tool_calls?, ...}]`
- [ ] SFT 训练: Qwen3-4B + LoRA rank=16, lr=2e-5, 3 epochs, batch_size=8
- [ ] 评估: 在测试集上验证模型生成的工具调用格式是否正确、参数是否合法

**产出**: `scripts/run_agent_sft.sh` — SFT 训练脚本

**验证**:
- SFT 后模型能生成语法正确的 tool_call JSON
- 工具选择准确率 > 60%（非代码场景）/ > 40%（代码场景）

#### T2.5 — Agent GRPO 训练

- [ ] 配置: 16p×4r=64, lr=1e-5, max_turns=15, max_model_len=12288, response_length=4096
- [ ] 奖励: 稀疏二元 (α=0.5) + rubric 连续 (1-α=0.5)，启用 loss masking
- [ ] 30 steps/round, 3-5 rounds
- [ ] 每步保存 generation_samples 用于分析

**产出**: `scripts/start_agent_train.sh` — Agent GRPO 训练脚本

**验证**:
- loss > 0（模型在学习）
- reward 趋势稳定或上升
- 无 "Empty response" 警告

#### T2.6 — 评测

- [ ] BFCL v4 多轮子集 (50 prompts) — 函数调用标准评测
- [ ] tau-bench 零售子集 (30 prompts) — 多轮工具使用评测
- [ ] 自有测试集 (50-100 prompts) — 回归检查
- [ ] 对比: baseline (SFT only) vs GRPO round 1/2/3
- [ ] 指标: 任务完成率 / 工具选择准确率 / 步骤效率 / 错误恢复率

**产出**: `scripts/eval_agent.py` — Agent 评测脚本

**T2 总验证 (门控):**
- [ ] loss masking 生效（梯度验证）
- [ ] 500+ 条冷启动轨迹生成
- [ ] SFT 后工具调用格式正确率 > 60%
- [ ] GRPO 训练任务完成率相比 SFT baseline 提升 > 5% 绝对值
- [ ] 无回归: 自有测试集纠错率下降 < 5%

---

### T3 — Agent 场景 Judge 扩展 ⬜

Agent 的评判维度与单轮对话完全不同。

#### T3.1 — Agent Rubric 生成

- [ ] 从 T2 生成的 Agent 轨迹中提取 agent 特有的错误模式
- [ ] B2 RubricGenerator 扩展: 支持 agent 场景的新 prompt 模板
- [ ] 6 个 agent 专用维度 → 3-5 条量化评分 rubric
- [ ] 每条 rubric: 详细扣分规则 + JSON 输出格式 + max_tokens 动态缩放

**产出**: `trainable_openclaw/evaluation/agent_rubric.py`

**验证**: 3 条 agent rubric 可正确评分，优质轨迹 > 劣质轨迹（区分度 p<0.05）

#### T3.2 — Agent Judge 执行

- [ ] B3 JudgeExecutor 扩展: 输入从单条 answer → 整条 trajectory（多步 thought + tool_call + tool_result）
- [ ] 轨迹评分 prompt 设计: 压缩工具调用上下文，保留关键步骤
- [ ] 合并评分: 多个 rubric 合并为单次 API 调用（省费用）
- [ ] Sync API: 兼容 Ray actor event loop

**产出**: `trainable_openclaw/evaluation/agent_judge.py`

**验证**: 10 条轨迹合并评分 < 5s, 无截断, 分数合理

#### T3.3 — 与 T2 训练闭环集成

- [ ] Agent Judge 替换 T2 训练中的 rubric 评分部分
- [ ] Agent Rubric 演进 (B4): 低分轨迹触发 rubric 更新
- [ ] 跑一轮完整训练 + 评测闭环

**T3 总验证 (门控):**
- [ ] Agent rubric 区分度验证通过
- [ ] Agent judge 合并评分正常（无截断、无异常分数）
- [ ] 训练闭环中 agent judge 稳定运行 (> 30 steps 无崩溃)

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

**产出**:
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
| 总 commits | 52+ |
| 单测通过 | 156 (7 个文件) |
| Phase 1-4 代码行数 | ~4,800 |
| 仿真训练对 | 557 (496 unique, 11 类别) |
| 动态 Rubric | 8 条 category-aware |
| Agent 服务 | 3 (serve_ppo + nanobot serve + nanobot GW) |
| MUA-RL 调研 | `docs/mua-rl_research.md` |

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

**Phase 1-3 完成** (18/21 steps)，156 单测通过，框架代码可运行。4 轮训练实验完成。

**Phase 4 T1 完成，T2 详细计划已出：**
- nanobot + serve_ppo (Qwen3-4B) WebUI + CLI 交互跑通，strip_think patch 生效
- serve_ppo :8000 / nanobot serve :8900 / nanobot GW :18790 (+ WebSocket :18791) 三服务架构
- Qwen3-4B thinking 不可关闭 → maxTokens=4096 + strip_think() 应用层方案
- **MUA-RL 调研完成** (`docs/mua-rl_research.md`) — loss masking + 稀疏二元奖励为 P0 优先迁移

**当前焦点 — T2 Agent Tool-Use 训练闭环：**

```
T2.1 loss masking (~200行) → T2.2 稀疏奖励 (~100行) → T2.3 场景+数据生成 (500-1000条)
  → T2.4 SFT 冷启动 → T2.5 GRPO 训练 → T2.6 评测 (BFCL v4 + tau-bench)
```

**为什么是这个顺序？**
1. loss masking 和稀疏奖励是小改动、高收益的基础设施
2. 冷启动数据生成需要调用 DeepSeek API（~$5-10），数据质量是后续训练的天花板
3. SFT 先验证格式正确性，GRPO 再优化质量——分步验证，降低调试复杂度
4. vLLM 多轮 rollout 移植成本太高，先用外部管线生成数据 + 单轮 GRPO 验证闭环

**T3 Agent Judge 扩展** — T2 跑通后再启动，避免提前优化

**训练收敛问题** — 长期方向：Qwen3.5 模型、7B+ 模型、更多步数、pairwise reward。当前阶段优先把 agent 训练闭环跑通，Qwen3-4B 容量问题不是当前瓶颈。
