# MUA-RL 调研报告

> 调研日期：2026-06-12
> 仓库: [github.com/zzwkk/MUA-RL](https://github.com/zzwkk/MUA-RL)
> 论文: Multi-turn User-interacting Agentic Tool Use RL (arXiv 2508.18669, 2025-08)
> 许可: Apache 2.0

## 一、项目概述

MUA-RL（Multi-turn User-interacting Agentic Tool Use RL）由美团 + 中科院自动化所 + 北京大学联合发布。是首个将 LLM 模拟用户**直接嵌入 GRPO 训练循环**的工具使用 agent 框架。

核心主张：工具使用训练中的用户交互不是噪声，而是关键信号源——模拟用户应参与每一轮 rollout，而非仅在数据生成阶段使用。

## 二、与 τ-bench 的关系

MUA-RL 基于 τ-bench（TAU-Bench, Sierra Research/Princeton, 2024）改造：

| | τ-bench (原始) | MUA-RL |
|---|---|---|
| 用途 | 评测基准 | 训练框架 |
| 工具 | 28 个领域 API | 16+16 个零售/航空工具 |
| 用户 | 固定脚本 | GPT-4o 动态模拟 |
| 环境 | 隐藏 JSON 数据库 | 开放 Python 环境 |
| 数据集 | 165 个任务（评测） | ~2000 条轨迹（训练） |

MUA-RL 复用了 τ-bench 的零售和航空场景设计，但将环境从封闭评测改为开放训练环境，并增加了动态用户模拟器。

## 三、核心技术

### 3.1 Loss Masking（损失掩码）

GRPO 计算 loss 时，按 token 来源分别处理：

| Token 来源 | loss_mask | 含义 |
|-----------|-----------|------|
| Agent 生成的文本/工具调用 | 1 | 正常计算梯度 |
| 工具执行返回结果 | 0 | 不计算梯度 |
| 用户模拟器消息 | 0 | 不计算梯度 |

**效果**：模型只从自己的决策和对话中学习，无法通过背工具输出模式、抄袭用户措辞、或刷对话长度来获取梯度优势。

实现路径（5 层调用链）：
1. `sglang_rollout.py` — rollout 阶段维护 `loss_mask` 列表
2. `sglang_rollout.py` — 批处理时拼接 `prompt_loss_mask` + `response_loss_mask`
3. `ray_trainer.py` — 多轮模式下 `response_mask = loss_mask`（而非 `attention_mask`）
4. `dp_actor.py` — Policy gradient loss 仅对 `loss_mask=1` 的 token 计算
5. 结果：工具输出和用户消息对模型权重更新零贡献

### 3.2 稀疏二元奖励（Sparse Binary Reward）

```python
reward = 1.0 if (output_check and database_hash_match) else 0.0
```

- **输出完整性检查**: 所有预期输出字符串必须在 agent 给用户的回复中出现
- **数据库状态检查**: 最终数据库 SHA-256 哈希必须与 ground truth 精确匹配

没有部分分、没有格式奖励、没有中间步骤奖励。论文证明这能防止 reward hacking，且自然导致无用工具（如 Think、Calculate）在训练中被自动剪枝。

### 3.3 模拟用户在 RL 循环内

- **模型**: GPT-4o-2024-11-20（可配置）
- **触发**: 当 policy model 输出 text（非 tool call）时，调用 GPT-4o 生成下一轮用户消息
- **角色设定**: 嵌入任务初始 prompt，不在训练代码中
- **动态行为**: 可改变需求、添加约束、表达困惑——不是固定脚本
- **代价**: 每轮训练触发 GPT-4o API 调用，是大规模训练的主要成本

### 3.4 三阶段训练流程

1. **冷启动 SFT**: ~2000 条专家轨迹，9 个场景（5 合成 + 4 真实 MCP）
2. **GRPO 训练**: 模拟用户全程参与 rollout，稀疏 binary reward，loss masking
3. **多场景泛化**: 单一模型在 9 个不同工具宇宙中训练，学习适应任意工具集

## 四、训练配置

### 全量配置（4 节点 × 8 GPU）

| 参数 | 值 |
|------|-----|
| Batch size | 32 |
| Mini-batch | 32 |
| Rollouts/prompt (G) | 8 |
| Learning rate | 1e-6 |
| KL coefficient | 0.001 (low_var_kl) |
| Max turns | 30 |
| Max model length | 32768 |
| Response length/turn | 1024 |
| Temperature | 1.0 |
| Epochs | 30 |
| Rollout engine | SGLang |
| GPU memory utilization | 0.8 |
| Tensor parallel | 2 |
| Sequence parallel | 4 (Ulysses) |

### 测试配置（单节点 8 GPU）

- Batch: 8, Rollouts: 16, 其他同上

## 五、开源代码结构

MUA-RL 是 veRL 的 fork，不是独立代码库。核心改动集中在：

```
MUA_environments/          # 环境抽象层（新）
├── base/                  # BaseEnvironment, ToolRegistry, DataLoader
└── taubench/              # 零售 + 航空场景
    ├── retail/            # 16 个零售工具
    └── airline/           # 航空工具

verl/tools/taubench_*/     # 工具实现（新）
verl/utils/reward_score/   # 稀疏二元奖励（新）
examples/sglang_multiturn/ # GRPO 配置 + 训练脚本（新）
verl/trainer/ppo/          # loss_mask 支持（修改）
verl/workers/actor/        # loss_mask 梯度计算（修改）
verl/workers/rollout/      # 多轮 + 用户模拟 + loss_mask（修改）
data/                      # Parquet 测试集（新）
```

HuggingFace 数据集: `zzwkk/MUA-RL-Dataset` (Apache 2.0)

## 六、与 trainable-openclaw 对比

| 维度 | MUA-RL | trainable-openclaw |
|------|--------|-------------------|
| 训练框架 | veRL + SGLang | veRL + vLLM |
| 算法 | GRPO + ref-policy KL | GRPO + group-norm advantage |
| 模型 | Qwen3-Non-Thinking 8B+ | Qwen3-4B（强制 think） |
| GPU 规模 | 32×H100 | 1×RTX 4090 (24GB) |
| 工具 | 领域专用 API（每场景重定义） | 17 个通用固定工具（nanobot） |
| 任务域 | 客服（零售/航空/电信） | 32 LMSYS 类别（通用助手） |
| 用户模拟 | GPT-4o 在 RL 循环内 | DeepSeek 在训练循环外 |
| 奖励 | 稀疏二元 (0/1) | 连续 rubric (0-1) |
| Loss Masking | 有 | 无 |
| 自进化系统 | 无 | Rubric 生成/演化/裁剪 |
| 服务化部署 | 无 | serve_ppo + nanobot + WebUI |
| 对话日志 | TensorBoard | SQLite + JSONL 双写 |
| 纠错引擎 | 无 | 多轮纠错对话 + 维度提取 |
| 许可 | Apache 2.0 | 内部项目 |

## 七、互补性分析

### MUA-RL 有、本项目没有的
- **Loss masking**: 已证明能防止 reward hacking，防止模型背工具输出、刷长度
- **稀疏二元奖励**: 简单有效，配合确定性验证（代码执行、文件哈希）直接用
- **用户模拟在 RL 循环内**: 提供更真实的交互信号

### 本项目有、MUA-RL 没有的
- **通用 17 工具 agent**: 不依赖场景重定义，固定工具集
- **Rubric 自进化**: 评分维度自动生成、演化、归档——完整闭环
- **真实用户服务**: nanobot + WebUI 在线服务，积累真实交互数据
- **纠错对话引擎**: 多轮纠错流水线，可提取（错误, 反馈, 修正）三元组
- **纯消费级 GPU**: 单卡 RTX 4090 可跑完整训练

### 不构成"撞车"
两者方向互补而非重复。MUA-RL 证明了"用户模拟 + GRPO + 工具"范式有效，本项目在此基础之上：
1. 从领域专用工具扩展到通用固定工具集
2. 从纯训练扩展到训练+服务一体化
3. 从固定奖励扩展到自进化评判系统

## 八、可迁移技术（优先级排序）

### P0 — 已具备迁移条件

**Loss Masking** (~200 行)
- 改动文件: serve_ppo 中的 rollout 追踪 + ray_trainer 中的 response_mask 切换
- 风险: 低。纯逻辑改动，不涉及模型架构
- 收益: 高。所有工具使用训练场景都受益

**稀疏二元奖励** (~100 行)
- 改动文件: reward_bridge.py，新增可验证任务的 ground truth 比对
- 适用场景: 代码执行（exit code）、文件状态（diff hash）、搜索准确性（URL/内容匹配）
- 组合方式: `final = 0.5 * binary + 0.5 * rubric_mean`

### P1 — 需要适配

**BaseTool 生命周期** (~300 行)
- 参考 MUA-RL 的 create → execute → calc_reward → release 模式
- 为 nanobot 的 17 个工具增加 per-instance 状态管理

**场景化训练数据设计**（设计工作）
- 参考 MUA-RL 的 9 场景设计方法
- 为 nanobot 设计 5-10 个日常任务场景，每个覆盖 4-8 个工具

### P2 — 暂不推荐

**多轮 rollout 进 vLLM** (~1000+ 行)
- MUA-RL 的多轮机制深度耦合 SGLang
- vLLM 多轮支持不成熟，移植风险高
- 短期替代: 外部 User Sim 管线（已有）

**用户模拟进 RL 循环**
- 需 GPT-4o API 每步调用，成本高
- 与当前 DeepSeek 离线模拟方案差异大
- 待 loss masking + 稀疏奖励验证后再考虑

## 九、风险与限制

1. **GPU 规模差 32 倍**: MUA-RL 的超参数在 32×H100 上调优，单卡 LoRA 需重新探索 lr、kl_coef、batch size
2. **Qwen3-4B 强制 thinking**: `<think>` 消耗约 30% 输出预算，工具调用的有效 response_length 更少
3. **稀疏奖励在通用任务上可能过稀疏**: 客服有数据库哈希可验证，"写一篇文章"难以定义二元完成标准
4. **DeepSeek vs GPT-4o 模拟质量差**: 论文消融实验显示模拟器质量直接影响训练效果
5. **SGLang vs vLLM**: 最核心的多轮 rollout 代码是 SGLang 专属的，vLLM 移植有不确定性

## 十、后续行动计划

1. [ ] 下载 `zzwkk/MUA-RL-Dataset`，分析冷启动数据格式
2. [ ] 在 serve_ppo 中实现 loss masking（P0）
3. [ ] 给 nanobot 设计 5-10 个日常任务场景（参考 MUA-RL 的 9 场景方法）
4. [ ] 用 DeepSeek + nanobot 工具生成 500-1000 条冷启动轨迹
5. [ ] 实现稀疏二元奖励作为辅助信号（P0）
6. [ ] 跑第一轮 MUA-RL 风格的 GRPO 训练
7. [ ] 评估是否需要多轮 rollout（根据前 6 步结果决定）
