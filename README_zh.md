# Trainable OpenClaw

**生产级自进化AI助手引擎。**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Early%20Development-orange)]()

Trainable OpenClaw 让任何大语言模型成为一个**在使用中持续自我进化的智能体**。基于 [veRL](https://github.com/volcengine/verl) 作为推理引擎，持续收集真实用户的对话反馈，通过 LLM 自主生成的 Rubrics 自动评估回复质量，在服务空闲时进行 LoRA 微调——全部在单一部署中完成。

> [English](README.md) | **中文**

---

## 解决什么问题

通用大模型不会从使用中自我改进。每次犯错——生成有 bug 的代码、误解用户意图、重复同样的错误——都是被浪费的学习机会。现有的 RLHF 流程是离线的、批量的、和真实用户脱节的。

## 我们的方案

**在运行时闭环学习。** Trainable OpenClaw 把每一次用户交互变成训练信号：

```
用户 → Agent → veRL引擎(推理) → 回复 → 用户
                                        ↓
                                  用户反馈收集
                                        ↓
                              LLM分析反馈模式
                                        ↓
                           LLM自主生成严格Rubrics
                                        ↓
                          Rubrics对输出打分 → 奖励信号
                                        ↓
                               空闲检测触发训练
                                        ↓
                          LoRA微调(veRL引擎)
                                        ↓
                          权重同步 → 切回推理服务
```

---

## 架构

```
┌─────────────────────────────────────────────────────┐
│                 Trainable OpenClaw                    │
├───────────────┬──────────────┬──────────────────────┤
│   API服务      │  评估流水线   │  训练编排器            │
│  (FastAPI)    │              │                       │
│               │              │                       │
│  /v1/chat     │  反馈→       │  空闲检测 →            │
│  /v1/health   │  Rubric→    │  训练调度 →            │
│  /v1/stats    │  Judge→     │  LoRA →               │
│               │  Reward     │  权重同步              │
├───────────────┴──────────────┴──────────────────────┤
│              veRL 引擎 (vLLM/SGLang)                 │
│         推理模式 ◄────────────► 训练模式              │
└─────────────────────────────────────────────────────┘
```

---

## 核心特性

- **自进化闭环** — 从每次对话中自动学习，无需人工标注
- **LLM自主生成Rubrics** — 评分标准从用户反馈模式中自动归纳，不是预设死规则
- **双模引擎** — 同一组GPU既做推理又做训练，空闲时自动切换（资源友好，单卡或AutoDL可跑）
- **GRPO原生支持** — 多答案生成 + group-based advantage 计算，天然适配
- **生产级设计** — 面向真实部署而非研究原型：结构化日志、健康检查、监控面板
- **OpenAI兼容API** — 任何使用 chat completions 格式的应用可直接接入

---

## 工作流程

### 1. 服务与收集
veRL 引擎运行在**推理模式**，通过 OpenAI 兼容 API 处理对话请求。每次对话按 session 和用户维度记录。

### 2. 分析反馈
LLM 阅读对话历史，识别**反馈模式**——用户持续抱怨什么、赞赏什么。

### 3. 生成 Rubrics
基于识别到的模式，LLM 自主创建**严格的、可量化的评分任务**。每条 Rubric 是一个精确的 prompt，包含明确的扣分规则（如："变量命名不符合 snake_case，每处扣2分"）。

### 4. 打分与排名
GRPO 训练时，系统对每个 prompt 生成 N 个候选回答。所有候选回答用生成的 Rubrics 分别打分，排名靠前的作为正样本。

### 5. 空闲时训练
当一段时间没有请求到达时，系统：
- 将推理引擎休眠（释放 GPU 显存）
- 用带评分的对话数据执行 LoRA 微调
- 将更新后的权重同步回推理引擎
- 以改进后的模型恢复推理服务

---

## 为什么做这个项目？（与 agent-lightning 的差异化）

微软的 [agent-lightning](https://github.com/microsoft/agent-lightning)（17K+ stars）是一个优秀的 Agent 训练框架。它用 LightningStore 中心枢纽支持任意 Agent 框架（LangChain、AutoGen、CrewAI、OpenAI SDK）和多种算法（RL、APO、SFT）。如果你已有 Agent 想训练 — 用它。

Trainable OpenClaw 做了一个**根本性的架构选择差异**：

| 维度 | agent-lightning | Trainable OpenClaw |
|------|----------------|---------------------|
| **核心范式** | 训练框架包裹 Agent | 自进化推理引擎 |
| **推理引擎** | 外部（通过 LiteLLM 代理） | 内部（深度改造 veRL） |
| **训练触发** | 算法驱动（显式循环） | 空闲驱动（后台自动） |
| **引擎集成深度** | 浅层 — 向 vLLM 发 HTTP 请求 | 深层 — 控制 sleep/wake/weight-sync |
| **代码量** | ~67 核心文件，多存储多算法 | ~6 核心模块，单一聚焦链路 |
| **目标用户** | 训练 Agent 的研究者 | 运营进化中模型的服务提供者 |
| **Agent 耦合** | Agent 代码在循环内运行 | Agent 是 API 的用户（解耦） |

**核心洞察：** agent-lightning 把推理引擎当作黑盒服务 — 发 prompt、收 response。Trainable OpenClaw 把推理引擎当作**产品本身**。我们深入改造 veRL 的 hybrid engine：

1. rollout replicas **保持唤醒**，直接服务用户请求（不走 proxy）
2. 训练由**真实空闲检测**触发（无请求 → sleep replicas → 训练 → wake）
3. 权重同步在**同一组 GPU worker** 上原地完成
4. 用户只需调用**一个 HTTP 端点**，无需写 Agent 代码

这让 Trainable OpenClaw 更像一个**越用越聪明的个性化推理服务**，而非一个你去提交 Agent 来训练的平台。

### 什么时候用哪个？

| 你的需求 | 用哪个 |
|---------|--------|
| "我想训练我的 Agent" | agent-lightning |
| "我想要一个越用越聪明的推理服务" | Trainable OpenClaw |

---

## 快速开始

> **环境要求：** Python 3.10+, CUDA 12.4+, 1-8 GPUs（小模型单卡即可）

```bash
# 克隆仓库
git clone https://github.com/your-org/trainable-openclaw.git
cd trainable-openclaw

# 安装依赖
pip install -e .

# 启动自进化推理服务
python -m trainable_openclaw.server.app --config configs/serve.yaml
```

服务启动后处于推理模式。发送对话请求：

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "写一个合并两个有序列表的Python函数"}]
  }'
```

空闲5分钟后自动触发训练。访问 `http://localhost:8000/dashboard` 查看监控面板。

---

## 项目结构

```
trainable-openclaw/
├── trainable_openclaw/        # 核心Python包
│   ├── server/                # FastAPI推理服务
│   ├── engine/                # veRL引擎封装
│   ├── agents/                # Agent连接器（openclaw, nanobot等）
│   ├── logging/               # 对话存储与压缩
│   ├── evaluation/            # Rubric生成与LLM打分
│   ├── training/              # 空闲检测与训练编排
│   └── dashboard/             # 监控面板
├── configs/                   # 配置文件
├── docs/                      # 文档与设计
├── papers/                    # 参考论文
├── tests/                     # 测试
├── scripts/                   # 工具脚本
├── verl-main-0516/            # veRL参考实现
└── requirements.txt
```

---

## 路线图

详细研发计划见 [docs/roadmap.md](docs/roadmap.md)。

**当前阶段（2026年5-6月）：** MVP —— 核心自进化闭环端到端跑通。

| 阶段 | 状态 |
|------|------|
| Phase 0 — 论文调研与算法确定 | 进行中 |
| Phase 1 — veRL双模引擎改造 | ✅ A1, A2 完成 |
| Phase 2 — 自进化评判系统 | 待开始 |
| Phase 3 — 集成与Dashboard | 待开始 |
| Phase 4 — 测试集与效果评估 | 待开始 |

---

## 参与贡献

Trainable OpenClaw 处于早期开发阶段，欢迎贡献！以下方向尤其需要帮助：

- **算法研究** — 改进 GRPO reward 设计、Rubric 生成质量
- **引擎后端** — 接入更多推理引擎（TensorRT-LLM, llama.cpp 等）
- **Agent连接器** — 对接更多 chatbot 框架
- **评估** — Rubric 质量与人工判断的对齐评测
- **文档** — 教程、部署指南、最佳实践

贡献指南（CONTRIBUTING.md）即将上线。

---

## 开源协议

[Apache 2.0](LICENSE) © 2026

---

## 致谢

本项目站在巨人的肩膀上，特别致谢以下开源项目：

- **[veRL](https://github.com/volcengine/verl)** — 字节跳动火山引擎的大模型强化学习框架，提供核心训练与推理基础设施
- **[vLLM](https://github.com/vllm-project/vllm)** — 高吞吐量LLM推理引擎，支撑推理后端
- **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** — NVIDIA大规模Transformer训练框架，实现高效分布式训练
- **[SGLang](https://github.com/sgl-project/sglang)** — 结构化生成语言，作为备选推理后端
- **[DeepSeek](https://github.com/deepseek-ai)** — 提供 DeepSeek-Flash 及开源模型生态
- **[PyTorch](https://github.com/pytorch/pytorch)** — 深度学习基础框架
- **[FastAPI](https://github.com/tiangolo/fastapi)** — 现代API框架，支撑服务层
- **[Ray](https://github.com/ray-project/ray)** — 分布式计算框架，实现多GPU扩展
