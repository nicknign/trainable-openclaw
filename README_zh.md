# Trainable OpenClaw

**自进化 LLM 推理引擎 — 从用户交互中学习，在空闲时自我改进。**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-MVP-yellow)]()

Trainable OpenClaw 将 [veRL](https://github.com/volcengine/verl) 封装为可自我改进的推理服务。收集对话反馈，通过 LLM 自主生成的 Rubric 评估回复质量，在服务空闲时自动微调模型——全部在单一部署中完成。

> [English](README.md) | **中文**

---

## 解决什么问题

通用大模型不会从使用中进步。每次犯错都是被浪费的学习机会。Trainable OpenClaw 把这个环闭合：

```
用户请求 → veRL引擎(推理) → 回复 → 用户
                                    ↓
                              反馈收集与分析
                                    ↓
                          LLM自动归纳反馈模式
                                    ↓
                       LLM自主生成评分Rubric
                                    ↓
                     多答案打分 → 奖励信号
                                    ↓
                         空闲检测触发训练
                                    ↓
                      LoRA微调 (GRPO)
                                    ↓
                    权重同步 → 恢复推理
```

---

## 架构

```
┌──────────────────────────────────────────────────────────┐
│                   Trainable OpenClaw                      │
├────────────┬──────────────┬───────────────┬──────────────┤
│  API服务    │  仿真流水线   │  评估流水线    │  训练编排     │
│  (FastAPI) │              │               │              │
│            │  用户模拟 →   │  反馈分析 →    │  空闲检测 →   │
│  /v1/chat  │  纠错对话 →   │  Rubric生成 → │  GRPO训练 →  │
│  /v1/health│  轨迹评估    │  Judge打分 →  │  权重同步     │
│            │              │  Rubric演进   │              │
├────────────┴──────────────┴───────────────┴──────────────┤
│                veRL 引擎 (vLLM + FSDP)                    │
│            推理模式 ◄──────────► 训练模式                  │
└──────────────────────────────────────────────────────────┘
```

---

## 当前状态（MVP — 2026年6月）

**框架代码全部完成并可运行。** 所有模块已开发、测试（154个单测全部通过）、并在远程 GPU 机器上用真实 API 端到端验证通过。

### 已完成的模块

| 模块 | 说明 | 测试 |
|------|------|------|
| API 服务 | OpenAI 兼容接口，WAL 日志记录 | 23 |
| 训练编排器 | 空闲检测 → 自动触发 GRPO 训练，训练中返回 503 | 24 |
| 仿真管线 | 5 种用户画像，多轮纠错对话，557 条训练对 | — |
| 反馈分析 | LLM 从对话日志中识别错误模式 | — |
| Rubric 生成 | LLM 从反馈中自动生成可量化评分标准 | — |
| Judge 执行器 | 多 Rubric 合并评分，Ray actor 兼容 sync API | 真实API ✅ |
| Rubric 演进 | 低分样本自动触发 Rubric 更新，过期归档 | 25 + 端到端 ✅ |
| 评估指标 | Spearman / accuracy@k / 覆盖率 / 收敛检测 | 27 |
| Pipeline | 训前评测 → 训练 → 训后评测 → Rubric 演进 CLI | 20 + GPU端到端 ✅ |
| Dashboard | Streamlit 面板：模式/进度/Rubric/对话 | 6 |

### 尚未解决的问题

- **训练不收敛** — Qwen3-4B + GRPO，42 步训练 reward 震荡无上升趋势。checkpoint 评测确认模型退化（纠错率 0.60 → 0.88，越高越差）。大概率是 4B 容量不足，需要换 7B+ 模型。
- **Checkpoint 跨重启丢失** — Phase 2/3 的检查点未能保存，仅 Phase 1 的 step_10 存留。
- **S4 反思模块** — 分析 FAIL 轨迹根因、改进 User Sim 的策略尚未构建。

---

## 快速开始

> **环境要求：** Python 3.10+, CUDA 12.4+, 1-8 GPUs（单张 48GB 显卡可跑 Qwen3-4B）

```bash
git clone https://github.com/nicknign/trainable-openclaw.git
cd trainable-openclaw
pip install -e .
```

### 1. 启动推理服务

```bash
bash scripts/start_train.sh
```

以 HYBRID 模式启动 veRL（vLLM + FSDP），LoRA rank=16。

### 2. 发送对话请求

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "写一个合并两个有序列表的Python函数"}]
  }'
```

### 3. 运行评测流水线

```bash
# 训前基线评测
python -m trainable_openclaw.pipeline --eval-only --max-test-prompts 10

# 生成训练配置
python -m trainable_openclaw.pipeline --gen-config

# 完整报告
python -m trainable_openclaw.pipeline --output results.json
```

### 4. 启动 Dashboard

```bash
streamlit run scripts/dashboard.py
```

### 5. 运行测试

```bash
python -m pytest tests/ -v --ignore=tests/test_a1_integration.py --ignore=tests/test_a2_integration.py --ignore=tests/test_a3_integration.py
# 154 通过
```

---

## 项目结构

```
trainable-openclaw/
├── trainable_openclaw/          # 核心包
│   ├── server/api.py            # FastAPI 推理服务 (OpenAI 兼容)
│   ├── server/__init__.py
│   ├── logging/                 # 对话日志 (SQLite + WAL) + CLI 查看器
│   ├── training/                # 空闲检测与训练编排
│   ├── evaluation/              # 自进化评判系统
│   │   ├── feedback.py          # B1: 反馈模式分析
│   │   ├── rubric.py            # B2: Rubric 生成 + 持久化
│   │   ├── rubric_engine.py     # B2: LLM 驱动 Rubric 流水线
│   │   ├── rubric_evolver.py    # B4: 低分日志驱动自动演进
│   │   ├── judge.py             # B3: 多 Rubric 合并评分执行器
│   │   ├── trajectory_eval.py   # S3: 轨迹分级 + 数据导出
│   │   ├── correction_rate.py   # D2: 纠错率评估器
│   │   └── metrics.py           # S5: 评估指标体系
│   └── pipeline.py              # C1: 主流水线编排器
├── scripts/                     # 启动与工具脚本
│   ├── start_train.sh           # serve_ppo 启动脚本
│   ├── run_simulation.py        # User Sim 纠错对话管线
│   ├── validate_modules.py      # 真实 API 验证 (judge + evolver + pipeline)
│   ├── run_post_eval.py         # Checkpoint 纠错率评测
│   ├── extract_lora.py          # FSDP checkpoint → LoRA 权重提取
│   ├── convert_lora.py          # FSDP LoRA → PEFT/vLLM adapter 格式转换
│   ├── analyze_run.py           # 训练日志解析 + 分析报告
│   ├── dashboard.py             # Streamlit 监控面板
│   └── chat.py                  # 交互式聊天 CLI
├── docs/                        # 文档
│   ├── roadmap.md               # 研发路线图
│   ├── code_guide.md            # 代码说明
│   └── validation_report.md     # 框架评测报告
├── tests/                       # 154 个单测 (6 个文件)
├── data/                        # 数据集 / Rubric / 对话数据库
├── runs/                        # 训练日志存档
├── verl-main-0516/              # 修改后的 veRL (serve_ppo + checkpoint + weight sync)
└── requirements.txt
```

---

## 与 agent-lightning 的差异

微软的 [agent-lightning](https://github.com/microsoft/agent-lightning) 是优秀的 Agent 训练框架。Trainable OpenClaw 做了一个不同的架构选择：

| | agent-lightning | Trainable OpenClaw |
|---|---|---|
| **范式** | 训练框架包裹 Agent | 自进化推理引擎 |
| **推理引擎** | 外部（LiteLLM 代理） | 内部（深度改造 veRL） |
| **训练触发** | 算法驱动循环 | 空闲驱动（后台自动） |
| **使用方式** | 写 Agent 代码接入 | 调 HTTP 接口即可 |

**用 agent-lightning** — 如果你已有 Agent，想训练它。
**用 Trainable OpenClaw** — 如果你想要一个越用越聪明的推理服务。

---

## 路线图

详细计划见 [docs/roadmap.md](docs/roadmap.md)。

| 阶段 | 状态 |
|------|------|
| Phase 0 — 论文调研与算法确定 | 背景持续 |
| Phase 1 — veRL 双模引擎改造 | ✅ A1+A2+A3 完成 |
| Phase 1.5 — 数据工程与仿真 | ✅ S1+S2+S3+S5 完成，S4 待做 |
| Phase 2 — 自进化评判系统 | ✅ B0+B1+B2+B3+B4 完成 |
| Phase 3 — 集成与 Dashboard | ✅ C1+C2 完成 |
| Phase 4 — 效果评估 | 🟡 D1+D2 完成，D3 远期 |

---

## 开源协议

[Apache 2.0](LICENSE) © 2026

---

## 致谢

- **[veRL](https://github.com/volcengine/verl)** — 核心训练与推理基础设施
- **[vLLM](https://github.com/vllm-project/vllm)** — 高吞吐量 LLM 推理
- **[DeepSeek](https://github.com/deepseek-ai)** — Judge 模型 (deepseek-v4-flash)
- **[FastAPI](https://github.com/tiangolo/fastapi)** — API 服务层
- **[Ray](https://github.com/ray-project/ray)** — 分布式计算
- **[PyTorch](https://github.com/pytorch/pytorch)** — 深度学习框架
