# Trainable OpenClaw

**A production-grade self-evolving AI assistant engine.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Early%20Development-orange)]()

Trainable OpenClaw turns any LLM into a **self-improving agent** that learns from real user interactions. It wraps [veRL](https://github.com/volcengine/verl) as the inference engine, continuously collects conversation feedback, evaluates quality via LLM-generated rubrics, and fine-tunes the model during idle time — all in a single deployment.

> **English** | [中文](README_zh.md)

---

## The Problem

General-purpose LLMs don't improve from usage. Every mistake they make — generating buggy code, misunderstanding context, repeating the same errors — is a lost learning opportunity. Existing RLHF pipelines are batch, offline, and disconnected from real users.

## Our Approach

**Close the loop at runtime.** Trainable OpenClaw turns every user interaction into a training signal:

```
User → Agent → veRL Engine (Inference) → Response → User
                                                    ↓
                                            User Feedback
                                                    ↓
                                     LLM Analyzes Feedback Patterns
                                                    ↓
                                   LLM Generates Strict Scoring Rubrics
                                                    ↓
                              Rubrics Score Model Outputs → Reward Signal
                                                    ↓
                                           Idle Detection Triggers
                                                    ↓
                                LoRA Fine-tuning on veRL Engine
                                                    ↓
                                    Weights Synced → Back to Serving
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Trainable OpenClaw                    │
├───────────────┬──────────────┬──────────────────────┤
│  API Server   │  Evaluation  │  Training Orchestrator│
│  (FastAPI)    │  Pipeline    │                       │
│               │              │                       │
│  /v1/chat     │  Feedback→   │  IdleDetector →       │
│  /v1/health   │  Rubric→     │  TrainScheduler →     │
│  /v1/stats    │  Judge→      │  LoRA →               │
│               │  Reward      │  WeightSync           │
├───────────────┴──────────────┴──────────────────────┤
│              veRL Engine (vLLM/SGLang)               │
│         Serving Mode ◄────────► Training Mode        │
└─────────────────────────────────────────────────────┘
```

---

## Key Features

- **Self-Evolving Loop** — Learns from every conversation automatically, no manual annotation needed
- **LLM-Generated Rubrics** — Scoring criteria emerge from user feedback patterns, not hardcoded rules
- **Dual-Mode Engine** — Same GPUs serve inference AND train, switching during idle periods (resource-friendly, runs on a single GPU or AutoDL)
- **GRPO-Ready** — Native support for multi-answer generation and group-based advantage computation
- **Production Grade** — Designed for real deployment, not research prototyping: structured logging, health checks, monitoring dashboard
- **OpenAI-Compatible API** — Drop-in replacement for any app using the chat completions format

---

## How It Works

### 1. Serve & Collect
The veRL engine runs in **serving mode**, handling chat requests via an OpenAI-compatible API. Every conversation is logged with session and user metadata.

### 2. Analyze Feedback
An LLM reads through conversation histories and identifies **feedback patterns** — what users consistently complain about or praise.

### 3. Generate Rubrics
Based on identified patterns, the LLM autonomously creates **strict, quantifiable scoring rubrics**. Each rubric is a precise prompt with explicit deduction rules (e.g., "Deduct 2 points for each PEP8 naming violation").

### 4. Score & Rank
For GRPO training, the system generates N candidate responses per prompt. All responses are scored against the generated rubrics. The best response wins.

### 5. Train During Idle Time
When no requests arrive for a configurable timeout, the system:
- Puts inference engines to sleep (frees GPU memory)
- Runs LoRA fine-tuning using the scored conversation data
- Syncs updated weights back to the inference engines
- Resumes serving with the improved model

---

## Why a New Project? (vs. agent-lightning)

Microsoft's [agent-lightning](https://github.com/microsoft/agent-lightning) (17K+ stars) is an excellent framework for training AI agents with RL. It wraps ANY agent framework (LangChain, AutoGen, CrewAI, OpenAI SDK) and supports multiple algorithms (RL, APO, SFT) via a central "LightningStore." If you have an existing agent and want to train it — use agent-lightning.

Trainable OpenClaw takes a **fundamentally different architectural bet**:

| Dimension | agent-lightning | Trainable OpenClaw |
|-----------|----------------|---------------------|
| **Core paradigm** | Training framework that wraps agents | Self-evolving inference engine |
| **Inference engine** | External (proxied via LiteLLM) | Internal (deeply modified veRL) |
| **Training trigger** | Algorithm-driven (explicit loop) | Idle-driven (background automatic) |
| **Engine integration** | Shallow — sends HTTP requests to vLLM | Deep — controls sleep/wake/weight-sync at engine level |
| **Code footprint** | ~67 core files, multi-store, multi-algo | ~6 core modules, single focused pipeline |
| **Target user** | Researcher training an agent | Service operator running an evolving model |
| **Agent coupling** | Agent code runs in the loop | Agent is the user of the API (decoupled) |

**The key insight:** agent-lightning treats the inference engine as a black-box service — it sends prompts and reads responses. Trainable OpenClaw treats the inference engine as the **product itself**. We modify veRL's hybrid engine so that:

1. The rollout replicas **stay awake** and serve user requests directly (not through a proxy)
2. Training is triggered by **real idle detection** (no requests → sleep replicas → train → wake)
3. Weight synchronization happens **in-place** on the same GPU workers
4. Users interact with **a single HTTP endpoint** — they don't need to write agent code

This makes Trainable OpenClaw more like a **personalized inference service** that quietly improves from usage, rather than a training framework you bring your agent to.

### When to use which?

| Your need | Use |
|-----------|-----|
| "I want to train my agent" | agent-lightning |
| "I want an inference service that gets smarter over time" | Trainable OpenClaw |

---

## Quick Start

> **Prerequisites:** Python 3.10+, CUDA 12.4+, 1-8 GPUs (single GPU works for small models)

```bash
# Clone the repository
git clone https://github.com/your-org/trainable-openclaw.git
cd trainable-openclaw

# Install dependencies
pip install -e .

# Start the self-evolving inference server
python -m trainable_openclaw.server.app --config configs/serve.yaml
```

The server starts in serving mode. Send a chat request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Write a Python function to merge two sorted lists"}]
  }'
```

When idle for 5 minutes, training kicks in automatically. Check the dashboard at `http://localhost:8000/dashboard`.

---

## Project Structure

```
trainable-openclaw/
├── trainable_openclaw/        # Core Python package
│   ├── server/                # FastAPI inference server
│   ├── engine/                # veRL engine wrapper
│   ├── agents/                # Agent connectors (openclaw, nanobot, etc.)
│   ├── logging/               # Conversation storage & compression
│   ├── evaluation/            # Rubric generation & LLM judge
│   ├── training/              # Idle detection & training orchestration
│   └── dashboard/             # Monitoring UI
├── configs/                   # Configuration files
├── docs/                      # Documentation & design docs
├── papers/                    # Research papers reference
├── tests/                     # Test suite
├── scripts/                   # Utility scripts
├── verl-main-0516/            # veRL reference implementation
└── requirements.txt
```

---

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the detailed development plan.

**Current phase (May–June 2026):** MVP — core self-evolving loop working end-to-end.

| Phase | Status |
|-------|--------|
| Phase 0 — Paper survey & algorithm design | In progress |
| Phase 1 — veRL dual-mode engine | ✅ A1, A2 complete |
| Phase 2 — Self-evolving evaluation system | Pending |
| Phase 3 — Integration & dashboard | Pending |
| Phase 4 — Benchmark & evaluation | Pending |

---

## Contributing

Trainable OpenClaw is in early development. We welcome contributions! Areas where help is especially valuable:

- **Algorithm research** — improving GRPO reward design, rubric generation quality
- **Engine backends** — adding support for more inference engines (TensorRT-LLM, llama.cpp)
- **Agent connectors** — integrating with more chatbot frameworks
- **Evaluation** — benchmarking rubric quality against human judgment
- **Documentation** — tutorials, deployment guides, best practices

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines (coming soon).

---

## License

[Apache 2.0](LICENSE) © 2026

---

## Acknowledgments

This project stands on the shoulders of giants. Special thanks to:

- **[veRL](https://github.com/volcengine/verl)** — Volcano Engine's reinforcement learning framework for LLMs, providing the core training and rollout infrastructure
- **[vLLM](https://github.com/vllm-project/vllm)** — High-throughput LLM serving engine, powering the inference backend
- **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** — NVIDIA's large-scale transformer training framework, enabling efficient distributed training
- **[SGLang](https://github.com/sgl-project/sglang)** — Structured generation language for LLMs, supported as an alternative inference backend
- **[DeepSeek](https://github.com/deepseek-ai)** — For DeepSeek-Flash and the open-source model ecosystem
- **[PyTorch](https://github.com/pytorch/pytorch)** — The foundational deep learning framework
- **[FastAPI](https://github.com/tiangolo/fastapi)** — Modern API framework powering the serving layer
- **[Ray](https://github.com/ray-project/ray)** — Distributed computing framework for scaling across GPUs
