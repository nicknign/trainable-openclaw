# Trainable OpenClaw

**A self-evolving LLM inference engine — learns from user interactions, improves during idle time.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-MVP-yellow)]()

Trainable OpenClaw wraps [veRL](https://github.com/volcengine/verl) into a self-improving inference service. It collects conversation feedback, evaluates quality via LLM-generated rubrics, and fine-tunes the model during idle time — all in a single deployment.

> **English** | [中文](README_zh.md)

---

## What It Does

General-purpose LLMs don't improve from usage. Every mistake is a lost learning opportunity. Trainable OpenClaw closes this loop:

```
User Request → veRL Engine (Inference) → Response → User
                                                  ↓
                                          Feedback Collection
                                                  ↓
                                    LLM Analyzes Feedback Patterns
                                                  ↓
                              LLM Generates Scoring Rubrics
                                                  ↓
                          Multi-answer Scoring → Reward Signal
                                                  ↓
                                    Idle Detection Triggers
                                                  ↓
                              LoRA Fine-tuning (GRPO)
                                                  ↓
                           Weight Sync → Back to Serving
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Trainable OpenClaw                      │
├────────────┬──────────────┬───────────────┬──────────────┤
│  API Server│  Simulation  │  Evaluation   │  Training    │
│  (FastAPI) │  Pipeline    │  Pipeline     │  Orchestrator│
│            │              │               │              │
│  /v1/chat  │  User Sim →  │  Feedback →   │  IdleDetect →│
│  /v1/health│  Correction →│  Rubric →     │  GRPO Train →│
│            │  Trajectory  │  Judge →      │  Weight Sync │
│            │              │  Evolve       │              │
├────────────┴──────────────┴───────────────┴──────────────┤
│                veRL Engine (vLLM + FSDP)                  │
│            Serving Mode ◄──────────► Training Mode        │
└──────────────────────────────────────────────────────────┘
```

---

## Current Status (MVP — June 2026)

**The framework is complete and functional.** All modules have been built, tested (154 tests), and verified end-to-end with real API calls on a remote GPU machine.

### What works

| Module | Description | Tests |
|--------|-------------|-------|
| API Server | OpenAI-compatible chat endpoint, WAL-logged conversations | 23 |
| Orchestrator | Idle detection → auto trigger GRPO training, 503 during training | 24 |
| Simulation | 5 user personas, multi-turn correction dialogues, 557 training pairs | — |
| Feedback Analyzer | LLM identifies error patterns from conversation logs | — |
| Rubric Generator | LLM generates quantifiable scoring rubrics from feedback | — |
| Judge Executor | Multi-rubric scoring with merged API calls, sync API for Ray | real-API ✅ |
| Rubric Evolver | Auto-evolve rubrics from low-score samples, archive stale ones | 25 + e2e ✅ |
| Metrics | Spearman, accuracy@k, coverage, convergence tracking | 27 |
| Pipeline | Pre-eval → Training → Post-eval → Rubric evolution CLI | 20 + GPU e2e ✅ |
| Dashboard | Streamlit panel: mode, progress, rubrics, conversations | 6 |

### What doesn't work yet

- **Training convergence** — Qwen3-4B + GRPO showed reward oscillation with no upward trend over 42 steps. A checkpoint evaluation confirmed degradation (correction rate 0.60 → 0.88). Likely a capacity issue — needs 7B+ model.
- **Checkpoint persistence across restarts** — Phase 2/3 checkpoints were lost, only Phase 1 step_10 survived.
- **S4 Reflection module** — Analyzing FAIL trajectories to improve the User Sim hasn't been built yet.

---

## Quick Start

> **Prerequisites:** Python 3.10+, CUDA 12.4+, 1-8 GPUs (single 48GB GPU works for Qwen3-4B)

```bash
git clone https://github.com/nicknign/trainable-openclaw.git
cd trainable-openclaw
pip install -e .
```

### 1. Start the inference server

```bash
bash scripts/start_train.sh
```

This starts veRL in HYBRID mode (vLLM + FSDP) with LoRA rank=16.

### 2. Send a chat request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Write a Python function to merge two sorted lists"}]
  }'
```

### 3. Run evaluation pipeline

```bash
# Pre-training baseline
python -m trainable_openclaw.pipeline --eval-only --max-test-prompts 10

# Generate training config
python -m trainable_openclaw.pipeline --gen-config

# Full pipeline report
python -m trainable_openclaw.pipeline --output results.json
```

### 4. Dashboard

```bash
streamlit run scripts/dashboard.py
```

### 5. Run tests

```bash
python -m pytest tests/ -v --ignore=tests/test_a1_integration.py --ignore=tests/test_a2_integration.py --ignore=tests/test_a3_integration.py
# 154 passed
```

---

## Project Structure

```
trainable-openclaw/
├── trainable_openclaw/          # Core Python package
│   ├── server/api.py            # FastAPI inference server (OpenAI-compatible)
│   ├── server/__init__.py
│   ├── logging/                 # Conversation store (SQLite + WAL) + CLI viewer
│   ├── training/                # Idle detection & training orchestration
│   ├── evaluation/              # Self-evolving evaluation system
│   │   ├── feedback.py          # B1: Feedback pattern analysis
│   │   ├── rubric.py            # B2: Rubric generation + store
│   │   ├── rubric_engine.py     # B2: LLM-driven rubric pipeline
│   │   ├── rubric_evolver.py    # B4: Auto-evolution from low-score logs
│   │   ├── judge.py             # B3: Multi-rubric scoring executor
│   │   ├── trajectory_eval.py   # S3: Trajectory grading + data export
│   │   ├── correction_rate.py   # D2: Correction rate evaluator
│   │   └── metrics.py           # S5: Evaluation metrics
│   └── pipeline.py              # C1: Main pipeline orchestrator
├── scripts/                     # Startup & utility scripts
│   ├── start_train.sh           # serve_ppo startup
│   ├── run_simulation.py        # User Sim correction dialogue pipeline
│   ├── validate_modules.py      # Real API validation (judge + evolver + pipeline)
│   ├── run_post_eval.py         # Checkpoint correction rate evaluation
│   ├── extract_lora.py          # FSDP checkpoint → LoRA weights
│   ├── convert_lora.py          # FSDP LoRA → PEFT/vLLM adapter format
│   ├── analyze_run.py           # Training log parser + analysis report
│   ├── dashboard.py             # Streamlit monitoring dashboard
│   └── chat.py                  # Interactive chat CLI
├── docs/                        # Documentation
│   ├── roadmap.md               # Development roadmap
│   ├── code_guide.md            # Code documentation
│   └── validation_report.md     # Framework validation report
├── tests/                       # 154 tests across 6 files
├── data/                        # Datasets, rubrics, conversation DB
├── runs/                        # Training run archives
├── verl-main-0516/              # Modified veRL (serve_ppo + checkpoint + weight sync)
└── requirements.txt
```

---

## How It Compares to agent-lightning

Microsoft's [agent-lightning](https://github.com/microsoft/agent-lightning) is an excellent framework for training AI agents with RL. Trainable OpenClaw makes a different architectural choice:

| | agent-lightning | Trainable OpenClaw |
|---|---|---|
| **Paradigm** | Training framework wrapping agents | Self-evolving inference engine |
| **Engine** | External (proxied via LiteLLM) | Internal (deeply modified veRL) |
| **Training trigger** | Algorithm-driven loop | Idle-driven (background automatic) |
| **User model** | You bring agent code | You call an HTTP endpoint |

**Use agent-lightning** if you have an agent and want to train it.
**Use Trainable OpenClaw** if you want an inference service that gets smarter over time.

---

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the full plan.

| Phase | Status |
|-------|--------|
| Phase 0 — Paper survey & algorithm design | Background ongoing |
| Phase 1 — veRL dual-mode engine | ✅ A1+A2+A3 complete |
| Phase 1.5 — Data engineering & simulation | ✅ S1+S2+S3+S5 complete, S4 pending |
| Phase 2 — Self-evolving evaluation | ✅ B0+B1+B2+B3+B4 complete |
| Phase 3 — Integration & dashboard | ✅ C1+C2 complete |
| Phase 4 — Evaluation | 🟡 D1+D2 complete, D3 deferred |

---

## License

[Apache 2.0](LICENSE) © 2026

---

## Acknowledgments

- **[veRL](https://github.com/volcengine/verl)** — Core training & inference infrastructure
- **[vLLM](https://github.com/vllm-project/vllm)** — High-throughput LLM serving
- **[DeepSeek](https://github.com/deepseek-ai)** — Judge model (deepseek-v4-flash)
- **[FastAPI](https://github.com/tiangolo/fastapi)** — API serving layer
- **[Ray](https://github.com/ray-project/ray)** — Distributed computing
- **[PyTorch](https://github.com/pytorch/pytorch)** — Deep learning framework
