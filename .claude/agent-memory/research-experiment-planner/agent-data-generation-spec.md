---
name: agent-data-generation-spec
description: Complete implementation spec for agent tool-use training data pipeline -- 5 scenarios, 1000 tasks, 500 filtered trajectories, mock+live generation, rejection sampling, rubric quality filtering, train/test split, serve_ppo conversion
metadata:
  type: reference
---

# Agent Tool-Use Data Generation Pipeline Spec (2026-06-12)

Full implementation spec covering data format, task generation, expert trajectory generation (mock Phase A + live Phase B), rejection sampling, quality filtering, train/test split, and conversion to serve_ppo training pool format.

Key files:
- `data/agent_trajectories/train_prompts.jsonl` -- final training prompts for serve_ppo
- `data/agent_trajectories/train_sft.jsonl` -- SFT cold-start data
- `scripts/generate_agent_trajectories.py` -- main generation script (supports --mock and --live)

Targets: 1000 task definitions → ~570 after rejection → ~500 after quality filtering → 400 train / 100 test.

See also: [[project-phase2-status]] for prior data pipeline work, [[mua-rl-research]] for MUA-RL reference.
