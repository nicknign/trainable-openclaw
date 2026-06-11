---
name: codefuse-agent-deep-dive
description: Deep dive on CodeFuse-Agent (CFuse) — tools, architecture, training data, and relevance to nanobot general tool-use training
metadata:
  type: reference
---

# CodeFuse-Agent (CFuse) Deep Dive (2026-06-12)

## Overview
- **GitHub**: `codefuse-ai/CodeFuse-Agent` (released Dec 5, 2025)
- **Tagline**: "A lightweight, cleanly-architected agent framework designed for research and experimentation"
- **Install**: Single `pip install`
- **SWE-bench Lite**: 61% (CFuse + Claude Sonnet 4.5 single attempt); 61.67% with Trajectory-Aware Test-Time Scaling
- **License**: Not confirmed (check repo LICENSE directly)

## Architecture (5 layers)
1. **Interaction**: Terminal UI / Headless / HTTP modes
2. **Agent Loop**: LLM calls, tool dispatch, iteration control
3. **Context Engine**: Message history, context compression, prompt assembly
4. **LLM Provider**: OpenAI-compatible API
5. **Tool Execution**: 6 built-in tools, local or remote (HTTP sandbox)

## Tools — All 6 are code-focused

| CFuse Tool | nanobot Equivalent | Match? |
|-----------|-------------------|--------|
| `read_file` | `read_file` | EXACT |
| `write_file` | `write_file` | EXACT |
| `edit_file` | `edit_file` | EXACT |
| `grep` | `grep` | EXACT |
| `glob` | `find_files` | Near-exact |
| `bash` | `exec` | Near-exact |

**nanobot tools CFuse lacks (11 of 17)**:
list_dir, web_search, web_fetch, spawn, message, complete_goal, apply_patch, run_cli_app, cron, long_task, write_stdin

## Training Data
**NONE.** This is a framework, not a dataset. No pre-built trajectories, SFT corpora, or training data of any kind.

## Technical Report
- `tech_report.md` in the repo — describes agent architecture and SWE-bench benchmark methodology
- Not a research paper — more of a system description
- No model training experiments described

## Dual Execution Modes
- **Local Mode**: Execute tool calls directly in local environment
- **HTTP Mode**: Serve as tool execution backend or delegate to remote sandboxes
- HTTP mode decouples agent decisions from environment execution — useful for RL training pipeline scaffolding

## CodeFuse Ecosystem (related but NOT CFuse)
- **CodeFuse-CGM** (NeurIPS 2025): Graph-based LLM for repo-level SE, 44% SWE-bench Lite
- **CodeFuse-muAgent**: Multi-agent framework with KG engine
- **SWE-Fuse**: 14K verified trajectories, 60.2% SWE-bench Verified (32B)
- **CodeFuse-13B**: Pre-trained code LLM (200TB training data)

## Relevance to nanobot
- **For general tool-use training**: LOW — purely code-focused, wrong domain
- **As reference architecture**: MEDIUM — clean 5-layer design, HTTP mode for decoupled execution is a good pattern
- **For training data generation**: LOW — would need to extend with non-code tools, better to start from scratch
- **Bottom line**: Not suitable for general assistant training. Only relevant if we pivot to code-specific agent training.

## Key References
- GitHub: https://github.com/codefuse-ai/CodeFuse-Agent
