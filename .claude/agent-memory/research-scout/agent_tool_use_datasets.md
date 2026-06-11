---
name: agent-tool-use-datasets
description: Comprehensive catalog of agent tool-use / function-calling datasets found, with relevance scoring for nanobot's 17-tool set
metadata:
  type: reference
---

# Agent Tool-Use / Function-Calling Datasets (2026-06-11 research)

## nanobot Tool Set (17 tools)
exec, read_file, write_file, edit_file, grep, find_files, list_dir, web_search, web_fetch, spawn (sub-agent), message, complete_goal, apply_patch, run_cli_app, cron, long_task, write_stdin

## Tier 1 - Best Matches (Code/File/Shell tools)

### MUA-RL (Meituan/CASIA, Aug 2025) -- BEST MATCH FOR GRPO TRAINING
- Paper: arXiv 2508.18669
- GitHub: `github.com/zzwkk/MUA-RL`
- Dataset: `huggingface.co/datasets/zzwkk/MUA-RL-Dataset` (~2000 SFT trajectories, 9 scenarios)
- Tools: Domain-specific APIs per scenario (flexible tool surface — agent learns to adapt, not memorize fixed set)
- Methodology: SFT cold start + GRPO + simulated user IN THE RL LOOP + sparse binary reward + loss masking
- Framework: VeRL (same as our stack), Qwen3 backbones (8B/14B/32B)
- Key techniques we must adopt: loss masking (mask tool outputs + user messages), sparse binary reward (r=1 for task completion else 0)
- Fit: HIGHEST relevance — methodology maps directly to nanobot GRPO training. See mua_rl_deep_dive.md.
- Score: HIGH (methodology, NOT dataset content)

### SWE-smith (NeurIPS 2025 Spotlight)
- HuggingFace: `SWE-bench/SWE-smith`
- Size: 52k task instances, 26k+ agent trajectories (5k expert used for training)
- Tools: bash, file read, file edit, search, git
- Format: SWE-agent trajectory format (thought/action/observation loops)
- License: Apache 2.0
- Best model: SWE-agent-LM-32B (40.2% on SWE-bench Verified)
- Fit: Closest match to nanobot's code tools (exec/read_file/write_file/edit_file/grep/find_files)
- Score: HIGH

### SWE-Gym (ICML 2025)
- GitHub: `github.com/SWE-Gym/SWE-Gym`
- Size: 2,400 instances from 11 Python repos
- Tools: bash, file edit, search
- Format: Agent-environment interaction trajectories
- License: Open source
- Result: 32% on SWE-Bench Verified (32B model)
- Fit: Smaller but executable-verified, good for GRPO training
- Score: HIGH

### CodeFuse-Agent / CFuse (2025-12)
- GitHub: `github.com/codefuse-ai/CodeFuse-Agent`
- Tools: read_file, write_file, edit_file, grep, glob, bash (6 tools, subset of nanobot)
- Format: Configurable agent profiles, HTTP/Local exec, 5-layer architecture
- SWE-bench Lite: 61% (with Claude Sonnet 4.5)
- Fit: 6/6 tools match nanobot's CODE tools, but purely code-focused. No non-code tasks.
- Limitation: No training data (framework only); purely SWE domain; 11 of nanobot's 17 tools not covered
- Score: LOW for general tasks (see codefuse_agent_deep_dive.md). MEDIUM as reference architecture.
- Verdict: Wrong domain for general assistant training. Good architecture reference (configurable profiles, HTTP sandbox mode).

### Toucan (2025)
- HuggingFace: `Agent-Ark/Toucan-1.5M`
- Size: 1.5M trajectories, 495 MCP servers, 2,000+ tools
- Format: Multi-step tool calls with real API execution
- License: Open source
- Fit: Covers many tool types including exec/read/write/search
- Limitation: 1.5M is overwhelming for single GPU; subset needed
- Score: MEDIUM (too large, needs filtering)

## Tier 2 - Function Calling (adaptable)

### FC-synth (recent, active)
- GitHub: `pierpierpy/FC-synth` (generation pipeline), HuggingFace: `pierjoe/function-calling-synthetic-2000` (example dataset)
- Size: Configurable generation; example is 2000 samples; 120+ domains, 200 mock tools across 21 categories
- Format: Multi-turn tool-calling conversations, handles "when NOT to call", edge cases, multi-language
- License: Check repo
- Fit: Could generate nanobot-specific training data by swapping in nanobot tool definitions. Good for format diversity.
- Score: MEDIUM (generation tool, not pre-built dataset)

### UniToolCall (2025)
- HuggingFace: `EIT-NLP/UniToolCall`
- Size: 22k+ tools, 390k+ training instances
- Format: Standardized QAOA (Query-Action-Observation-Answer), covers single/multi-hop, serial/parallel, cross-turn dependency
- License: Check HuggingFace
- Result: Qwen3-8B fine-tuned reaches 93% single-turn strict precision on BFCL
- Fit: Good for learning tool-calling FORMAT, but tools are REST APIs not file/shell/web. Useful as format exemplars for SFT cold start.
- Score: MEDIUM (format reference, not content match)

### ToolACE (ICLR 2025)
- HuggingFace: `Team-ACE/ToolACE`
- Size: 10k-100k samples, 26,507 APIs, 390 domains
- Format: JSON/YAML/XML/Markdown function calls, single/parallel/dependent, multi-turn
- License: Apache 2.0 (BEST LICENSE)
- Best model: ToolACE-8B (competitive with GPT-4 on BFCL)
- Fit: Excellent for general function calling; would need adaptation to nanobot tools
- Score: HIGH

### xLAM Function Calling 60k / APIGen (2024)
- HuggingFace: `Salesforce/xlam-function-calling-60k`
- Size: 60,000 entries, 3,673 executable APIs, 21 categories
- Format: XML-tagged (query/tools/answers), verifiable via execution
- License: CC-BY-NC-4.0 (NON-COMMERCIAL)
- Best model: xLAM-1B beats GPT-3.5 on BFCL
- Fit: Highest quality function calling data, but NC license is problematic
- Score: MEDIUM (license constraint)

### Glaive Function Calling v2
- HuggingFace: `glaiveai/glaive-function-calling-v2`
- Size: ~113K examples
- Format: Multi-turn (system/user/assistant/function_call/function_response)
- License: Apache 2.0 (original), CC-BY-SA-4.0 (variants)
- Limitation: Simple scenarios only, no multi-function calls per turn, no error handling
- Fit: Good baseline for function call format, needs augmentation
- Score: MEDIUM

### Arcee Agent Data (2024)
- HuggingFace: `arcee-ai/agent-data`
- Size: Blend of Glaive FC v2 + xLAM + Agent-Flan + Magpie Pro 300k
- Format: Multi-format training (XML-style function tags)
- License: MIT
- Fit: Good multi-source blend with sequential tool call extension
- Score: MEDIUM-HIGH

### BFCL Dataset (NeurIPS 2024)
- HuggingFace: `gorilla-llm/Berkeley-Function-Calling-Leaderboard`
- Size: ~3,500 prompts (V4), ~2,000 (V1)
- Format: JSON with function definitions, AST-evaluable, multi-turn (V3+)
- License: Apache 2.0
- Fit: Primarily evaluation benchmark, not training dataset
- Score: MEDIUM (eval use)

## Tier 3 - Agent Trajectories (multi-turn, reasoning)

### AgentBank (EMNLP 2024)
- HuggingFace: `Solaris99/AgentBank`
- Size: 50,000+ trajectories, 16 tasks, 5 skill dimensions
- Format: Multi-turn CoT + action + observation, chatbot-style conversations
- License: Open access
- Fit: Large general-purpose agent dataset, would need tool remapping
- Score: MEDIUM

### ToolMind (2025)
- HuggingFace: `Nanbeige/ToolMind`
- Size: 360k samples (160k synthetic + 200k augmented), 20k+ functions
- Format: Multi-agent simulated (User/Assistant/Tool roles), turn-level filtered
- License: Check on HF
- Performance: +5.4% BFCL-v4, +14.22% tau-bench (Qwen3-14B)
- Fit: Excellent quality with reasoning traces, but very large
- Score: MEDIUM

### AgentTuning / AgentInstruct (ACL 2024)
- GitHub: `github.com/THUDM/AgentTuning`
- Size: 1,866 trajectories, 6 tasks
- Format: ReAct-style CoT prompting, GPT-4 generated, reward-filtered
- License: Open source
- Fit: High quality, small enough for single GPU
- Limitation: Very small (1,866), tools don't match nanobot
- Score: LOW-MEDIUM

## Tier 4 - Benchmarks (evaluation only)

### tau-bench (Sierra Research, 2024) + tau2-bench (2025)
- GitHub: `github.com/sierra-research/tau-bench` (original), `github.com/sierra-research/tau2-bench` (current)
- Size: 165 tasks (115 retail + 50 airline), domain-specific database APIs (~28 tools total)
- tau2-bench adds: 114 telecom tasks (dual-control Dec-POMDP, user also has tools)
- Format: Multi-turn user-agent conversations against hidden JSON databases
- License: MIT or Apache 2.0 (ambiguous, check repo LICENSE)
- Training data: Eval-only benchmark, BUT ToolMind generated 12,882 training trajectories from tau-bench environments
- Fit: Good seed prompt source for self-generating training data. pass^k metric is excellent for evaluation.
- See: tau_bench_deep_dive.md for full analysis

### GAIA (Meta/Fair, 2023)
- Size: 466 questions
- Format: Multi-step reasoning requiring web search, file reading, code execution
- License: Open
- Fit: Good eval for general agent capabilities

### AgentBench (ICLR 2024)
- Size: 8 environments (OS, DB, Web, etc.)
- Format: Multi-turn agent interactions
- Fit: Broad evaluation benchmark

## Key Synthesis Observations

1. **No dataset matches nanobot's tool set exactly.** The closest is SWE-smith (bash + file ops) and CodeFuse-Agent (read_file, write_file, edit_file, grep, glob, bash).

2. **Apache 2.0 datasets to prioritize:** ToolACE, SWE-smith, BFCL, Glaive FC v2

3. **Self-generation is the norm.** Every top project (ToolACE, APIGen, Toucan, ToolMind, SWE-smith) generates its own data synthetically. The open datasets are mostly evaluation benchmarks.

4. **Multi-turn + real execution > single-turn synthetic.** TOUCAN and SWE-smith's key insight: verifiable data from real execution dramatically outperforms LLM-simulated tool responses.

5. **GRPO training for agents is emerging.** MUA-RL (2025) shows GRPO with simulated users works for multi-turn tool use. This maps directly to our architecture.
