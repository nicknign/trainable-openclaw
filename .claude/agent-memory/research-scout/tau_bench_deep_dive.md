---
name: tau-bench-deep-dive
description: Comprehensive search results on tau-bench, tau2-bench, tau3-bench — actual task files, HF datasets, historical trajectories, licenses, and training suitability for nanobot GRPO
metadata:
  type: reference
---

# tau-bench (Sierra Research) -- Full Search Results (2026-06-12)

## Paper & Origin
- **Title**: "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains"
- **Authors**: Shunyu Yao, Noah Shinn, Pedram Razavi, Karthik Narasimhan (Sierra Research / Princeton)
- **arXiv**: 2406.12045, June 2024
- **tau2-bench paper (Barres et al.)**: arXiv 2506.07982, June 2025

## Repositories (GitHub)

| Repo | Status | License |
|------|--------|---------|
| `sierra-research/tau-bench` | **Original, deprecated** (tasks not updated) | MIT |
| `sierra-research/tau2-bench` | **Current** (package name `tau2`, version tau3-bench v1.0.0) | MIT |
| `AGI-Eval-Official/tau2-bench-revised` | Community fork: airline task fixes + user sim prompt improvements (Claude Opus 4.5 fixes + GLM-4.5 approach) | MIT |

## Domains & Task Counts (tau2-bench, the current version)

### tau2-bench/data/tau2/domains/{domain}/tasks.json
Data lives at `data/tau2/domains/{domain}/tasks.json` (JSON, NOT embedded in Python).

| Domain | Total Tasks | Train | Test | Notes |
|--------|-------------|-------|------|-------|
| **airline** | 50 | 30 | 20 | Flight booking, cancellation, modification |
| **retail** | 114 | 74 | 40 | Order management, returns, exchanges |
| **telecom** | 2,285 | 74 | 40 (+small:20) | Mobile data issues, roaming, dual-control |
| **banking_knowledge** | 97 | No split | No split | RAG-based customer support (698 documents) |
| **mock** | 10 | No split | No split | Test domain |

Split defined in `split_tasks.json` files: `{"train": [...task_ids...], "test": [...task_ids...], "base": [...all_task_ids...]}`.
Only airline and retail have explicit train/test splits. telecom has `small`/`train`/`test`/`full`/`base`.

### Task Format (JSON)
```json
{
  "id": "0",
  "description": {"purpose": "...", "relevant_policies": null, "notes": null},
  "user_scenario": {
    "persona": null,
    "instructions": {
      "task_instructions": "Behavior if agent says X...",
      "domain": "airline",
      "reason_for_call": "You want to cancel reservation EHGLP3...",
      "known_info": "You are Emma Kim. Your user id is emma_kim_9957.",
      "unknown_info": null
    }
  },
  "initial_state": null,
  "evaluation_criteria": {
    "actions": [{"action_id": "1_0", "name": "get_user_details", "arguments": {...}}],
    "communicate_info": [],
    "nl_assertions": ["Agent should refuse to proceed with the cancellation."],
    "reward_basis": ["DB", "COMMUNICATE"]
  },
  "annotations": null
}
```

## Tools per Domain

### Airline (13 tools)
book_reservation, cancel_reservation, get_reservation_details, get_user_details, list_all_airports, search_direct_flight, search_onestop_flight, send_certificate, update_reservation_baggages, update_reservation_flights, update_reservation_passengers, get_flight_status, calculate, think, transfer_to_human_agents

### Retail (15 tools)
cancel_pending_order, exchange_delivered_order_items, find_user_id_by_name_zip, find_user_id_by_email, get_order_details, get_product_details, get_item_details, get_user_details, list_all_product_types, modify_pending_order_address, modify_pending_order_items, modify_pending_order_payment, modify_user_address, return_delivered_order_items, calculate, think, transfer_to_human_agents

### Telecom (18+ tools, dual-control)
disconnect_vpn, enable_roaming, disable_roaming, refuel_data, set_network_mode_preference, toggle_airplane_mode, toggle_data, toggle_data_saver_mode, toggle_roaming, suspend_line, resume_line, get_data_usage, set_data_usage, get_customer_by_phone/name/id, get_bills_for_customer, send_payment_request, etc.
**Key**: Telecom has user-side tools (dual-control Dec-POMDP) -- user can also take actions in the environment.

## Historical Trajectories (ACTUAL TRAINING DATA!)

The **original** tau-bench repo contains actual model run trajectories at `/historical_trajectories/`:

| File | Trajectories | Model | Size |
|------|-------------|-------|------|
| `gpt-4o-airline.json` | 200 | gpt-4o | 4.2 MB |
| `gpt-4o-retail.json` | 460 | gpt-4o | 11.0 MB |
| `sonnet-35-new-airline.json` | 400 | Claude 3.5 Sonnet | 11.0 MB |
| `sonnet-35-new-retail.json` | (est. 400+) | Claude 3.5 Sonnet | 26.9 MB |

**Total: ~860+ real agent conversation trajectories with tool calls.**

### Trajectory Format
```json
{
  "task_id": 0,
  "reward": 0.0,
  "info": {"task": {...}, "expected_actions": [...]},
  "traj": [
    {"role": "system", "content": "# Airline Agent Policy\n..."},
    {"role": "user", "content": "Hi! I'm looking to book a flight..."},
    {"role": "assistant", "content": "To assist you, I'll need your user ID."},
    {"role": "user", "content": "Sure, my user ID is mia_li_3668."},
    {"role": "assistant", "content": null, "tool_calls": [{"function": {"name": "get_user_details", "arguments": "{\"user_id\": \"mia_li_3668\"}"}, ...}]},
    {"role": "tool", "content": "{\"name\": {\"first_name\": \"Mia\", ...}}", "name": "get_user_details"},
    ...
  ],
  "trial": 0
}
```

**This is directly usable as SFT training data**: multi-turn conversations with correct tool call sequences (for reward=1 trajectories) and failure examples (for reward=0 trajectories).

## HuggingFace Datasets Found

### TRAINING DATASETS (directly usable)

| Dataset | Rows | Columns | Description |
|---------|------|---------|-------------|
| **`jkazdan/taubench_traces_training_data`** | 50 | `messages` | Multi-turn airline traces with tool calls in OpenAI messages format |
| **`amityco/apigen-tau-bench-split-turn`** | 46,127 | `messages`, `tools`, `conversations`, `text` | Massive training set: system instructions + tool definitions + multi-turn conversations + tokenized text |
| **`amityco/tau-bench-retail-train-next-action-all-step-v0.2`** | 2,206 | `conversations`, `answer` | Retail: each row is partial conversation + next expected tool call (next-action prediction format) |
| **`amityco/tau-bench-retail-train-next-action-all-step-score-v0.2`** | 2,206 | (conversations + scores) | Same as above but with per-step reward scores |
| **`amityco/tau-bench-retail-train-next-action-hard-v0.2`** | (small) | conversations | Harder-only retail examples |

### EVALUATION-ONLY

| Dataset | Rows | Description |
|---------|------|-------------|
| **`Snorkeler/tau-bench-react-claude`** | 0 (empty) | Listed but contains no data files -- dead dataset |
| **`chenjoachim/TAU-Benchmark`** | 1,794 | **NOT** the Sierra tau-bench! This is a QA/multimodal benchmark with `question`/`answer`/`type`/`hopType`/`audioPath` fields. Completely different. |
| **`jerry128/taubench-tool-calling-Qwen2.5-7B-Instruct-*`** | ~50 | Model-specific run traces for evaluation |

### mzio/aprm-* series (10+ datasets)
These are SFT training data generated by running agents on tau-bench retail/airline, all ~100-500 examples each, using think+act formats. Authorized by `mzio`. Format includes think tokens. Good for studying agent reasoning patterns.

## tau-bench-fixed / tau2-bench-revised

**`AGI-Eval-Official/tau2-bench-revised`** (on GitHub):
- NOT a separate dataset -- a **patch** for the original tau2-bench
- Contains:
  1. Fixed `airline/tasks.json` (adopts Claude Opus 4.5 fixes)
  2. Improved retail user simulation prompt (from GLM-4.5 paper)
  3. Improved telecom user simulation prompt (for transfer handling)
- MIT licensed
- Drop-in replacement: replace files in original tau2-bench

## Comparison with MUA-RL

| Dimension | tau-bench (Sierra) | tau2-bench revised | MUA-RL |
|-----------|-------------------|--------------------|--------|
| **Training data** | NO built-in training split BUT historical trajectories (~860) in repo + 46K on HF | Same patches | YES -- 2,000 SFT trajectories |
| **Domain** | Retail + Airline (customer service) | Same + fixes | Retail + Airline (customer service) |
| **Tools** | 13-15 REST-like APIs (get_order_details, book_reservation, etc.) | Same | 13-15 REST-like APIs |
| **Format** | Eval tasks = JSON scenarios; Trajectories = OpenAI messages with tool calls | Same | SFT: {"messages": [...], "tools": [...]} |
| **License** | MIT | MIT | Apache 2.0 |
| **GRPO ready?** | Requires self-generation of rollouts | Same patches | YES -- designed for GRPO (VeRL framework) |
| **Dual-control** | No (except telecom) | Improved user sim | No |
| **Task count** | 165 tasks (50+115) | 165 tasks | Unknown (pipeline-generated) |

## Training/Evaluation Split Summary (tau2-bench)

| Domain | Train | Test | Notes |
|--------|-------|------|-------|
| airline | 30 | 20 | Explicit split in split_tasks.json |
| retail | 74 | 40 | Explicit split |
| telecom | 74 | 40 (+20 small) | Explicit split |
| banking_knowledge | None | 97 | No split defined |

## Key Takeaways for nanobot Training

1. **Best training data**: `amityco/apigen-tau-bench-split-turn` (46,127 rows) -- messages + tools + conversations format, directly convertible to SFT format
2. **Historical trajectories**: 860+ real model runs in the original tau-bench repo -- can filter for reward=1 trajectories as gold SFT data
3. **tau-bench is NOT eval-only anymore**: While originally designed as an evaluation benchmark, the community has generated substantial training data and the repo ships with actual run traces
4. **For GRPO training**: Unlike MUA-RL (which is purpose-built for GRPO), tau-bench requires self-generating rollouts OR using existing traces as SFT cold-start
5. **Recommended approach**: Use HF training data as SFT cold-start, then use the 165 tasks as GRPO prompts with the tau-bench environment as reward signal
6. **License**: All MIT -- safe to use

## Downloaded Data (2026-06-12)

All data downloaded to `data/tau_bench/raw/` on Windows (local machine).

### Historical Trajectories (tau-bench)
| File | Size | Trajectories |
|------|------|-------------|
| `gpt-4o-airline.json` | 4.0 MB | 200 |
| `gpt-4o-retail.json` | 10.5 MB | 460 |
| `sonnet-35-new-airline.json` | 10.5 MB | 400 |
| `sonnet-35-new-retail.json` | 25.7 MB | 400+ |

**Format**: JSON array of `{task_id, reward, trial, info, traj: [...]}`. Messages in standard OpenAI API format (role, content, tool_calls with function.name/arguments as JSON strings, tool responses with role=tool + name). No tool_call_id on tool responses (older format).

### APIGen HF Data
- **Source**: `amityco/apigen-tau-bench-split-turn` (46,127 rows, HF datasets library)
- **Samples**: `apigen_sample.json` (100 rows, 5.8 MB), `apigen_single_sample.json` (1 row, 47 KB)
- **Columns**: messages, tools, conversations (structurally identical to messages), text (tokenized)
- **Format**: messages have {content, reasoning_content, role, tool_calls[]}. Same OpenAI structure as historical.

### Task Definitions (tau2-bench)
- `tasks_airline.json` (155 KB): 50 tasks. `tasks_retail.json` (347 KB): 114 tasks.
- **Format**: `{id, description {purpose, relevant_policies, notes}, user_scenario {persona, instructions {task_instructions, domain, reason_for_call, known_info, unknown_info}}, initial_state, evaluation_criteria {actions[], communicate_info[], nl_assertions[], reward_basis[]}, annotations}`
- Evaluation: actions (expected DB ops), nl_assertions (LLM judge), reward_basis (DB/COMMUNICATE/ACTION)

### Splits
- `split_tasks_airline.json`: train=30, test=20, base=50
- `split_tasks_retail.json`: train=74, test=40, base=114
- **GRPO prompt pool**: 104 train tasks total

### Tool Definitions (Python source)
- `tools_airline.py` (29 KB): 13 airline tools (book_reservation, cancel_reservation, get_reservation_details, get_user_details, search_direct_flight, search_onestop_flight, send_certificate, update_reservation_flights, update_reservation_baggages, update_reservation_passengers, get_flight_status, calculate, think, transfer_to_human_agents)
- `tools_retail.py` (30 KB): 15 retail tools (cancel_pending_order, exchange_delivered_order_items, find_user_id_by_name_zip, find_user_id_by_email, get_order_details, get_product_details, get_item_details, get_user_details, list_all_product_types, modify_pending_order_address, modify_pending_order_items, modify_pending_order_payment, modify_user_address, return_delivered_order_items, calculate, think, transfer_to_human_agents)

### Format Analysis
Complete field-by-field analysis at `docs/tau_bench_format.md` -- covers field mapping, nanobot compatibility, conversion requirements (minor: add tool_call_id to tool responses), training data potential: SFT ~46,500 examples, GRPO 104 train prompts + 60 test tasks.

### Cloned Repos (can delete to save ~830 MB)
- `data/tau_bench/raw/tau-bench/` (56 MB) -- original repo with historical trajectories
- `data/tau_bench/raw/tau2-bench/` (776 MB) -- current repo with tasks + full Python source

### Download Script
`scripts/download_tau_bench.py` -- automated download for all three sources (historical trajectories, APIGen HF, task definitions)
