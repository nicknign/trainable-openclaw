---
name: mua-rl-deep-dive
description: Comprehensive deep dive on MUA-RL codebase — directory structure, all source files analyzed, loss masking mechanism traced, user simulation architecture, GRPO training pipeline, and direct comparison with trainable-openclaw
metadata:
  type: reference
---

# MUA-RL Deep Dive (Updated 2026-06-12)

## Paper
- **Title**: "MUA-RL: Multi-turn User-interacting Agent Reinforcement Learning for agentic tool use"
- **Authors**: Weikang Zhao, Xili Wang, Chengdi Ma et al. (Meituan + CASIA + Peking University)
- **arXiv**: 2508.18669, August 2025
- **GitHub**: `github.com/zzwkk/MUA-RL` (62 stars, Apache 2.0)
- **Dataset**: `huggingface.co/datasets/zzwkk/MUA-RL-Dataset`

## Repository Analysis

### What the Repo ACTUALLY Contains
MUA-RL is a **FORK of the veRL framework** (by Volcano Engine/ByteDance), NOT a standalone training repo. The MUA-RL specific additions are layered on top of veRL's existing multi-turn rollout infrastructure.

### Directory Structure (key directories only)

```
MUA-RL/
  MUA_environments/             # <-- NEW: Environment/tool system
    base/
      environment.py            # BaseEnvironment ABC
      data_loader.py            # BaseDataLoader ABC
      tool_registry.py          # ToolRegistry - manages tool instances
    factory.py                  # EnvironmentFactory - creates envs by type
    manager.py                  # EnvironmentManager - singleton, caches envs
    registry.py                 # Global environment registry
    taubench/
      retail/
        environment.py          # TauBenchRetailEnvironment - 16 tools
        data_loader.py           # Loads orders.json/products.json/users.json
      airline/
        environment.py          # TauBenchAirlineEnvironment
        data_loader.py

  examples/sglang_multiturn/    # <-- NEW: Training configs/scripts
    config/
      mua_multiturn_grpo.yaml   # Hydra config (max_turns=30, sglang rollout)
      tool_config/
        taubench_tool_config.yaml  # 16 retail + airline tools as YAML
    mua_grpo.sh                 # 4-node 32-GPU training script
    mua_8_gpus_test.sh          # Single-node 8-GPU test script

  data/                         # <-- Parquet test sets
    retail_test.parquet
    retail_test_empty_output.parquet
    airline_test.parquet
    airline_test_empty_output.parquet

  verl/tools/                   # <-- MODIFIED: Tool implementations added
    taubench_retail/            # 16 tool implementations (each a BaseTool subclass)
    taubench_airline/           # Airline tools
    base_tool.py                # BaseTool class (create/execute/calc_reward/release)
    schemas.py                  # OpenAIFunctionToolSchema

  verl/utils/reward_score/
    taubench.py                 # <-- NEW: Sparse binary reward (DB hash compare)

  verl/trainer/ppo/
    ray_trainer.py              # <-- MODIFIED: Loss mask support for multi-turn
    core_algos.py               # <-- STANDARD: GRPO advantage (no changes)

  verl/workers/rollout/sglang_rollout/
    sglang_rollout.py           # <-- MODIFIED: Multi-turn with user sim + loss mask
```

### License: Apache 2.0 (confirmed via GitHub API)

## Key Technical Details Found in Source Code

### 1. Loss Masking — Traced End-to-End

**Where loss_mask is created:**
In `sglang_rollout.py`, each `AsyncRolloutRequest` tracks `loss_mask` alongside `input_ids`, `attention_mask`, `position_ids`:
- Agent-generated text tokens → `loss_mask=1`
- Tool execution result tokens → `loss_mask=0`
- User message tokens (from GPT-4o simulator) → `loss_mask=0`
- The mask is maintained per-turn during the multi-turn conversation loop

**Where loss_mask is batched:**
`sglang_rollout.py:1072-1073` — Extracted from each request:
```python
prompt_loss_mask.append(torch.tensor(req.prompt_loss_mask, ...))
response_loss_mask.append(torch.tensor(req.response_loss_mask, ...))
```
`sglang_rollout.py:1117` — Combined:
```python
loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)
```

**Where loss_mask replaces response_mask:**
`ray_trainer.py:154-156` — Critical switch:
```python
if multi_turn:
    loss_mask = data.batch["loss_mask"]
    response_mask = loss_mask[:, -response_length:]
else:
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]
```

`dp_actor.py:380-383` — In the actual loss computation:
```python
if multi_turn:
    response_mask = data["loss_mask"][:, -response_length:]
else:
    response_mask = attention_mask[:, -response_length:]
```

**What this means:** The GRPO policy loss, advantage computation, and KL penalty are ALL computed using this masked response_mask. The model receives NO gradient signal from tool outputs or user messages — only from its own generated text.

### 2. Sparse Binary Reward Function

`verl/utils/reward_score/taubench.py` — `compute_score()`:
```python
def compute_score(taubench_database, messages, solution_str, ground_truth, extra_info):
    reward = 1.0
    # Check outputs: all ground_truth outputs must appear in agent messages
    for output in task_outputs:
        found = False
        for message in messages:
            if output.lower() in message['content'].lower():
                found = True; break
        if not found:
            r_outputs = 0.0; reward = 0.0
    
    # Check database state: final DB hash must match ground truth
    if not data_hash == gt_data_hash:
        reward = 0.0
    return reward  # 1.0 or 0.0 only
```

Two checks: (1) all expected information was communicated to the user, (2) the database ended up in the correct state. No partial credit, no format rewards.

### 3. User Simulation Architecture

- External LLM: `gpt-4o-2024-11-20` (configured via CHAT_MODEL, API_KEY, BASE_URL env vars)
- The user simulation is called DURING rollout in the SGLang engine
- Not a separate Python script — integrated into the sglang_rollout multi-turn loop
- When agent produces text without tool calls (i.e., talking to user), the User LLM is invoked to generate the next user message
- User messages are appended to the conversation with `loss_mask=0`
- Persona prompts are embedded in the task data, not hardcoded

### 4. Training Configuration

From `mua_grpo.sh` (4-node, 32-GPU):
- Batch size: 32, Mini-batch: 32, Rollouts per prompt: 8
- Learning rate: 1e-6, KL coefficient: 0.001, KL type: low_var_kl
- Max turns: 30, Max model length: 32768
- Temperature: 1.0, Epochs: 30
- SGLang rollout engine (NOT vLLM)
- GPU memory utilization: 0.8
- tensor_model_parallel_size: 2
- ulysses_sequence_parallel_size: 4
- Optimizer: AdamW (via veRL defaults)

From `mua_8_gpus_test.sh` (1-node, 8-GPU test):
- Batch size: 8, Rollouts: 16
- Same LR, KL penalty, temperature
- Response length per turn: 1024

### 5. Tool Architecture

Each tool is a `BaseTool` subclass with 4 lifecycle methods:
- `create(instance_id)` — initialize per-conversation tool state
- `execute(instance_id, parameters, shared_data)` — run the tool, return (response, reward, metrics)
- `calc_reward(instance_id)` — calculate step-level reward
- `release(instance_id)` — cleanup

Tools receive `shared_data` from the environment (e.g., in-memory database state). Retail environment has 16 tools: calculate, cancel_pending_order, exchange_delivered_order_items, find_user_id_by_email, find_user_id_by_name_zip, get_order_details, get_product_details, get_user_details, list_all_product_types, modify_pending_order_address, modify_pending_order_items, modify_pending_order_payment, modify_user_address, return_delivered_order_items, think, transfer_to_human_agents.

### 6. Environment Architecture

`BaseEnvironment` ABC provides:
- `get_data_loader()` — loads domain data (orders.json, etc.)
- `get_tool_registry()` — registers all tools
- `get_shared_data()` — returns mutable DB state that tools read/write
- `reset_environment()` — reinitialize for new episodes

`EnvironmentManager` singleton:
- Caches environment instances by type
- Extracts environment type from ability string ("retail", "airline")
- Provides tools list and terminal tools to the rollout engine

## HuggingFace Dataset

- `zzwkk/MUA-RL-Dataset` — Parquet format
- **CORRECTION (2026-06-12)**: Actually downloaded and inspected. Contains **1,580 samples** in `train` split only (no val/test). Contrary to the paper's emphasis on tau-bench (retail + airline), the dataset has **9 custom scenarios** with NO retail or airline:
  - `job_application_en` (263), `course_registration_en` (253), `real_estate_rental_en` (247), `meeting_room_booking_en` (240), `library_borrowing_en` (239), `travel_plan_zh` (216), `anime_cn` (75), `flight_information_en` (36), `cook_zh` (11)
  - Only `flight_information_en` is vaguely airline-adjacent (flight search, not booking)
- Format: `id` (UUID), `source` (scenario name), `messages` (JSON string — standard OpenAI function-calling), `tools` (JSON string — NON-STANDARD container: `{type:"function", function: [array]}` instead of standard one-function-per-block)
- Trajectories: full multi-turn conversations (avg 15.2 messages) with system prompt + user↔assistant↔tool loop, 66% end with `###STOP###` delimiter
- Detailed analysis: `tmp/mua_rl_data_analysis.md`
- License: Apache 2.0 (same as repo)
- Managed via HuggingFace datasets library (no auth needed for `load_dataset`)

## Comparison: MUA-RL vs trainable-openclaw

| Aspect | MUA-RL | trainable-openclaw |
|--------|--------|-------------------|
| **Training framework** | veRL (fork) + SGLang | veRL (recent fork) + vLLM |
| **Algorithm** | GRPO with KL penalty | GRPO with group-norm advantage |
| **Base model** | Qwen3-Non-Thinking 8B/14B/32B | Qwen3-4B (with mandatory thinking) |
| **GPU scale** | 4 nodes x 8 GPUs (32) | 1 x RTX 4090 (24GB) |
| **Rollout engine** | SGLang | vLLM (serve_ppo hybrid mode) |
| **Multi-turn** | Yes, built into sglang_rollout | No, single-turn only in training |
| **User simulation** | GPT-4o IN the RL loop (SGLang integration) | DeepSeek-v4-flash OUTSIDE training (separate script) |
| **Reward** | Sparse binary (0/1) via DB hash + message check | Rubric-based continuous (0-1) via LLM judge |
| **Loss masking** | Yes — masks tool outputs + user messages in loss | No — uses attention_mask only |
| **Tools** | Domain-specific (9 custom scenarios, 6-33 tools each, 129 total) | General purpose (17 nanobot tools) |
| **Task type** | Customer service (9 custom scenarios: library, courses, jobs, meetings, real estate, travel, anime, flights, cooking) | General assistant (OASST2 categories) |
| **SFT cold start** | ~2000 trajectories (9 scenarios) | ~500 trajectories (self-generated simulation) |
| **Data pipeline** | LLM-simulated + real MCP, human+LLM verified | OASST2 + User Sim correction dialogues |
| **Code maturity** | Published paper, Apache 2.0, ~5000KB repo | Personal project, in development |
| **Model weights** | Released on HuggingFace (MUA-RL-32B etc.) | Not released |
| **Inference** | Standard veRL inference | serve_ppo + nanobot gateway |
| **Logging** | TensorBoard + console | ConversationStore (SQLite) + JSONL |
| **Turn cap** | 30 turns (configurable) | Not implemented |

## Key Takeaways for trainable-openclaw

### What We Can DIRECTLY Reuse from MUA-RL

1. **Loss masking technique**: The approach of deriving `response_mask` from `loss_mask` during multi-turn training is directly applicable. We need to:
   - Track which tokens are agent-generated vs tool-output vs user-message during rollout
   - Pass `loss_mask` in the DataProto batch
   - Switch `response_mask` derivation in `ray_trainer.py` and `dp_actor.py`

2. **Environment/tool architecture**: The `BaseTool` lifecycle pattern (create→execute→calc_reward→release) is clean and adaptable to nanobot's filesystem+shell tools.

3. **SGLang multi-turn rollout**: MUA-RL's sglang_rollout multi-turn loop is the production implementation. However, we use vLLM, not SGLang, so this would need porting.

### What We CANNOT Directly Reuse

1. **SGLang-specific code**: We use vLLM's async server. The multi-turn rollout with in-loop user simulation is SGLang-specific and would need a complete rewrite for vLLM.

2. **User simulation in RL loop**: MUA-RL calls GPT-4o during SGLang's multi-turn loop. Our architecture separates user simulation (run_simulation.py) from training (serve_ppo). Integrating user sim into our vLLM-based training loop would be a substantial engineering effort.

3. **Domain-specific tools**: MUA-RL's 16 retail/airline tools have no overlap with nanobot's 17 general tools.

### Strategic Recommendations

**High-impact, low-effort adaptation:**
1. Implement loss masking in serve_ppo's GRPO training pipeline
2. Adopt sparse binary reward concept for tool-use scenarios (not replacing rubric rewards, but as an additional reward signal)

**Medium-impact, medium-effort:**
3. Port the BaseTool/Environment architecture into nanobot's tool system
4. Add multi-turn training support to our vLLM-based rollout

**Low-priority, research-only:**
5. Full user simulation in RL loop (requires significant vLLM modifications)
6. Port from vLLM to SGLang for multi-turn support

### Critical Differences

- **Scale**: MUA-RL runs on 32 H100s. We run on 1 RTX 4090. Their batch size (32) and sequence length (32K) are impossible for us. We need LoRA + gradient accumulation + shorter sequences.
- **Thinking**: MUA-RL uses Qwen3-Non-Thinking (no `<think>` tokens). We use Qwen3-4B which has hardcoded thinking. This affects max_model_len requirements and rollout throughput.
- **User simulation scope**: MUA-RL simulates customers within constrained business scenarios. Our User Sim handles 32 LMSYS categories with open-ended correction dialogues — much broader scope but less structured reward signal.

## References
- Paper: https://arxiv.org/abs/2508.18669
- GitHub: https://github.com/zzwkk/MUA-RL (Apache 2.0)
- Dataset: https://huggingface.co/datasets/zzwkk/MUA-RL-Dataset
- Models: https://huggingface.co/zzwkk/MUA-RL-32B
- Blog analysis: https://cognaptus.com/blog/2025-08-27-talk-tool-triumph-training-agents-with-real-conversations/
- Paper review: https://www.themoonlight.io/zh/review/mua-rl-multi-turn-user-interacting-agent-reinforcement-learning-for-agentic-tool-use
- WeChat analysis: http://mp.weixin.qq.com/s?__biz=MzAwMTY3NjA1OA==&mid=2247483769&idx=1&sn=6fea2cda712a57a586f8090262572280
