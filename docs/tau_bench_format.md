# tau-bench Data Format Analysis

Generated 2026-06-12. Covers all downloaded data sources and their compatibility with nanobot.

## Data Sources Overview

| Source | Location | Rows | Domains |
|--------|----------|------|---------|
| Historical Trajectories (tau-bench) | `data/tau_bench/raw/*.json` (4 files) | ~860 total | airline, retail |
| APIGen tau-bench (HF) | `data/tau_bench/raw/apigen_sample.json` (100-sample subset) | 46,127 total | airline, retail |
| Task Definitions (tau2-bench) | `data/tau_bench/raw/tasks_airline.json`, `tasks_retail.json` | 50 + 114 = 164 | airline, retail |
| Train/Test Splits | `data/tau_bench/raw/split_tasks_airline.json`, `split_tasks_retail.json` | N/A | airline, retail |
| Tool Definitions (Python) | `data/tau_bench/raw/tools_airline.py`, `tools_retail.py` | 13 + 15 tools | N/A |

### File Inventory (raw/)

| File | Size | Description |
|------|------|-------------|
| `gpt-4o-airline.json` | 4.0 MB | 200 gpt-4o trajectories, airline domain |
| `gpt-4o-retail.json` | 10.5 MB | 460 gpt-4o trajectories, retail domain |
| `sonnet-35-new-airline.json` | 10.5 MB | 400 Claude 3.5 Sonnet trajectories, airline |
| `sonnet-35-new-retail.json` | 25.7 MB | 400+ Claude 3.5 Sonnet trajectories, retail |
| `apigen_sample.json` | 5.8 MB | First 100 samples from amityco/apigen-tau-bench-split-turn |
| `apigen_single_sample.json` | 47 KB | Single APIGen sample for detailed inspection |
| `tasks_airline.json` | 155 KB | 50 airline task definitions |
| `tasks_retail.json` | 347 KB | 114 retail task definitions |
| `split_tasks_airline.json` | 1 KB | train=30 / test=20 split |
| `split_tasks_retail.json` | 2 KB | train=74 / test=40 split |
| `tools_airline.py` | -- | 13 airline tool implementations (Python) |
| `tools_retail.py` | -- | 15 retail tool implementations (Python) |

---

## 1. Historical Trajectories (tau-bench)

**License**: MIT
**Format**: JSON array of trajectory objects

### Top-Level Structure

```json
[
  {
    "task_id": 0,
    "reward": 0.0,
    "trial": 0,
    "info": {
      "task": { /* task definition object */ },
      "source": "human",
      "user_cost": 0.0,
      "reward_info": { ... }
    },
    "traj": [ /* array of messages */ ]
  },
  ...
]
```

- `reward`: 0.0 or 1.0 (binary pass/fail per trial)
- `trial`: integer trial number (0-indexed)
- `traj`: full multi-turn conversation with tool calls and responses

### Message Format

Each message in `traj` is a dict with the following fields by role:

| Role | Fields | Notes |
|------|--------|-------|
| `system` | `role`, `content` | Agent policy prompt + time context |
| `user` | `role`, `content` | User utterances only |
| `assistant` | `role`, `content`, `tool_calls` | `content` is `null` when tool_calls present |
| `tool` | `role`, `name`, `content` | Tool response; `name` = function name, `content` = JSON string |

**Assistant message with tool_calls:**
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_oIHazX6yQrB8hUwl4cRilFKj",
      "type": "function",
      "function": {
        "name": "get_user_details",
        "arguments": "{\"user_id\":\"mia_li_3668\"}"
      }
    }
  ]
}
```

**Tool response:**
```json
{
  "role": "tool",
  "name": "get_user_details",
  "content": "{\"name\": {\"first_name\": \"Mia\", \"last_name\": \"Li\"}, ...}"
}
```

**Key characteristics:**
- Standard OpenAI API message format
- `tool_calls` uses OpenAI's `function` wrapper
- Tool call arguments are JSON strings (NOT dicts)
- Tool responses use `role: "tool"` with `name` field
- No `tool_call_id` on tool responses (older OpenAI API format)

### Complete Example (task_id=0, reward=0.0)

This is a 32-message conversation where the agent:
1. Receives flight booking request
2. Calls `get_user_details` to verify identity
3. Calls `search_direct_flight` for flight options
4. Calls `search_onestop_flight` when user rejects direct options
5. Calls `calculate` for price computation
6. Calls `book_reservation` (fails initially due to miscalculation)
7. Calls `think` to debug, `calculate` to fix price
8. Calls `book_reservation` again (succeeds)
9. Delivers booking confirmation

The reward is 0.0 because this trajectory did not match the expected action sequence.

---

## 2. APIGen tau-bench (HuggingFace)

**Source**: `amityco/apigen-tau-bench-split-turn` on HuggingFace
**License**: MIT (derived from tau-bench)
**Size**: 46,127 rows

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `messages` | `list[dict]` | Full conversation (OpenAI format with tool calls) |
| `tools` | `list[dict]` | Available tool definitions (OpenAI function specs) |
| `conversations` | `list[dict]` | Identical structure to `messages` (duplicate) |
| `text` | `string` | Tokenized text representation of the conversation |

### Message Fields (APIGen)

Each message has ALL four fields, but `tool_calls` and `reasoning_content` are empty lists/strings when not applicable:

```json
{
  "content": "To assist you with the cancellation request...",
  "reasoning_content": "...",       // LLM thinking trace (may be empty string)
  "role": "assistant",
  "tool_calls": [...]               // empty list [] when no tool calls
}
```

**Key differences from historical trajectories:**
1. **`reasoning_content` field**: Always present (may be empty string `""`). Contains LLM reasoning/thinking trace.
2. **`tool_calls` is always present**: Empty list `[]` when no tool calls, vs. absent in historical.
3. **Same OpenAI structure**: `tool_calls[].function.name` + `tool_calls[].function.arguments` (JSON string).

### Tool Definitions Format

```json
{
  "type": "function",
  "function": {
    "name": "get_reservation_details",
    "description": "Get the details of a reservation...",
    "parameters": "{...}"  // JSON Schema string (NOT parsed dict)
  }
}
```

### messages vs conversations

The `conversations` column has **identical structure and content** to `messages`. Both are `list[dict]` with the same fields. Likely a formatting artifact with `conversations` being a pre-tokenized variant (paired with the `text` column).

---

## 3. Task Definitions (tau2-bench)

**Source**: `data/tau2/domains/{airline,retail}/tasks.json`
**Format**: JSON array of task objects

### Task Structure

```json
{
  "id": "0",                                    // string task ID
  "description": {
    "purpose": "Testing that agent refuses...", // what this task evaluates
    "relevant_policies": null,
    "notes": null
  },
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
    "actions": [],                               // expected DB actions (for action matching)
    "communicate_info": [],                      // expected info communication
    "nl_assertions": [                           // natural language assertions
      "Agent should refuse to proceed with the cancellation."
    ],
    "reward_basis": ["DB", "COMMUNICATE"]        // evaluation dimensions
  },
  "annotations": null
}
```

### Evaluation Criteria Types

- **`actions`**: Expected database operations that the agent must perform. Each action has an `action_id`, `name` (function name), `arguments`.
- **`nl_assertions`**: Natural language statements that the evaluator checks against the conversation using an LLM judge.
- **`reward_basis`**: Which evaluation dimensions apply — `DB` (database state checks), `COMMUNICATE` (information communicated), or `ACTION` (tool call sequence matching).
- **`communicate_info`**: Specific information the agent should tell the user.

### Task Counts

| Domain | Total | Train | Test |
|--------|-------|-------|------|
| Airline | 50 | 30 | 20 |
| Retail | 114 | 74 | 40 |
| **Total** | **164** | **104** | **60** |

Split keys: `train`, `test`, `base` (all), `small` (telecom only).

---

## 4. Tool Definitions

**Source**: `src/tau2/domains/{airline,retail}/tools.py`
**Format**: Python classes inheriting from `ToolKitBase`

### Airline (13 tools)
`book_reservation`, `cancel_reservation`, `get_reservation_details`, `get_user_details`, `list_all_airports`, `search_direct_flight`, `search_onestop_flight`, `send_certificate`, `update_reservation_baggages`, `update_reservation_flights`, `update_reservation_passengers`, `get_flight_status`, `calculate`, `think`, `transfer_to_human_agents`

### Retail (15 tools)
`cancel_pending_order`, `exchange_delivered_order_items`, `find_user_id_by_name_zip`, `find_user_id_by_email`, `get_order_details`, `get_product_details`, `get_item_details`, `get_user_details`, `list_all_product_types`, `modify_pending_order_address`, `modify_pending_order_items`, `modify_pending_order_payment`, `modify_user_address`, `return_delivered_order_items`, `calculate`, `think`, `transfer_to_human_agents`

### APIGen Tool Definitions

APIGen's `tools` column provides these same tools in OpenAI function-calling format (name + description + JSON Schema parameters). This is the format we need for nanobot integration.

---

## 5. Comparison with nanobot Message Format

### nanobot Internal Message (Session)

```python
msg = {
    "role": "user",           # or "assistant", "system", "tool"
    "content": "...",
    "timestamp": "2026-...",  # nanobot internal (strip for training)
    "tool_calls": [...],      # present on assistant messages
    "tool_call_id": "...",    # present on tool messages  
    "name": "...",            # present on tool messages
    "reasoning_content": "...", # optional
    "thinking_blocks": [...], # optional
    # nanobot-specific (strip for training):
    "media": [...],           # image attachments
    "cli_apps": [...],        # CLI app metadata
    "mcp_presets": [...],     # MCP server metadata
    "_channel_delivery": ..., # internal flag
}
```

### Field Mapping for Training Data

| tau-bench Field | nanobot Training Field | Notes |
|-----------------|----------------------|-------|
| `role` | `role` | Direct copy |
| `content` | `content` | Direct copy |
| `tool_calls[].function.name` | `tool_calls[].function.name` | Same structure |
| `tool_calls[].function.arguments` | `tool_calls[].function.arguments` | Both are JSON strings |
| `tool_calls[].id` | `tool_calls[].id` | Keep for OpenAI compatibility |
| `name` (tool responses) | `name` | Same |
| (missing in historical) | `tool_call_id` | Need to synthesize from `tool_calls[].id` for tool responses |
| `reasoning_content` (APIGen) | `reasoning_content` | Only in APIGen data, strip for basic SFT |
| N/A | `timestamp` | nanobot internal; omit in training data |

### Conversion Needed

1. **Historical trajectories -> nanobot training format**:
   - **Minor**: Add `tool_call_id` to tool responses. Historical trajectories use `name` but not `tool_call_id`. The `tool_call_id` can be synthesized by matching tool responses to preceding `tool_calls[].id`.
   - **Minor**: Reconstruct `tool_calls[].id` if missing (historical trajectories have them).
   - **Strip**: Remove `info` wrapper, keep only `traj` messages + task metadata.
   
2. **APIGen -> nanobot training format**:
   - **Minor**: Drop `conversations` (duplicate) and `text` columns.
   - **Keep**: `messages` and `tools` columns as-is; they are already compatible.
   - **Strip**: `reasoning_content` can be kept (nanobot supports it) or removed for basic SFT.

3. **Task definitions -> nanobot GRPO prompts**:
   - Convert `user_scenario.instructions.reason_for_call` + `known_info` into a user prompt.
   - Convert `evaluation_criteria` into reward signals.
   - System prompt: extract from the domain's agent policy (present in historical trajectories' `system` messages).

### What nanobot Replay Strips

From `session/manager.py:224-227`, nanobot strips `tool_calls`, `reasoning_content`, `thinking_blocks`, `tool_call_id`, and `name` from replay messages when building LLM context. This means tool call history is reconstructed differently for nanobot's internal replay vs. external training data. For training, we should use the full message format (including tool_calls).

---

## 6. Training Data Potential

### For SFT (Cold-Start Fine-Tuning)

| Source | Usable Examples | Filter |
|--------|----------------|--------|
| Historical trajectories (reward=1) | ~430 (estimate, 50% of 860) | Filter for `reward == 1.0` only |
| Historical trajectories (reward=0) | ~430 (failure examples) | Can be used as negative examples (DPO rejected) |
| APIGen | 46,127 | All usable; filter by quality if needed |
| **Total SFT potential**: ~46,500+ examples | | |

### For GRPO (RL Training)

| Source | Prompt Pool | Reward Signal |
|--------|------------|---------------|
| Task definitions (train split) | 104 prompts (30 airline + 74 retail) | `evaluation_criteria` via LLM judge or DB state check |
| Self-generated rollouts | Unlimited | Use tau-bench environment as reward function |

### Recommended Approach

1. **SFT cold-start**: Use 500-2000 APIGen examples (filtered for correct tool usage) as SFT data before GRPO.
2. **GRPO prompts**: Use 104 train-split task definitions. Convert `reason_for_call` + `known_info` into user prompts.
3. **Reward signal**: For GRPO, use the task's `evaluation_criteria.nl_assertions` with an LLM judge, or run the tau-bench evaluation harness directly.
4. **Test set**: 60 test-split tasks for evaluation. No contamination.

---

## 7. Data Quality Notes

### Historical Trajectories
- **Reward distribution**: Mixed 0.0 and 1.0 (model-dependent pass rates).
- **Conversation quality**: Real LLM conversations with actual tool calls. Tool call sequences can be incorrect (reward=0).
- **Token count**: 10-50 messages per trajectory. System prompt is ~1-2K tokens. Full trajectories are long.

### APIGen
- **Scale**: 46K examples is substantial. Format is clean and consistent.
- **`reasoning_content`**: Present on many assistant messages (thinking traces from the generating model). These are model-specific and may not transfer well.
- **Duplication risk**: The `messages` and `conversations` columns are identical — use only one.
- **Quality**: No per-example quality scores provided. Need to assess manually or via LLM judge.
