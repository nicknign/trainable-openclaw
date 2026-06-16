# Agent Loop Migration Plan

> 用 verl 内置 Agent Loop 替换 nanobot 作为训练时的多轮工具调用路径。
> nanobot 保留用于 serving/eval，不改动。

## 1. 架构对比

### 当前 (nanobot 路径)

```
verl rollout (vllm 醒着) → sleep vllm → compute_score → curl nanobot → ❌ vllm 已休眠
                                                                    → ✓ fallback 单轮评分
```

问题：单卡时 nanobot 不可达，多轮变单轮。多卡可解决但需额外维护 nanobot 服务。

### 新方案 (Agent Loop 路径)

```
AgentLoopManager.wake_up(vllm)
  → ToolAgentLoop.run():
      PENDING  → apply_chat_template(prompt + tool_schemas)
      GENERATE → vllm.generate() → 模型生成
      TOOLS    → extract_tool_calls() → execute_tools() → tokenize results
      GENERATE → vllm.generate()  (下一轮)
      ... 循环至 max_user_turns=3 或 无 tool_call
  → return AgentLoopOutput(prompt_ids, response_ids, response_mask)
AgentLoopManager.sleep(vllm)
→ trainer._compute_reward_colocate() → compute_score() → rubric 评分
```

工具调用在 rollout 阶段完成，vllm 全程醒着。无外部服务依赖。

---

## 2. 改动清单

### 2.1 新建文件

| 文件 | 用途 | 预估行数 |
|------|------|----------|
| `trainable_openclaw/training/agent_tools.py` | `@function_tool` 装饰的 retail 工具，供 AgentLoop 调用 | ~50 行 |
| `scripts/data/add_agent_name.py` | 给 train/val jsonl 添加 `agent_name: tool_agent` 字段 | ~30 行 |

### 2.2 修改文件

| 文件 | 改动 | 影响范围 |
|------|------|----------|
| `scripts/train/grpo_retail.yaml` | rollout 从 hybrid→async，新增 multi_turn + agent 配置段 | 训练配置 |
| `trainable_openclaw/training/grpo_reward.py` | compute_score 适配 AgentLoop 输出的完整轨迹（含 tool tokens） | reward 逻辑 |
| `scripts/deploy/setup_env.sh` | 不再检查 nanobot（训练时不需要） | 环境脚本 |
| `scripts/train/run_grpo_direct.sh` | 更新注释和说明 | 启动脚本 |

### 2.3 不改的文件

| 文件 | 原因 |
|------|------|
| `trainable_openclaw/nanobot_tools/tau_bench_retail.py` | nanobot 仍用于 serving/eval |
| `trainable_openclaw/training/rollout_env.py` | ToolExecutor 仍被 reward fallback 使用 |
| `trainable_openclaw/training/rubric_rules.py` | 评分逻辑不变 |
| `trainable_openclaw/agent/tau_bench_tools/` | 核心工具定义不变，双方共享 |
| `scripts/deploy/start_experience.sh` | nanobot 服务栈仍用于 eval |

---

## 3. 配置变更

### grpo_retail.yaml 关键 diff

```yaml
# ── 删除 ──
actor_rollout_ref:
  hybrid_engine: true        # 不再用 hybrid
  rollout:
    mode: async              # (实际已是默认)
    free_cache_engine: true  # checkpoint_engine 替代
    enable_sleep_mode: true  # checkpoint_engine 替代

# ── 新增/修改 ──
actor_rollout_ref:
  model:
    path: /data/models/Qwen3.5-4B   # Qwen3.5 — 更强工具调用，async 模式正常加载（无 hybrid dummy 问题）
  rollout:
    name: vllm
    mode: async              # Agent Loop 要求 async（非 hybrid，模型从磁盘加载）
    n: 2
    temperature: 0.9
    top_p: 0.95
    prompt_length: 2048
    response_length: 4096    # 需容纳多轮对话+tool tokens
    gpu_memory_utilization: 0.45
    tensor_model_parallel_size: 1
    max_model_len: 8192
    multi_turn:
      enable: True
      max_user_turns: 3          # ← 3 轮交互
      max_assistant_turns: 6     # 每轮用户后可能有多轮 assistant
      format: qwen3_coder        # Qwen3.5 原生 XML tool call 格式: <tool_call><function=...>
      function_tool_path: pkg://trainable_openclaw.training.agent_tools
      max_tool_response_length: 1024
      tokenization_sanity_check_mode: ignore_strippable  # Qwen3 系列推理内容处理
    agent:
      num_workers: 2             # = train_batch_size
      default_agent_loop: tool_agent
    calculate_log_probs: true    # GRPO 需要 log_probs

data:
  return_raw_chat: True          # Agent Loop 要求 raw chat messages
```

### 不再需要的配置项

- `free_cache_engine` — checkpoint_engine 替代
- `enable_sleep_mode` — checkpoint_engine 替代
- `hybrid_engine` — async 模式不需要
- `kl_loss_coef` — 之前已设为 0

---

## 4. 数据格式变更

### 当前数据格式

```json
{
  "id": "retail_task_110",
  "prompt": "You name is Nancy Davis...",
  "evaluation": {...},
  "tools": [...]
}
```

### Agent Loop 要求

```json
{
  "id": "retail_task_110",
  "agent_name": "tool_agent",
  "prompt": [
    {"role": "system", "content": "You are a retail customer service agent..."},
    {"role": "user", "content": "You name is Nancy Davis..."}
  ],
  "evaluation": {...},
  "extra_info": {
    "tools_kwargs": {...}
  }
}
```

需通过 `scripts/data/add_agent_name.py` 一次转换。核心变更：
1. `prompt` 从 string → list[message]
2. 添加 `agent_name: "tool_agent"`
3. (可选) 添加 system prompt

---

## 5. Reward 函数适配

当前 `compute_score` 接收 `solution_str`（解码文本），自己解析 tool calls。

Agent Loop 输出的 batch 包含：
- `responses`: token ids，含 LLM tokens (mask=1) + tool tokens (mask=0)
- `response_mask`: 标记哪些是模型生成的 token
- `non_tensor_batch["__num_turns__"]`: 对话轮数
- `non_tensor_batch["raw_prompt"]`: 原始 prompt messages

`compute_score` 需改为：
1. 从 `extra_info` 中获取 `num_turns`、`raw_prompt`
2. 将 response_ids 解码并解析 conversation 结构
3. 对完整轨迹做 rubric 评分

或者更简单的方式：AgentLoop 的 `AgentLoopOutput.extra_fields` 可携带 trajectory 的结构化表示，由 agent_tools.py 在工具执行时记录。

**具体方案待实现时确定，核心不变：rubric 规则不变，评分逻辑不变。**

---

## 6. agent_tools.py 设计

```python
# trainable_openclaw/training/agent_tools.py
"""
@function_tool wrappers for tau-bench retail tools.
AgentLoop calls these during multi-turn rollout.
"""
from verl.tools.function_tool import function_tool
from trainable_openclaw.agent.tau_bench_tools.registry import register_tau_bench_tools
from trainable_openclaw.agent.tau_bench_tools.mock_db import MockDatabase

_db = MockDatabase("retail")

def _make_fn(tool):
    """Create a typed async function that verl can infer JSON schema from."""
    async def fn(**kwargs):
        result = _db.execute(tool, kwargs)
        return result  # dict → ToolResponse(text=json.dumps(result))
    fn.__name__ = tool.name
    fn.__doc__ = tool.description
    # 注意：verl 需要 Google-style docstring + type hints 来生成 schema
    # 如果自动推断不够精确，可用 function_tool(..., schema=...) 手动指定
    return function_tool(fn)

# 注册所有 retail 工具
_tools = register_tau_bench_tools("retail")
for t in _tools:
    globals()[t.name] = _make_fn(t)
```

**风险点**：`@function_tool` 从函数签名自动推断 JSON Schema。MockTool 的 parameters 是手写的 JSON Schema，可能比自动推断的更精确。如果自动推断不足，改用 `function_tool(schema=tool.parameters)(fn)` 直接传入。

---

## 7. 实施步骤

| 步骤 | 内容 | 验证 |
|------|------|------|
| 1 | 写 `agent_tools.py` — @function_tool 包装 | `python -c "from trainable_openclaw.training.agent_tools import *"` |
| 2 | 写 `add_agent_name.py` 转换数据 | 检查 jsonl 有 `agent_name` 字段 |
| 3 | 修改 `grpo_retail.yaml` — async + multi_turn + agent | Hydra 配置解析成功 |
| 4 | 修改 `grpo_reward.py` — 适配 AgentLoop 输出 | `test_reward.py` 通过 |
| 5 | 环境准备：确保 verl 已 `pip install -e .` | `setup_env.sh` 通过 |
| 6 | 单步训练测试（`total_training_steps: 1`） | 无 OOM、无 import 错误、reward 正常 |
| 7 | 正式训练 | rewards 曲线上升 |

---

## 8. 风险和未知

| 风险 | 影响 | 缓解 |
|------|------|------|
| Qwen3.5-4B 在 async 模式加载 | 之前 hybrid+dummy 报 IndexError | async 模式 vllm 从磁盘正常加载模型，不走 dummy |
| `qwen3_coder` format (XML) 与 Qwen3.5 实际输出不一致 | 工具调用解析失败 | Qwen3XMLToolParser 专为 Qwen3-Coder/Qwen3.5 设计，格式 `<tool_call><function=...>` |
| `@function_tool` schema 推断不精确 | 模型生成错误的工具参数 | 使用 `schema=` 参数直接传 MockTool.parameters |
| `tokenization_sanity_check` Qwen3.5 不通过 | 大量 warning | 已设 `ignore_strippable`（Qwen3 系列会移除历史推理内容） |
| 多轮 response_length 不够 (当前 4096) | 截断 | 增大到 8192 或更高 |
| Agent Loop 不走我们的 reward 函数 | reward 始终为 0 | 检查 `extra_info` 传递链路 |
| LoRA 在 async 模式下不兼容 | 训练失败 | verl 文档说 async 支持 LoRA，需实测 |

---

## 9. 与旧方案的关系

```
                    ┌──────────────────────────┐
                    │  agent/tau_bench_tools/  │  ← 核心工具（唯一真相源）
                    └──────────┬───────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   ┌──────▼──────┐   ┌────────▼────────┐   ┌───────▼──────┐
   │ nanobot     │   │ Agent Loop      │   │ rollout_env  │
   │ (serving)   │   │ (training)      │   │ (legacy)     │
   │ 84行 glue   │   │ ~50行 glue      │   │ 50行 glue    │
   └─────────────┘   └─────────────────┘   └──────────────┘
        保留              新增（本plan）         保留（fallback）
```
