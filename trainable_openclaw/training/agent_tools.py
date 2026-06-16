"""
@function_tool wrappers for tau-bench retail tools.

AgentLoop calls these during multi-turn rollout — tool execution happens
inside the rollout phase while vllm is awake.  Each tool is a thin wrapper
around the shared MockTool registry, registered via verl's @function_tool
decorator with explicit JSON Schema from the MockTool definition.

Usage in config:
    actor_rollout_ref.rollout.multi_turn.function_tool_path: trainable_openclaw/training/agent_tools.py
"""

from __future__ import annotations

from verl.tools.function_tool import function_tool
from trainable_openclaw.agent.tau_bench_tools.registry import register_tau_bench_tools
from trainable_openclaw.agent.tau_bench_tools.mock_db import MockDatabase

_db = MockDatabase("retail")


def _make_fn(tool):
    """Create a sync callable that executes the MockTool against the shared DB."""

    def fn(**kwargs):
        result = _db.execute(tool, kwargs)
        return result  # dict → verl serialises to ToolResponse(text=json.dumps(result))

    fn.__name__ = tool.name
    fn.__doc__ = tool.description
    return fn


# Wrap each MockTool as a @function_tool with explicit schema (no inference).
_mock_tools = register_tau_bench_tools("retail")
for _t in _mock_tools:
    _schema = {
        "type": "function",
        "function": {
            "name": _t.name,
            "description": _t.description,
            "parameters": _t.parameters,
        },
    }
    _fn = _make_fn(_t)
    # Register as module-level callable so function_tool picks it up
    globals()[_t.name] = function_tool(name=_t.name, schema=_schema)(_fn)
