"""
Nanobot Tool classes for tau-bench retail domain.

Wraps the 14 retail MockTools + 3 shared utilities as nanobot Tool subclasses
sharing a module-level MockDatabase.  Drop this file into nanobot's tools
directory so the built-in package scanner discovers and registers them.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from nanobot.agent.tools.base import Tool

logger = logging.getLogger(__name__)

from trainable_openclaw.agent.tau_bench_tools.registry import register_tau_bench_tools
from trainable_openclaw.agent.tau_bench_tools.mock_db import MockDatabase

# ── Shared database state (module-level singleton) ──────────────────────────

_db: MockDatabase | None = None
_db_tools: dict[str, Any] = {}
_initialized: bool = False


def _init_db():
    global _db, _db_tools, _initialized
    if _initialized:
        return
    _db = MockDatabase("retail")
    _db_tools = {t.name: t for t in register_tau_bench_tools("retail")}
    _initialized = True


def _execute_tool(name: str, kwargs: dict[str, Any]) -> str:
    _init_db()
    tool = _db_tools.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = tool.execute(kwargs, _db.state)
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Dynamically create one Tool subclass per tau-bench tool ─────────────────

def _make_tool_cls(mock_tool: Any) -> type[Tool]:
    """Create a nanobot Tool subclass wrapping a MockTool instance."""

    _name = mock_tool.name
    _description = mock_tool.description
    _parameters = mock_tool.parameters

    class _TauBenchTool(Tool):
        name: str = _name
        description: str = _description
        parameters: dict[str, Any] = _parameters

        async def execute(self, **kwargs: Any) -> Any:
            return _execute_tool(_name, kwargs)

    # Generate a valid Python class name from the tool name
    cls_name = "".join(part.capitalize() for part in _name.split("_")) + "Tool"
    _TauBenchTool.__name__ = cls_name
    _TauBenchTool.__qualname__ = cls_name
    return _TauBenchTool


# Instantiate all tool classes at module-import time so pkgutil discovers them
_init_db()
for _tool in _db_tools.values():
    _cls = _make_tool_cls(_tool)
    # Register as module-level name so dir(module) finds it for auto-discovery
    setattr(sys.modules[__name__], _cls.__name__, _cls)

# Count tools exposed (for startup logging)
_RETAIL_TOOL_COUNT = len(_db_tools)
