"""
Tool registry for tau-bench mock tools.

Provides ``register_tau_bench_tools()`` which returns a list of all
MockTool instances for a given scenario ("retail" or "airline").

Usage::

    from trainable_openclaw.agent.tau_bench_tools.registry import register_tau_bench_tools

    tools = register_tau_bench_tools("retail")
    schemas = [t.to_schema() for t in tools]

    # Pass schemas to nanobot's tool registry
"""

from __future__ import annotations

import logging
from typing import Any

from trainable_openclaw.agent.tau_bench_tools.base import MockTool
from trainable_openclaw.agent.tau_bench_tools.retail import retail_tools
from trainable_openclaw.agent.tau_bench_tools.airline import airline_tools

logger = logging.getLogger(__name__)


def register_tau_bench_tools(scenario: str = "retail") -> list[MockTool]:
    """Return all MockTool instances for the given scenario.

    Args:
        scenario: ``"retail"`` (17 tools) or ``"airline"`` (15 tools).

    Returns:
        A list of ``MockTool`` instances ready for use with ``MockDatabase.execute()``.

    Raises:
        ValueError: If scenario is not recognised.
    """
    scenario = scenario.strip().lower()
    if scenario == "retail":
        return list(retail_tools)
    elif scenario == "airline":
        return list(airline_tools)
    else:
        raise ValueError(f"Unknown scenario: {scenario!r}. Expected 'retail' or 'airline'.")
