"""
Base framework for tau-bench mock tools.

Provides the MockTool abstract base class and three shared utility tools
(calculate, think, transfer_to_human_agents) used by both the airline and
retail domains.
"""

from __future__ import annotations

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MockTool:
    """Base class for tau-bench mock tools.

    Subclasses must define ``name``, ``description``, and ``parameters``
    (a JSON Schema dict for the tool arguments).  They override ``execute()``
    to implement domain-specific logic against the in-memory ``db_state`` dict.

    Usage::

        class FindUser(MockTool):
            name = "find_user_id_by_email"
            description = "Find a user by email address."
            parameters = {
                "type": "object",
                "properties": {"email": {"type": "string", "description": "..."}},
                "required": ["email"],
            }
            def execute(self, arguments, db_state):
                ...
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    def to_schema(self) -> dict[str, Any]:
        """Return an OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool against the mock database state.

        Args:
            arguments: Dict of parameter values matching the JSON Schema.
            db_state: The mutable in-memory database state (dict).

        Returns:
            A JSON-serializable dict, typically ``{"status": "success", "result": ...}``
            or ``{"status": "error", "message": "..."}``.
        """
        raise NotImplementedError

    def _validate_args(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Check required args are present. Returns error dict or None."""
        required: list[str] = self.parameters.get("required", [])
        for arg in required:
            if arg not in arguments:
                return {"status": "error", "message": f"Missing required argument: {arg}"}
        return None


# ---------------------------------------------------------------------------
# Shared utility tools (domain-independent)
# ---------------------------------------------------------------------------


class _CalculateTool(MockTool):
    name = "calculate"
    description = (
        "Evaluate a mathematical expression. "
        "Supports +, -, *, /, parentheses, and decimal numbers."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate, e.g. '2+3*4'.",
            }
        },
        "required": ["expression"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        expr = arguments["expression"].strip()
        # Sanitize: only digits, operators, parens, dot, whitespace
        if not re.fullmatch(r"[\d+\-*/().%\s]+", expr):
            return {"status": "error", "message": f"Invalid expression: {expr}"}
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return {"status": "success", "result": str(result)}
        except Exception as e:
            return {"status": "error", "message": f"Calculation error: {e}"}


class _ThinkTool(MockTool):
    name = "think"
    description = (
        "Use this tool to think through a problem step by step. "
        "The tool does nothing — it simply records your thought process."
    )
    parameters = {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Your internal reasoning or thought process.",
            }
        },
        "required": ["thought"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        # No-op: simply acknowledge
        return {"status": "success", "result": "Thought recorded."}


class _TransferToHumanTool(MockTool):
    name = "transfer_to_human_agents"
    description = (
        "Transfer the conversation to a human agent. "
        "Use this when you cannot resolve the issue or when the user "
        "explicitly requests a human. Provide a summary of the situation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Summary of the issue, actions taken, and what the human agent needs to do.",
            }
        },
        "required": ["summary"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "success",
            "result": {
                "message": f"Transferred to human agent. Summary: {arguments.get('summary', '')}",
            },
        }


# Module-level singletons — import these for use in domain modules
calculate_tool = _CalculateTool()
think_tool = _ThinkTool()
transfer_to_human_agents_tool = _TransferToHumanTool()
