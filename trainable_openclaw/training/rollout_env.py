"""
Multi-turn rollout environment for tau-bench retail agent training.

Provides ToolExecutor (local mock DB tool execution), RuleSimulatedUser
(deterministic user simulator based on nl_assertions), and TauBenchRolloutEnv
(multi-turn rollout orchestration).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from trainable_openclaw.agent.tau_bench_tools.mock_db import MockDatabase, seed
from trainable_openclaw.agent.tau_bench_tools.registry import register_tau_bench_tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------


class ToolExecutor:
    """Execute tau-bench tool calls against an in-memory mock database.

    Loads tool definitions from the project's MockTool registry and executes
    them locally — no network calls, no nanobot API.
    """

    def __init__(self, scenario: str = "retail"):
        self._scenario = scenario
        self._db = MockDatabase(scenario)
        self._tools: dict[str, Any] = {t.name: t for t in register_tau_bench_tools(scenario)}

    def execute(self, tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool call against the mock DB.

        Returns a dict with at least ``{"status": "success"}`` or
        ``{"status": "error", "message": "..."}``.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
        try:
            result = self._db.execute(tool, kwargs)
            return result
        except Exception as exc:
            logger.warning("Tool %s failed: %s", tool_name, exc)
            return {"status": "error", "message": str(exc)}

    def reset(self) -> None:
        """Reset the mock DB to a fresh state."""
        self._db = MockDatabase(self._scenario)


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------

_TOOL_CALL_PATTERN = re.compile(
    r'<function_call>\s*(.+?)\s*</function_call>',
    re.DOTALL,
)

_JSON_OBJECT_PATTERN = re.compile(
    r'\{[^{}]*"name"\s*:\s*"[^"]+"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}',
    re.DOTALL,
)


def parse_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    """Extract tool call dicts from model output text.

    Handles:
    - <function_call>{"name": "...", "arguments": {...}}</function_call>
    - Raw JSON objects containing "name" and "arguments" keys
    - OpenAI-style: {"function": {"name": "...", "arguments": "..."}}

    Returns list of dicts with ``name`` and ``arguments`` keys.
    """
    if not text:
        return []

    calls: list[dict[str, Any]] = []

    # Try <function_call> wrapper first
    for match in _TOOL_CALL_PATTERN.finditer(text):
        try:
            obj = json.loads(match.group(1).strip())
            calls.append(_normalize_tool_call(obj))
        except json.JSONDecodeError:
            continue

    if calls:
        return calls

    # Try plain JSON objects with name+arguments
    for match in _JSON_OBJECT_PATTERN.finditer(text):
        try:
            obj = json.loads(match.group())
            if "name" in obj and ("arguments" in obj or "function" in obj):
                calls.append(_normalize_tool_call(obj))
        except json.JSONDecodeError:
            continue

    # Try finding any JSON object with a "function" key (OpenAI style)
    if not calls:
        openai_calls = _parse_openai_style_tool_calls(text)
        calls.extend(openai_calls)

    return calls


def _normalize_tool_call(obj: dict) -> dict:
    """Normalize various tool call formats to {'name': str, 'arguments': dict}."""
    if "function" in obj:
        func = obj["function"]
        name = func.get("name", "")
        args = func.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return {"name": name, "arguments": args}
    return {"name": obj.get("name", ""), "arguments": obj.get("arguments", {})}


def _parse_openai_style_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse OpenAI-style tool calls with string-encoded arguments.

    Uses brace-counter to find complete JSON objects containing a 'function' key.
    """
    calls: list[dict[str, Any]] = []

    idx = 0
    while idx < len(text):
        func_pos = text.find('"function"', idx)
        if func_pos < 0:
            break

        # Search backwards for the opening brace
        brace_start = text.rfind('{', 0, func_pos)
        if brace_start < 0:
            idx = func_pos + len('"function"')
            continue

        # Brace-counter to find the matching closing brace
        depth = 0
        brace_end = brace_start
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    brace_end = i
                    break

        if brace_end <= brace_start:
            idx = func_pos + len('"function"')
            continue

        candidate = text[brace_start:brace_end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and ("function" in obj or "name" in obj):
                calls.append(_normalize_tool_call(obj))
        except json.JSONDecodeError:
            pass

        idx = brace_end + 1

    return calls


# ---------------------------------------------------------------------------
# RuleSimulatedUser
# ---------------------------------------------------------------------------

_RESULT_MAP = {
    "success": "success",
    "完成": "success",
    "ok": "success",
}


@dataclass
class UserResponse:
    message: str
    status: str        # "continue" | "complete" | "give_up"
    satisfaction: float  # 0.0 - 1.0


class RuleSimulatedUser:
    """Deterministic user simulator based on task nl_assertions.

    No LLM calls — purely rule-based. Checks tool execution results against
    the task's evaluation criteria to determine when the task is complete.
    """

    def __init__(self, task: dict):
        self._task = task
        self._assertions = self._extract_assertions()
        self._assertions_met: set[str] = set()
        self._round_count = 0
        self._no_progress_rounds = 0

    # ------------------------------------------------------------------
    # Identity extraction (mirrors simulated_user.py logic)
    # ------------------------------------------------------------------

    @property
    def initial_message(self) -> str:
        prompt = self._task.get("prompt", "")
        lines = [l.strip() for l in prompt.split("\n") if l.strip()]

        name_match = re.search(
            r"(?:You are|(?:You|Your) name is)\s+(.+?)(?:\s+(?:and|with|your|whose)\s|[,.]\s|$)",
            lines[0],
        )
        user_name = name_match.group(1).strip().rstrip(",") if name_match else ""
        user_id = ""
        email = ""
        zip_code = ""
        for line in lines[:2]:
            uid = re.search(r"user id is\s+(\S+)", line, re.IGNORECASE)
            if uid:
                user_id = uid.group(1).rstrip(".,")
            em = re.search(r"email\s+(?:address\s+)?is\s+(\S+@\S+)", line, re.IGNORECASE)
            if em:
                email = em.group(1).rstrip(".,")
            zc = re.search(r"zip\s*code\s+(?:is\s+)?(\d{5})", line, re.IGNORECASE)
            if not zc:
                zc = re.search(r"zipcode\s+(\d{5})", line, re.IGNORECASE)
            if zc:
                zip_code = zc.group(1)

        identity_parts = []
        if user_name:
            identity_parts.append(f"My name is {user_name}")
        if user_id:
            identity_parts.append(f"my user ID is {user_id}")
        elif email:
            identity_parts.append(f"my email is {email}")
        if zip_code:
            identity_parts.append(f"my zip code is {zip_code}")
        identity = ". ".join(identity_parts)
        if identity:
            identity = identity[0].upper() + identity[1:] + "."

        want_lines = [l for l in lines if "want to" in l.lower() or "need to" in l.lower() or "wish to" in l.lower()]
        if want_lines:
            request = re.sub(r"^(You|you)\s", "I ", want_lines[0])
        else:
            task_lines = [l for l in lines if not l.lower().startswith("you are")
                          and "user id" not in l.lower()
                          and "email" not in l.lower()
                          and "zip code" not in l.lower()
                          and not l.lower().startswith("your name")
                          and not l.lower().startswith("you name")]
            request = " ".join(task_lines) if task_lines else "I need some help with my order."
            request = re.sub(r"^(You|you)\s", "I ", request)

        if identity:
            return f"Hi there! {identity} {request}"
        return f"Hi there! {request}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def respond(
        self,
        agent_msg: str,
        round_num: int,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> UserResponse:
        self._round_count = round_num

        tool_results = tool_results or []
        self._update_assertions(tool_results, agent_msg)

        if self._all_assertions_met():
            return UserResponse(
                message=self._build_completion_message(),
                status="complete",
                satisfaction=1.0,
            )

        if self._should_give_up():
            return UserResponse(
                message=self._build_give_up_message(),
                status="give_up",
                satisfaction=0.0,
            )

        return UserResponse(
            message=self._build_continue_message(agent_msg, tool_results),
            status="continue",
            satisfaction=self._compute_partial_satisfaction(),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_assertions(self) -> list[str]:
        ev = self._task.get("evaluation", {})
        if isinstance(ev, str):
            try:
                ev = json.loads(ev)
            except (json.JSONDecodeError, TypeError):
                ev = {}
        assertions = ev.get("nl_assertions", []) if isinstance(ev, dict) else []
        return [str(a) for a in assertions]

    def _update_assertions(self, tool_results: list[dict], agent_msg: str) -> None:
        for tr in tool_results:
            result = tr.get("result", tr)
            status = result.get("status", "") if isinstance(result, dict) else ""
            if status in ("success", "ok", "done"):
                matched = self._match_assertion_to_tool(tr.get("name", ""))
                if matched:
                    self._assertions_met.add(matched)

        # Check agent message for completion indicators — only mark assertions
        # that are explicitly addressed in the message
        completion_indicators = [
            r"all set\b", r"resolved\b", r"done\b", r"completed\b",
        ]
        if any(re.search(p, agent_msg, re.IGNORECASE) for p in completion_indicators):
            for a in self._assertions:
                if a not in self._assertions_met and self._check_assertion_against_message(a, agent_msg):
                    self._assertions_met.add(a)

    def _match_assertion_to_tool(self, tool_name: str) -> str | None:
        tool_lower = tool_name.lower()
        for a in self._assertions:
            a_lower = a.lower()
            if "cancel" in a_lower and "cancel" in tool_lower:
                return a
            if "return" in a_lower and "return" in tool_lower:
                return a
            if "exchange" in a_lower and "exchange" in tool_lower:
                return a
            if "modify" in a_lower and "modify" in tool_lower:
                return a
            if ("look up" in a_lower or "find" in a_lower or "identity" in a_lower or "user" in a_lower) and "find_user" in tool_lower:
                return a
            if ("order" in a_lower and "information" in a_lower) and "get_order" in tool_lower:
                return a
            if ("order" in a_lower and "look" in a_lower) and ("get_order" in tool_lower or "find" in tool_lower):
                return a
        return None

    def _check_assertion_against_result(self, assertion: str, tool_result: dict) -> bool:
        """Check if a tool result satisfies an assertion."""
        name = tool_result.get("name", "")
        result = tool_result.get("result", {})
        if not isinstance(result, dict):
            return False

        # Map assertions to tool names
        if "cancel" in assertion.lower() and "cancel" in name.lower():
            return result.get("status") in ("success", "ok")
        if "return" in assertion.lower() and "return" in name.lower():
            return result.get("status") in ("success", "ok")
        if "exchange" in assertion.lower() and "exchange" in name.lower():
            return result.get("status") in ("success", "ok")
        if "modify" in assertion.lower() and "modify" in name.lower():
            return result.get("status") in ("success", "ok")
        if "order information" in assertion.lower() or "order" in assertion.lower():
            if "get_order" in name.lower() or "find" in name.lower():
                return result.get("status") in ("success", "ok")
        if "user" in assertion.lower() and "find" in name.lower():
            return result.get("status") in ("success", "ok")

        return False

    def _check_assertion_against_message(self, assertion: str, msg: str) -> bool:
        """Check if agent message addresses an assertion."""
        msg_lower = msg.lower()
        if "cancel" in assertion.lower():
            return "cancell" in msg_lower or "cancel" in msg_lower
        if "return" in assertion.lower():
            return "return" in msg_lower
        if "exchange" in assertion.lower():
            return "exchange" in msg_lower
        if "refund" in assertion.lower():
            return "refund" in msg_lower
        if "tracking" in assertion.lower():
            return "tracking" in msg_lower
        return False

    def _all_assertions_met(self) -> bool:
        if not self._assertions:
            return self._round_count >= 3
        return len(self._assertions_met) >= len(self._assertions)

    def _should_give_up(self) -> bool:
        if self._round_count >= 8:
            return True
        if self._no_progress_rounds >= 3:
            return True
        return False

    def _compute_partial_satisfaction(self) -> float:
        if not self._assertions:
            return 0.5
        met = len(self._assertions_met)
        total = len(self._assertions)
        return min(0.9, met / max(total, 1))

    def _build_completion_message(self) -> str:
        return "Thank you! That's exactly what I needed. Have a great day!"

    def _build_give_up_message(self) -> str:
        return "I don't think you're understanding what I need. I'll try contacting support another way."

    def _build_continue_message(self, agent_msg: str, tool_results: list[dict]) -> str:
        unmet = [a for a in self._assertions if a not in self._assertions_met]
        if unmet and self._round_count <= 3:
            remaining = "; ".join(unmet[:2])
            return f"I still need help with: {remaining}"
        if not tool_results:
            self._no_progress_rounds += 1
            return "Have you looked into my request yet? I need you to check my account or orders."
        self._no_progress_rounds = 0
        return "Okay, and what about the rest of my request?"


# ---------------------------------------------------------------------------
# TauBenchRolloutEnv
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    prompt: dict
    conversations: list[dict] = field(default_factory=list)
    rounds: int = 0
    status: str = "in_progress"
    satisfaction: float = 0.0


class TauBenchRolloutEnv:
    """Multi-turn rollout environment for tau-bench retail tasks.

    Executes full conversations: model generates, tools execute, simulated user
    responds, loop until completion or max turns.
    """

    def __init__(self, prompts: list[dict], max_turns: int = 10, scenario: str = "retail"):
        self._prompts = prompts
        self._max_turns = max_turns
        self._scenario = scenario
        self._executor = ToolExecutor(scenario)

    def rollout_one(self, prompt: dict, model_fn: Callable) -> Trajectory:
        """Run a single full multi-turn rollout.

        Args:
            prompt: Task dict from train_prompts_augmented.jsonl.
            model_fn: Callable that takes a list of messages (conversation history)
                and returns a string (the model's response).
        """
        self._executor.reset()
        sim_user = RuleSimulatedUser(prompt)
        conversation: list[dict] = []
        user_msg = sim_user.initial_message

        for turn in range(1, self._max_turns + 1):
            conversation.append({"role": "user", "content": user_msg})

            model_output = model_fn(conversation)
            conversation.append({"role": "assistant", "content": model_output})

            tool_calls = parse_tool_calls_from_text(model_output)
            tool_results: list[dict] = []
            for tc in tool_calls:
                result = self._executor.execute(tc["name"], tc["arguments"])
                tool_results.append({"name": tc["name"], "result": result})

            if tool_results:
                conversation.append({"role": "tool", "content": json.dumps(tool_results, default=str)})

            response = sim_user.respond(model_output, turn, tool_results)

            if response.status == "complete":
                return Trajectory(
                    prompt=prompt,
                    conversations=conversation,
                    rounds=turn,
                    status="complete",
                    satisfaction=response.satisfaction,
                )
            elif response.status == "give_up":
                return Trajectory(
                    prompt=prompt,
                    conversations=conversation,
                    rounds=turn,
                    status="give_up",
                    satisfaction=response.satisfaction,
                )

            user_msg = response.message

        return Trajectory(
            prompt=prompt,
            conversations=conversation,
            rounds=self._max_turns,
            status="timeout",
            satisfaction=0.0,
        )

    def rollout_batch(self, prompts: list[dict], model_fn: Callable) -> list[Trajectory]:
        """Run rollouts for a batch of prompts."""
        results: list[Trajectory] = []
        for prompt in prompts:
            traj = self.rollout_one(prompt, model_fn)
            results.append(traj)
        return results
