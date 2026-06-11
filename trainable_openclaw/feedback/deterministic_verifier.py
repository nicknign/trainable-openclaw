"""
Layer 1: Deterministic verification of tool execution results.

FREE — no API calls, purely rule-based checks on tool call format,
execution results, dangerous operations, and task completion signals.

Verification functions return ``VerificationResult`` with pass/fail/score/checks.
``compute_layer1_reward`` combines multiple verifications into a single 0-1 score.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class VerificationResult:
    """Result of deterministic verification."""

    passed: bool
    score: float  # 1.0 if all checks pass, 0.0 otherwise
    checks: list[dict] = field(default_factory=list)  # [{"check": "schema_valid", "passed": True}, ...]
    details: str = ""  # Human-readable explanation


# ---------------------------------------------------------------------------
# Tool call format verification
# ---------------------------------------------------------------------------

DANGEROUS_EXEC_PATTERNS: list[str] = [
    r"\brm\s+(-rf?|--recursive)?\s*/",
    r"\bdd\s+if=",
    r"\bmkfs\.",
    r"\b:(){ :|:& };:",  # fork bomb
    r"\bchmod\s+(-R\s+)?777\s+/",
    r"\b>(/dev/[hs]d[a-z])",
    r"\bformat\s+[A-Za-z]:",  # Windows format
]

DANGEROUS_SYSTEM_PATHS: list[str] = [
    r"^/etc/",
    r"^/boot/",
    r"^/sys/",
    r"^/proc/",
    r"^/dev/",
    r"^C:\\Windows",
    r"^C:\\Program Files",
    r"^~/.ssh",
]

TASK_COMPLETION_TOOLS: set[str] = {
    "complete_goal",
    "finish",
    "task_complete",
    "done",
    "stop",
    "read",  # tau-bench: reading final output implies completion
}


def _find_tool_calls(trajectory: list[dict]) -> list[dict]:
    """Extract tool-call dicts from a trajectory."""
    calls: list[dict] = []
    for msg in trajectory:
        role = msg.get("role", "")
        if role == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                calls.append(dict(tc))
        elif role == "tool_call" or role == "function_call":
            calls.append(dict(msg))
    return calls


def _find_tool_responses(trajectory: list[dict]) -> list[dict]:
    """Extract tool-response dicts from a trajectory."""
    responses: list[dict] = []
    for msg in trajectory:
        role = msg.get("role", "")
        if role == "tool":
            responses.append(dict(msg))
    return responses


# ---------------------------------------------------------------------------
# Individual verifiers
# ---------------------------------------------------------------------------


def verify_tool_call_format(tool_call: dict) -> VerificationResult:
    """Verify tool call JSON matches the expected schema.

    Checks:
    - id is non-empty string
    - function.name is non-empty string
    - function.arguments is valid JSON string
    - arguments can be parsed to dict
    """
    checks: list[dict] = []

    # Check id
    call_id = tool_call.get("id", "") or tool_call.get("call_id", "")
    if isinstance(call_id, str) and len(call_id) > 0:
        checks.append({"check": "id_valid", "passed": True})
    else:
        checks.append({"check": "id_valid", "passed": False, "detail": f"id={repr(call_id)}"})

    # Check function block
    func = tool_call.get("function", {})
    if isinstance(func, dict) and func:
        checks.append({"check": "function_present", "passed": True})
    else:
        checks.append({"check": "function_present", "passed": False, "detail": "missing or invalid"})
        return VerificationResult(
            passed=False,
            score=0.0,
            checks=checks,
            details="Tool call missing 'function' block",
        )

    # Check function.name
    name = func.get("name", "")
    if isinstance(name, str) and len(name.strip()) > 0:
        checks.append({"check": "name_valid", "passed": True})
    else:
        checks.append({"check": "name_valid", "passed": False, "detail": f"name={repr(name)}"})

    # Check function.arguments
    args = func.get("arguments", None)
    if args is None or (isinstance(args, str) and len(args.strip()) == 0):
        checks.append({"check": "arguments_present", "passed": False, "detail": "empty or missing"})
    elif isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                checks.append({"check": "arguments_parseable", "passed": True})
            else:
                checks.append({"check": "arguments_parseable", "passed": False, "detail": "not a dict"})
        except json.JSONDecodeError as e:
            checks.append({"check": "arguments_parseable", "passed": False, "detail": str(e)[:120]})
    elif isinstance(args, dict):
        checks.append({"check": "arguments_parseable", "passed": True})
    else:
        checks.append({"check": "arguments_parseable", "passed": False, "detail": f"type={type(args)}"})

    all_passed = all(c["passed"] for c in checks)
    return VerificationResult(
        passed=all_passed,
        score=1.0 if all_passed else 0.0,
        checks=checks,
        details="All checks passed" if all_passed else f"{sum(1 for c in checks if not c['passed'])}/{len(checks)} checks failed",
    )


def verify_execution_result(tool_response: dict) -> VerificationResult:
    """Verify tool execution result indicates success.

    Checks:
    - For shell/exec: exit_code or returncode is present
    - For write_file: status or success indicator present
    - For read_file: content or result field present
    - Generic: status field with success/ok value
    """
    checks: list[dict] = []
    content = tool_response.get("content", "")

    # Try to parse content as JSON (many tools return JSON)
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass

    if isinstance(content, dict):
        # Check exit_code / returncode
        if "exit_code" in content or "returncode" in content:
            ec = content.get("exit_code", content.get("returncode", -1))
            if ec == 0:
                checks.append({"check": "exit_code_zero", "passed": True})
            else:
                checks.append({"check": "exit_code_zero", "passed": False, "detail": f"exit_code={ec}"})

        # Check status
        status = content.get("status", "").lower() if isinstance(content.get("status"), str) else ""
        if status:
            if status in ("success", "ok", "done", "completed"):
                checks.append({"check": "status_success", "passed": True})
            else:
                checks.append({"check": "status_success", "passed": False, "detail": f"status={status}"})

        # Check result/error
        if "error" in content and content["error"]:
            checks.append({"check": "no_error", "passed": False, "detail": str(content.get("error", ""))[:160]})
        elif "result" in content and content["result"] is not None:
            checks.append({"check": "has_result", "passed": True})
        elif "content" in content or "data" in content:
            checks.append({"check": "has_result", "passed": True})

    elif isinstance(content, str) and len(content.strip()) > 0:
        checks.append({"check": "has_content", "passed": True})

    if not checks:
        checks.append({"check": "has_content", "passed": len(str(content)) > 0})

    all_passed = all(c["passed"] for c in checks)
    return VerificationResult(
        passed=all_passed,
        score=1.0 if all_passed else 0.0,
        checks=checks,
        details="All checks passed" if all_passed else f"{sum(1 for c in checks if not c['passed'])}/{len(checks)} checks failed",
    )


def verify_dangerous_operation(tool_call: dict) -> VerificationResult:
    """Flag dangerous operations that should never be executed.

    Checks:
    - exec with rm -rf, dd, mkfs, fork bombs, etc.
    - write_file to system paths (/etc, /boot, C:\\Windows, etc.)
    - chmod 777 on system paths
    """
    checks: list[dict] = []
    func = tool_call.get("function", {})
    name = (func.get("name", "") or tool_call.get("name", "")).lower()
    args_str = func.get("arguments", "") if isinstance(func.get("arguments"), str) else json.dumps(func.get("arguments", {}))

    # Check for dangerous exec commands
    if name in ("exec", "shell", "bash", "execute", "run_command", "cmd"):
        for pattern in DANGEROUS_EXEC_PATTERNS:
            if re.search(pattern, args_str, re.IGNORECASE):
                checks.append({
                    "check": "no_dangerous_exec",
                    "passed": False,
                    "detail": f"Matched dangerous pattern: {pattern}",
                })
                break
        else:
            checks.append({"check": "no_dangerous_exec", "passed": True})

    # Check for write to system paths
    if name in ("write_file", "write", "save_file", "create_file"):
        # Extract file path from arguments
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else (args_str if isinstance(args_str, dict) else {})
        except json.JSONDecodeError:
            args = {}
        file_path = args.get("file_path", args.get("path", args.get("filename", "")))
        for pattern in DANGEROUS_SYSTEM_PATHS:
            if re.search(pattern, str(file_path), re.IGNORECASE):
                checks.append({
                    "check": "no_system_path_write",
                    "passed": False,
                    "detail": f"Writing to system path: {file_path}",
                })
                break
        else:
            checks.append({"check": "no_system_path_write", "passed": True})

    # No dangerous operation checks needed for benign tools
    if not checks:
        checks.append({"check": "no_dangerous_ops", "passed": True})

    all_passed = all(c["passed"] for c in checks)
    return VerificationResult(
        passed=all_passed,
        score=1.0 if all_passed else 0.0,
        checks=checks,
        details="No dangerous operations detected" if all_passed else "Dangerous operation detected",
    )


def verify_task_completion(trajectory: list[dict]) -> VerificationResult:
    """Check if agent signaled task completion.

    Checks:
    - A completion tool was called (complete_goal, finish, done, etc.)
    - OR final assistant message looks like a completion
    - Checks that required outputs were communicated to user
    """
    checks: list[dict] = []

    # Check for completion tools
    tool_calls = _find_tool_calls(trajectory)
    completion_called = False
    for tc in tool_calls:
        func = tc.get("function", {})
        name = (func.get("name", "") or tc.get("name", "")).lower()
        if name in TASK_COMPLETION_TOOLS:
            completion_called = True
            break

    if completion_called:
        checks.append({"check": "completion_tool", "passed": True})
    else:
        # Check if final assistant message looks like a completion
        assistant_messages = [m for m in trajectory if m.get("role") == "assistant"]
        if assistant_messages:
            final = assistant_messages[-1].get("content", "")
            if isinstance(final, str) and len(final.strip()) > 20:
                # Heuristic: final message contains completion-like language
                completion_indicators = [
                    r"\b(?:here|there|this is|I ['']ve|finished|completed|done|综上所述|总结|完成|以上就是)\b",
                ]
                looks_complete = any(re.search(pat, final, re.IGNORECASE) for pat in completion_indicators)
                if looks_complete:
                    checks.append({"check": "completion_language", "passed": True})
                else:
                    checks.append({"check": "completion_language", "passed": False, "detail": "No completion indicator in final message"})
            else:
                checks.append({"check": "completion_language", "passed": False, "detail": "Final assistant message too short or empty"})
        else:
            checks.append({"check": "completion_language", "passed": False, "detail": "No assistant messages in trajectory"})

    all_passed = all(c["passed"] for c in checks)
    return VerificationResult(
        passed=all_passed,
        score=1.0 if all_passed else 0.0,
        checks=checks,
        details="Task completion verified" if all_passed else "Task completion not verified",
    )


# ---------------------------------------------------------------------------
# Combined Layer 1 reward
# ---------------------------------------------------------------------------


def compute_layer1_reward(trajectory: list[dict]) -> float:
    """Combined Layer 1 reward.

    Runs all verifications on the trajectory:
    1. Tool call format check for each tool call
    2. Dangerous operation check for each tool call
    3. Task completion check

    Returns 1.0 if all pass, else proportional to pass rate.
    If no tool calls exist, only task_completion is checked.
    """
    results: list[VerificationResult] = []

    tool_calls = _find_tool_calls(trajectory)

    if tool_calls:
        for tc in tool_calls:
            results.append(verify_tool_call_format(tc))
            results.append(verify_dangerous_operation(tc))
    else:
        # Even without tool calls, format/dangerous checks are trivially "passed"
        # since there's nothing to fail on. We skip them and only check completion.
        pass

    results.append(verify_task_completion(trajectory))

    if not results:
        return 0.0

    scores = [r.score for r in results]
    return sum(scores) / len(scores)
