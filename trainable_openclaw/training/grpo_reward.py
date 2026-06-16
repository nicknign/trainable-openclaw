"""
GRPO reward function — three inference modes via REWARD_MODE env var.

    REWARD_MODE=agent (default): score verl Agent Loop multi-turn trajectory
        directly from solution_str. Zero API cost. Tools execute inside
        Agent Loop during rollout.

    REWARD_MODE=direct: score verl's rollout output directly (single-turn
        tool execution + rubric). No nanobot dependency.

    REWARD_MODE=nanobot: call nanobot for multi-turn agent loop, then score
        with rubric rules. Falls back to single-turn if nanobot unreachable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any

import requests

logger = logging.getLogger(__name__)

NANOBOT_URL = os.environ.get("NANOBOT_URL", "http://localhost:8900")
NANOBOT_TIMEOUT = int(os.environ.get("NANOBOT_TIMEOUT", "120"))
REWARD_MODE = os.environ.get("REWARD_MODE", "agent")  # "agent", "direct", or "nanobot"

_CACHE: dict[str, Any] = {}

# Agent Loop trajectory markers
_AGENT_TOOL_RESPONSE_RE = re.compile(
    r'<tool_response>\s*(.*?)\s*</tool_response>',
    re.DOTALL,
)


def _cached(key: str, factory, *args):
    if key not in _CACHE:
        _CACHE[key] = factory(*args)
    return _CACHE[key]


def _get_executor() -> Any:
    from trainable_openclaw.training.rollout_env import ToolExecutor
    return _cached("executor", ToolExecutor, "retail")


def _get_rubric_engine() -> Any:
    from trainable_openclaw.training.rubric_rules import RubricRuleEngine
    return _cached("rubric_engine", RubricRuleEngine)


def _extract_user_message(extra_info: dict | None, ground_truth: str) -> str:
    """Extract user message from extra_info or ground_truth.

    Handles both old string-prompt and new list-prompt (Agent Loop) formats.
    """
    if extra_info:
        # Agent Loop format: prompt is a list of messages
        prompt = extra_info.get("prompt", "")
        if isinstance(prompt, list):
            for msg in prompt:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if content:
                        return content
        elif isinstance(prompt, str) and prompt.strip():
            return prompt

        # Try raw_prompt from Agent Loop non_tensor_batch
        raw = extra_info.get("raw_prompt", "")
        if isinstance(raw, list):
            for msg in raw:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if content:
                        return content
        elif isinstance(raw, str) and raw.strip():
            return raw

    # Fall back to ground_truth (old format)
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        if isinstance(gt, dict):
            prompt = gt.get("prompt", "")
            if isinstance(prompt, list):
                for msg in prompt:
                    if msg.get("role") == "user":
                        return msg.get("content", "")
            elif isinstance(prompt, str) and prompt.strip():
                return prompt
    except (json.JSONDecodeError, TypeError):
        pass
    return "I need help with my order."


# ---------------------------------------------------------------------------
# Nanobot interface (unchanged)
# ---------------------------------------------------------------------------


def call_nanobot(user_message: str) -> tuple[str, bool]:
    """Call nanobot for full multi-turn agent loop.

    Returns (response_text, success).
    """
    try:
        session_id = f"grpo-{uuid.uuid4().hex[:8]}"
        resp = requests.post(
            f"{NANOBOT_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": user_message}],
                "session_id": session_id,
            },
            timeout=NANOBOT_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return content, True
    except Exception as exc:
        logger.warning("nanobot call failed: %s", exc)
        return "", False


# ---------------------------------------------------------------------------
# Agent Loop trajectory parsing
# ---------------------------------------------------------------------------


def _parse_agent_loop_trajectory(
    solution_str: str,
    user_message: str,
) -> list[dict]:
    """Parse Agent Loop multi-turn output into conversation for rubric scoring.

    Agent Loop output has interleaved LLM text and <tool_call>/<tool_response>
    XML blocks.  We split on <tool_response> markers to build a conversation
    with alternating assistant/tool roles.
    """
    conversation: list[dict] = [{"role": "user", "content": user_message}]

    segments = _AGENT_TOOL_RESPONSE_RE.split(solution_str)

    for i, segment in enumerate(segments):
        if i % 2 == 0:
            # Assistant message (may contain <tool_call> blocks)
            text = segment.strip()
            if text:
                conversation.append({"role": "assistant", "content": text})
        else:
            # Tool response
            text = segment.strip()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {"raw": text}
            conversation.append({"role": "tool", "content": json.dumps(data, default=str)})

    return conversation


# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------


def _check_solution_quality(solution_str: str) -> float:
    """Rate the model's response format quality (0.0 - 1.0).

    Checks for valid tool call format (handles both <function_call> from
    nanobot/direct mode and <tool_call> from Agent Loop mode).
    """
    from trainable_openclaw.training.rollout_env import parse_tool_calls_from_text

    if not solution_str or not solution_str.strip():
        return 0.0

    tool_calls = parse_tool_calls_from_text(solution_str)
    if not tool_calls:
        return 0.5  # plain text — neutral

    quality = 1.0
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})
        if not name:
            quality -= 0.3
        if isinstance(args, str):
            try:
                json.loads(args)
            except json.JSONDecodeError:
                quality -= 0.2
        if isinstance(args, dict) and len(args) == 0:
            quality -= 0.1

    return max(0.0, quality)


# ---------------------------------------------------------------------------
# Main reward function (verl interface)
# ---------------------------------------------------------------------------


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Compute reward for a rollout trajectory.

    Three modes (set via REWARD_MODE env var):

    agent (default):
        solution_str contains the full Agent Loop multi-turn trajectory
        (LLM + tool tokens). Parse it, build conversation, score with rubric.

    direct:
        solution_str is a single-turn response. Execute tool calls locally,
        build single-turn conversation, score with rubric.

    nanobot:
        Call nanobot for multi-turn agent loop. If unreachable, fall back
        to single-turn scoring (same as direct mode).

    Args:
        data_source: Task identifier.
        solution_str: Model generated text (decoded from response_ids).
        ground_truth: Task data (JSON string).
        extra_info: Metadata from verl.

    Returns:
        Reward float (0.0 - 1.0).
    """
    from trainable_openclaw.training.rollout_env import parse_tool_calls_from_text

    executor = _get_executor()
    executor.reset()
    engine = _get_rubric_engine()

    user_message = _extract_user_message(extra_info, ground_truth)

    # ── 1. Format quality from solution_str ────────────────────────────
    quality = _check_solution_quality(solution_str)

    # ── 2. Score the trajectory ────────────────────────────────────────
    if REWARD_MODE == "agent":
        # Agent Loop mode: solution_str = full multi-turn trajectory
        # Parse the interleaved assistant messages + tool responses,
        # build conversation, score with rubric.
        conv = _parse_agent_loop_trajectory(solution_str, user_message)
        result = engine.score({"conversations": conv, "status": "in_progress"})
        reward = quality * result["weighted_reward"]

    elif REWARD_MODE == "direct":
        # Direct mode: score verl's rollout output directly (single-turn)
        tool_calls = parse_tool_calls_from_text(solution_str)
        tool_results = _execute_tools(executor, tool_calls)
        conv = _build_conversation(user_message, solution_str, tool_results)
        result = engine.score({"conversations": conv, "status": "in_progress"})
        reward = quality * result["weighted_reward"]

    else:
        # Nanobot mode: call nanobot for full multi-turn agent loop
        nb_response, nb_ok = call_nanobot(user_message)

        if not nb_ok:
            # Nanobot failed — fall back to single-turn scoring
            tool_calls = parse_tool_calls_from_text(solution_str)
            tool_results = _execute_tools(executor, tool_calls)
            conv = _build_conversation(user_message, solution_str, tool_results)
            result = engine.score({"conversations": conv, "status": "in_progress"})
            reward = quality * result["weighted_reward"]
        else:
            # Nanobot succeeded — score the multi-turn trajectory
            nb_tool_calls = parse_tool_calls_from_text(nb_response)
            executor.reset()
            nb_results = _execute_tools(executor, nb_tool_calls)
            nb_conv = _build_conversation(user_message, nb_response, nb_results)
            nb_score = engine.score({"conversations": nb_conv, "status": "complete"})
            if quality < 0.3:
                reward = 0.1 * quality
            else:
                reward = quality * nb_score["weighted_reward"]

    reward = round(min(1.0, max(0.0, reward)), 4)
    logger.debug("Reward=%.4f quality=%.2f mode=%s src=%s", reward, quality, REWARD_MODE, data_source)
    return reward


def _execute_tools(executor, tool_calls: list[dict]) -> list[dict]:
    results: list[dict] = []
    for tc in tool_calls:
        try:
            result = executor.execute(tc["name"], tc["arguments"])
        except Exception:
            result = {"status": "error", "message": "execution failed"}
        results.append({"name": tc.get("name", "unknown"), "result": result})
    return results


def _build_conversation(user_msg: str, agent_msg: str, tool_results: list[dict]) -> list[dict]:
    conv = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": agent_msg},
    ]
    if tool_results:
        conv.append({"role": "tool", "content": json.dumps(tool_results, default=str)})
    return conv


# ---------------------------------------------------------------------------
# Batch version (for offline testing)
# ---------------------------------------------------------------------------


def compute_rewards_batch(
    prompts: list[dict],
    responses: list[str],
    **kwargs,
) -> list[float]:
    rewards: list[float] = []
    for prompt, response in zip(prompts, responses):
        reward = compute_score(
            data_source=prompt.get("source", "taubench_retail"),
            solution_str=response,
            ground_truth=json.dumps(prompt),
            extra_info=prompt,
        )
        rewards.append(reward)
    return rewards
