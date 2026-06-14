"""
GRPO reward function compatible with verl framework.

Provides compute_score() matching verl's per-item reward function signature:
    compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)

During training, verl calls this for each rollout response. We parse the
model-generated response, execute any tool calls against the mock DB, build a
trajectory, and score it with the RubricRuleEngine.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_ENGINE: Any = None
_EXECUTOR: Any = None
_PROMPT_CACHE: dict[str, dict] = {}


def _load_prompts(path: str = "") -> dict[str, dict]:
    """Load training prompts into a dict keyed by task id."""
    if not path:
        path = os.environ.get(
            "TAU_BENCH_TRAIN_PROMPTS",
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "tau_bench", "train_prompts_augmented.jsonl"),
        )
    prompts: dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    tid = obj.get("id", obj.get("task_id", ""))
                    prompts[str(tid)] = obj
    except FileNotFoundError:
        logger.warning("Training prompts file not found: %s", path)
    return prompts


def _get_rubric_engine() -> Any:
    global _ENGINE
    if _ENGINE is None:
        from trainable_openclaw.training.rubric_rules import RubricRuleEngine
        _ENGINE = RubricRuleEngine()
    return _ENGINE


def _get_executor() -> Any:
    global _EXECUTOR
    if _EXECUTOR is None:
        from trainable_openclaw.training.rollout_env import ToolExecutor
        _EXECUTOR = ToolExecutor("retail")
    return _EXECUTOR


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Compute reward for a single model rollout.

    Args:
        data_source: Task identifier (e.g. "taubench_retail").
        solution_str: The model's generated response text.
        ground_truth: Ground truth data (usually JSON string with task info).
        extra_info: Additional metadata from verl.

    Returns:
        Reward float (0.0 - 1.0).
    """
    from trainable_openclaw.training.rollout_env import parse_tool_calls_from_text

    executor = _get_executor()
    executor.reset()

    # Parse tool calls from model response
    tool_calls = parse_tool_calls_from_text(solution_str)

    # Execute tool calls against mock DB
    tool_results: list[dict] = []
    for tc in tool_calls:
        try:
            result = executor.execute(tc["name"], tc["arguments"])
            tool_results.append({"name": tc["name"], "result": result})
        except Exception as exc:
            logger.debug("Tool %s execution failed: %s", tc.get("name"), exc)
            tool_results.append({"name": tc.get("name", "unknown"), "result": {"status": "error", "message": str(exc)}})

    # Build a simple conversation for scoring
    conversation = [
        {"role": "user", "content": _extract_user_message(extra_info, ground_truth)},
        {"role": "assistant", "content": solution_str},
    ]
    if tool_results:
        conversation.append({"role": "tool", "content": json.dumps(tool_results, default=str)})

    # Score the trajectory
    engine = _get_rubric_engine()
    result = engine.score({"conversations": conversation, "status": "in_progress"})

    reward = result["weighted_reward"]
    logger.debug("Reward=%.4f for data_source=%s (tools=%d)", reward, data_source, len(tool_calls))
    return reward


def _extract_user_message(extra_info: dict | None, ground_truth: str) -> str:
    """Extract user message from extra_info or ground_truth."""
    if extra_info:
        prompt = extra_info.get("prompt", "")
        if prompt:
            return prompt
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        if isinstance(gt, dict):
            prompt = gt.get("prompt", "")
            if prompt:
                return prompt
    except (json.JSONDecodeError, TypeError):
        pass
    return "I need help with my order."


# ---------------------------------------------------------------------------
# Batch version (for use outside verl, e.g. eval)
# ---------------------------------------------------------------------------


def compute_rewards_batch(
    prompts: list[dict],
    responses: list[str],
    **kwargs,
) -> list[float]:
    """Compute rewards for a batch of prompt-response pairs.

    Useful for offline evaluation or debugging.
    """
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
