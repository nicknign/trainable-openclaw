"""
Evaluation package — interactive and rubric-based agent evaluation.

Interactive (new):
    SimulatedUser → InteractiveEvaluator → EvalReport
    Metric: rounds-to-completion (lower = better)

Rubric-based (legacy, retained for reference):
    trajectory_eval → feedback → rubric → judge
"""

from trainable_openclaw.evaluation.simulated_user import SimulatedUser, UserResponse
from trainable_openclaw.evaluation.interactive_eval import (
    AgentRunner,
    InteractiveEvaluator,
    TaskResult,
    EvalReport,
)
from trainable_openclaw.evaluation.qwen_agent_runner import QwenAgentRunner

__all__ = [
    "SimulatedUser",
    "UserResponse",
    "AgentRunner",
    "QwenAgentRunner",
    "InteractiveEvaluator",
    "TaskResult",
    "EvalReport",
]
