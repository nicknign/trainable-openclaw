"""
Interactive evaluation harness for tau-bench agents.

Runs the agent ↔ simulated-user loop and collects efficiency metrics:
rounds-to-completion, first-try success rate, recovery rate, abandonment rate.

Works with any agent that implements the ``AgentRunner`` protocol (an LLM API
stand-in today; swap in the trained Qwen3-4B + nanobot later).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from trainable_openclaw.evaluation.simulated_user import SimulatedUser, UserResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id: str
    domain: str
    completed: bool
    rounds: int
    satisfaction: float
    status: str          # "complete" | "give_up" | "timeout"
    conversation: list[dict] = field(default_factory=list)

    @property
    def first_try_success(self) -> bool:
        return self.completed and self.rounds == 1

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "completed": self.completed,
            "rounds": self.rounds,
            "satisfaction": self.satisfaction,
            "status": self.status,
            "first_try": self.first_try_success,
        }


@dataclass
class EvalReport:
    results: list[TaskResult]
    total_tasks: int
    completed_tasks: int
    avg_rounds: float
    first_try_rate: float
    recovery_rate: float   # completed after correction / total completed
    abandonment_rate: float

    def to_dict(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "completion_rate": round(self.completed_tasks / self.total_tasks, 3) if self.total_tasks else 0,
            "avg_rounds": round(self.avg_rounds, 2),
            "first_try_rate": round(self.first_try_rate, 3),
            "recovery_rate": round(self.recovery_rate, 3),
            "abandonment_rate": round(self.abandonment_rate, 3),
        }

    def print(self) -> str:
        d = self.to_dict()
        lines = [
            "=" * 50,
            "  INTERACTIVE EVALUATION REPORT",
            "=" * 50,
            f"  Tasks:         {d['total_tasks']}",
            f"  Completed:     {d['completed_tasks']} ({d['completion_rate']:.1%})",
            f"  Avg rounds:    {d['avg_rounds']}",
            f"  First-try:     {d['first_try_rate']:.1%}",
            f"  Recovery:      {d['recovery_rate']:.1%}",
            f"  Abandoned:     {d['abandonment_rate']:.1%}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent runner protocol
# ---------------------------------------------------------------------------

class AgentRunner:
    """Pluggable agent backend for interactive evaluation.

    Default implementation uses an LLM API with tool calling.  Swap in a
    trained model + nanobot by providing a different ``run`` callable later.
    """

    def __init__(
        self,
        tools: list[dict],
        tool_executor,
        model: str = "deepseek-chat",
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
    ):
        self._tools = tools
        self._execute_tool = tool_executor  # callable: (name, args) -> result dict
        self._model = model
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self._client = None
        self._conversation_messages: list[dict] = []

    def reset_conversation(self):
        """Reset the persistent conversation state for a new task."""
        self._conversation_messages = [{"role": "system", "content": _AGENT_SYSTEM_PROMPT}]

    def run(self, user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
        """Run the agent for one turn: receive user message, call tools if
        needed, return a text response and any tool results.

        Uses a persistent ``_conversation_messages`` list so tool calls and
        results from previous rounds are retained across calls to ``run()``.
        Call ``reset_conversation()`` before starting a new task.

        Returns:
            (text_response_to_user, list_of_tool_results)
        """
        # Lazy-init for backward compatibility (tests that don't call reset_conversation)
        if not self._conversation_messages:
            self.reset_conversation()

        # Append the incoming user message to the persistent history
        self._conversation_messages.append({"role": "user", "content": user_message})

        client = self._get_client()

        # Internal loop: LLM may request tool calls, execute them, feed back
        tool_results = []
        for _ in range(10):  # max 10 internal tool-call iterations per turn
            response = client.chat.completions.create(
                model=self._model,
                messages=self._conversation_messages,
                tools=self._tools,
                temperature=0.3,
                max_tokens=2000,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    result = self._execute_tool(name, args)
                    tool_results.append({"name": name, "content": result})
                    self._conversation_messages.append({"role": "assistant", "tool_calls": [tc.model_dump()]})
                    self._conversation_messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
            else:
                # Text response — append to persistent history and return
                text = msg.content or ""
                self._conversation_messages.append({"role": "assistant", "content": text})
                return (text, tool_results)

        fallback = "I wasn't able to complete that request."
        self._conversation_messages.append({"role": "assistant", "content": fallback})
        return (fallback, tool_results)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

_AGENT_SYSTEM_PROMPT = (
    "You are a customer service agent. Use the available tools to help the customer efficiently. "
    "IMPORTANT: The customer's first message already contains their request and identity information. "
    "Do NOT ask for information they already provided. Extract what you need from their message "
    "and start using tools immediately. "
    "After completing tool calls, give a clear summary to the customer. "
    "If a tool returns an error, try alternative approaches before giving up. "
    "Be concise and direct."
)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class InteractiveEvaluator:
    """Orchestrates the agent ↔ simulated-user loop over a set of test tasks."""

    def __init__(
        self,
        agent: AgentRunner,
        model: str = "deepseek-chat",
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        max_rounds: int = 10,
    ):
        self._agent = agent
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_rounds = max_rounds

    def evaluate(self, tasks: list[dict]) -> EvalReport:
        """Run interactive evaluation on a list of task definitions.

        Each task dict must contain: ``id``, ``domain``, ``prompt``,
        ``evaluation``, ``tools``.
        """
        results: list[TaskResult] = []
        for i, task in enumerate(tasks):
            logger.info("Evaluating task %d/%d: %s", i + 1, len(tasks), task.get("id", "?"))
            tr = self._evaluate_one(task)
            results.append(tr)

        return self._build_report(results)

    def evaluate_file(self, path: str | Path, limit: int = 0) -> EvalReport:
        """Load tasks from a JSONL file and evaluate them."""
        tasks = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        if limit:
            tasks = tasks[:limit]
        return self.evaluate(tasks)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _evaluate_one(self, task: dict) -> TaskResult:
        sim_user = SimulatedUser(
            task, model=self._model, api_key=self._api_key, base_url=self._base_url,
        )
        # Reset agent conversation state per task (no-op for non-AgentRunner backends)
        if hasattr(self._agent, "reset_conversation"):
            self._agent.reset_conversation()
        conversation: list[dict] = []
        user_msg = sim_user.initial_message

        for r in range(1, self._max_rounds + 1):
            agent_msg, tool_results = self._agent.run(user_msg, sim_user.history)
            conversation.append({"role": "agent", "content": agent_msg, "tool_results": tool_results})

            feedback = sim_user.respond(agent_msg, tool_results)
            conversation.append({"role": "user", "content": feedback.message})

            if feedback.status == "complete":
                return TaskResult(
                    task_id=task.get("id", "?"),
                    domain=task.get("domain", ""),
                    completed=True,
                    rounds=r,
                    satisfaction=feedback.satisfaction,
                    status="complete",
                    conversation=conversation,
                )
            elif feedback.status == "give_up":
                return TaskResult(
                    task_id=task.get("id", "?"),
                    domain=task.get("domain", ""),
                    completed=False,
                    rounds=r,
                    satisfaction=feedback.satisfaction,
                    status="give_up",
                    conversation=conversation,
                )

            user_msg = feedback.message

        return TaskResult(
            task_id=task.get("id", "?"),
            domain=task.get("domain", ""),
            completed=False,
            rounds=self._max_rounds,
            satisfaction=0.0,
            status="timeout",
            conversation=conversation,
        )

    def _build_report(self, results: list[TaskResult]) -> EvalReport:
        total = len(results)
        if total == 0:
            return EvalReport(results=[], total_tasks=0, completed_tasks=0, avg_rounds=0.0, first_try_rate=0.0, recovery_rate=0.0, abandonment_rate=0.0)

        completed = [r for r in results if r.completed]
        n_completed = len(completed)
        avg_rounds = sum(r.rounds for r in results) / total
        first_try = sum(1 for r in results if r.first_try_success) / total
        recovery = sum(1 for r in completed if r.rounds > 1) / n_completed if n_completed else 0.0
        abandoned = sum(1 for r in results if r.status == "give_up") / total

        return EvalReport(
            results=results,
            total_tasks=total,
            completed_tasks=n_completed,
            avg_rounds=avg_rounds,
            first_try_rate=first_try,
            recovery_rate=recovery,
            abandonment_rate=abandoned,
        )
