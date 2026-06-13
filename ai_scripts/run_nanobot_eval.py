"""Run interactive evaluation with Qwen3.5-2B via nanobot + DeepSeek simulated user.

nanobot API (:8900) manages agent conversation and tool calling internally.
Each turn sends the simulated user's message to nanobot with a persistent
session_id for conversation continuity.

Usage:
    python ai_scripts/run_nanobot_eval.py                  # task index 0
    python ai_scripts/run_nanobot_eval.py --task 3         # task index 3
    python ai_scripts/run_nanobot_eval.py --task 0 --max-rounds 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainable_openclaw.evaluation.simulated_user import SimulatedUser
from trainable_openclaw.evaluation.interactive_eval import (
    TaskResult, EvalReport,
)

# ── config ──────────────────────────────────────────────────────────────────

NANOBOT_URL = os.environ.get("NANOBOT_URL", "http://localhost:8900")
AGENT_MODEL = "qwen3.5-2b-nanobot"

# DeepSeek for simulated user
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"

PROMPTS_PATH = Path(__file__).resolve().parent.parent / "data" / "tau_bench" / "test_prompts_augmented.jsonl"


def load_tasks() -> list[dict]:
    tasks = []
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def call_nanobot(message: str, session_id: str, timeout: int = 600) -> str:
    """Send a single message to nanobot API and return the assistant response."""
    resp = requests.post(
        f"{NANOBOT_URL}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": message}],
            "session_id": session_id,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    return body["choices"][0]["message"]["content"]


class NanobotEvaluator:
    """Evaluator: nanobot agent (via API) + DeepSeek simulated user."""

    def __init__(self, max_rounds: int = 10, teacher_mode: bool = True):
        self._max_rounds = max_rounds
        self._teacher_mode = teacher_mode

    def evaluate(self, tasks: list[dict]) -> EvalReport:
        results: list[TaskResult] = []
        for i, task in enumerate(tasks):
            tid = task.get("id", "?")
            domain = task.get("domain", "")
            print(f"\n{'='*60}")
            print(f"Task {i+1}/{len(tasks)}: {tid} ({domain})")
            print(f"Prompt: {task['prompt'][:150]}...")
            print(f"{'='*60}")
            tr = self._evaluate_one(task)
            results.append(tr)
            status_icon = "[OK]" if tr.completed else "[FAIL]"
            print(f"{status_icon} completed={tr.completed} rounds={tr.rounds} satisfaction={tr.satisfaction} status={tr.status}")
        return self._build_report(results)

    def _evaluate_one(self, task: dict) -> TaskResult:
        task_id = task.get("id", "?")
        domain = task.get("domain", "")

        sim_user = SimulatedUser(
            task,
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            mode="teacher" if self._teacher_mode else "customer",
        )

        # Use task ID as session key for conversation isolation
        session_id = f"eval-{task_id}"

        conversation: list[dict] = []
        user_msg = sim_user.initial_message

        for r in range(1, self._max_rounds + 1):
            print(f"\n--- Round {r} ---")
            print(f"[SimUser] {user_msg[:200]}")

            try:
                agent_msg = call_nanobot(user_msg, session_id)
            except Exception as exc:
                print(f"[Agent ERROR] {exc}")
                return TaskResult(
                    task_id=task_id, domain=domain,
                    completed=False, rounds=r, satisfaction=0.0,
                    status="agent_error", conversation=conversation,
                )

            conversation.append({"role": "agent", "content": agent_msg})
            print(f"[Agent] {agent_msg[:300]}")

            feedback = sim_user.respond(agent_msg, [])
            conversation.append({"role": "user", "content": feedback.message})
            print(f"[SimUser] {feedback.message[:200]} (status={feedback.status}, sat={feedback.satisfaction})")

            if feedback.status == "complete":
                return TaskResult(
                    task_id=task_id, domain=domain,
                    completed=True, rounds=r, satisfaction=feedback.satisfaction,
                    status="complete", conversation=conversation,
                )
            elif feedback.status == "give_up":
                return TaskResult(
                    task_id=task_id, domain=domain,
                    completed=False, rounds=r, satisfaction=feedback.satisfaction,
                    status="give_up", conversation=conversation,
                )
            user_msg = feedback.message

        return TaskResult(
            task_id=task_id, domain=domain,
            completed=False, rounds=self._max_rounds, satisfaction=0.0,
            status="timeout", conversation=conversation,
        )

    def _build_report(self, results: list[TaskResult]) -> EvalReport:
        total = len(results)
        if total == 0:
            return EvalReport(results=[], total_tasks=0, completed_tasks=0,
                              avg_rounds=0.0, first_try_rate=0.0,
                              recovery_rate=0.0, abandonment_rate=0.0)
        n_completed = sum(1 for r in results if r.completed)
        avg_rounds = sum(r.rounds for r in results) / total
        first_try = sum(1 for r in results if r.first_try_success) / total
        recovery = sum(1 for r in results if r.completed and r.rounds > 1) / n_completed if n_completed else 0.0
        abandoned = sum(1 for r in results if r.status == "give_up") / total
        return EvalReport(results=results, total_tasks=total,
                          completed_tasks=n_completed, avg_rounds=avg_rounds,
                          first_try_rate=first_try, recovery_rate=recovery,
                          abandonment_rate=abandoned)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run eval with nanobot agent + DeepSeek user")
    parser.add_argument("--task", type=int, default=0, help="Task index (0-based)")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max rounds")
    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        print("ERROR: Set DEEPSEEK_API_KEY environment variable.")
        sys.exit(1)

    # Verify nanobot is reachable
    try:
        r = requests.get(f"{NANOBOT_URL}/health", timeout=5)
        print(f"nanobot health: {r.json()}")
    except Exception as exc:
        print(f"ERROR: Cannot reach nanobot at {NANOBOT_URL}: {exc}")
        sys.exit(1)

    tasks = load_tasks()
    task = tasks[args.task]
    tid = task["id"]
    domain = task["domain"]

    print(f"Task: {tid} ({domain})")
    print(f"Agent: nanobot ({AGENT_MODEL}) @ {NANOBOT_URL}")
    print(f"SimUser: {DEEPSEEK_MODEL} @ {DEEPSEEK_BASE_URL}")
    print(f"Max rounds: {args.max_rounds}")
    print(f"Task prompt: {task['prompt'][:200]}...")
    print()

    evaluator = NanobotEvaluator(max_rounds=args.max_rounds)

    t0 = time.time()
    report = evaluator.evaluate([task])
    elapsed = time.time() - t0

    result = report.results[0]
    print("\n" + "=" * 60)
    print("  EVALUATION RESULT")
    print("=" * 60)
    print(f"  Task:        {result.task_id} ({result.domain})")
    print(f"  Completed:   {result.completed}")
    print(f"  Rounds:      {result.rounds}")
    print(f"  Satisfaction: {result.satisfaction}")
    print(f"  Status:      {result.status}")
    print(f"  Time:        {elapsed:.0f}s")
    print("=" * 60)
    print("\n--- Full Conversation ---")
    for i, turn in enumerate(result.conversation):
        role = turn["role"].upper()
        content = turn.get("content", "")
        print(f"\n[{role}] Round {i//2 + 1}")
        print(content[:600])


if __name__ == "__main__":
    main()
