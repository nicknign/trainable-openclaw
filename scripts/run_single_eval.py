"""Run interactive evaluation on a single task for quick verification.

Usage:
    python scripts/run_single_eval.py                  # first task
    python scripts/run_single_eval.py --task 3         # task index 3
    python scripts/run_single_eval.py --task-id retail_task_40  # by task ID
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainable_openclaw.evaluation import AgentRunner, InteractiveEvaluator


API_KEY = "sk-38a1b2445e3a427b8bbf74a13ffee42a"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
PROMPTS_PATH = Path("data/tau_bench/test_prompts_augmented.jsonl")


def build_tool_executor(scenario: str):
    from trainable_openclaw.agent.tau_bench_tools.registry import register_tau_bench_tools
    from trainable_openclaw.agent.tau_bench_tools.mock_db import MockDatabase

    db = MockDatabase(scenario)
    tools = register_tau_bench_tools(scenario)

    def execute(name: str, args: dict) -> dict:
        for t in tools:
            if t.name == name:
                try:
                    return t.execute(args, db.state)
                except Exception as e:
                    return {"error": str(e)}
        return {"error": f"Unknown tool: {name}"}

    tool_schemas = [
        {"type": "function", "function": t.to_schema()["function"]} for t in tools
    ]
    return execute, tool_schemas


def load_tasks() -> list[dict]:
    tasks = []
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Run single-task interactive evaluation")
    parser.add_argument("--task", type=int, default=0, help="Task index (0-based)")
    parser.add_argument("--task-id", type=str, default="", help="Task ID to run")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max rounds")
    args = parser.parse_args()

    tasks = load_tasks()

    if args.task_id:
        matches = [t for t in tasks if t["id"] == args.task_id]
        if not matches:
            print(f"Task '{args.task_id}' not found. Available IDs:")
            for t in tasks:
                print(f"  {t['id']} ({t['domain']})")
            sys.exit(1)
        task = matches[0]
    else:
        task = tasks[args.task]

    task_id = task["id"]
    domain = task["domain"]
    print(f"Task: {task_id} ({domain})")
    print(f"Prompt: {task['prompt'][:200]}...")
    print(f"Tools: {task.get('tools', [])}")
    print()

    executor, schemas = build_tool_executor(domain)
    print(f"Loaded {len(schemas)} tools for {domain}")
    print()

    agent = AgentRunner(
        tools=schemas, tool_executor=executor,
        model=MODEL, api_key=API_KEY, base_url=BASE_URL,
    )
    evaluator = InteractiveEvaluator(
        agent=agent,
        model=MODEL, api_key=API_KEY, base_url=BASE_URL,
        max_rounds=args.max_rounds,
    )

    t0 = time.time()
    report = evaluator.evaluate([task])
    elapsed = time.time() - t0
    result = report.results[0]

    print("=" * 60)
    print(f"  completed={result.completed}  rounds={result.rounds}"
          f"  satisfaction={result.satisfaction}"
          f"  status={result.status}  time={elapsed:.0f}s")
    print("=" * 60)
    print()

    for i, turn in enumerate(result.conversation):
        role = turn["role"].upper()
        content = turn.get("content", "")
        tools = turn.get("tool_results", [])
        print(f"--- Round {i//2 + 1} [{role}] ---")
        print(content[:500])
        if tools:
            print(f"  [Tools called: {len(tools)}]")
            for t in tools:
                tc = str(t.get("content", ""))
                print(f"  - {t['name']}: {tc[:200]}")
        print()


if __name__ == "__main__":
    main()
