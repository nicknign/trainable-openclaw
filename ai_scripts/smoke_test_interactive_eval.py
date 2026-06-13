"""Smoke test: run interactive evaluation on a few tau-bench tasks with real LLM.

Requires DEEPSEEK_API_KEY environment variable.

Usage:
    python scripts/smoke_test_interactive_eval.py [--tasks N] [--max-rounds M]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainable_openclaw.agent.tau_bench_tools.registry import register_tau_bench_tools
from trainable_openclaw.agent.tau_bench_tools.mock_db import MockDatabase
from trainable_openclaw.evaluation import AgentRunner, InteractiveEvaluator


def load_tasks(limit: int = 3) -> list[dict]:
    prompts_path = Path("data/tau_bench/grpo_prompts.jsonl")
    with open(prompts_path, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    # Pick a variety: simple retail tasks first
    retail = [t for t in tasks if t.get("domain") == "retail"]
    airline = [t for t in tasks if t.get("domain") == "airline"]
    selected = retail[: limit * 2 // 3] + airline[: limit // 3]
    return selected[:limit]


def build_tool_executor(scenario: str):
    db = MockDatabase(scenario)
    tools = register_tau_bench_tools(scenario)

    def execute(name: str, args: dict) -> dict:
        for t in tools:
            if t.name == name:
                try:
                    return t.execute(db, args)
                except Exception as e:
                    return {"error": str(e)}
        return {"error": f"Unknown tool: {name}"}

    tool_schemas = [
        {"type": "function", "function": t.to_schema()["function"]} for t in tools
    ]
    return execute, tool_schemas


def main():
    parser = argparse.ArgumentParser(description="Smoke test interactive evaluation")
    parser.add_argument("--tasks", type=int, default=2, help="Number of tasks to evaluate")
    parser.add_argument("--max-rounds", type=int, default=8, help="Max rounds per task")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: Set DEEPSEEK_API_KEY environment variable.")
        sys.exit(1)

    tasks = load_tasks(args.tasks)
    print(f"Loaded {len(tasks)} tasks for evaluation\n")

    # We need the tools from all domains
    retail_exec, retail_schemas = build_tool_executor("retail")
    airline_exec, airline_schemas = build_tool_executor("airline")

    # For simplicity, use retail tools for retail tasks, airline for airline
    # This is handled per-task in the full implementation

    for i, task in enumerate(tasks):
        domain = task.get("domain", "retail")
        executor, schemas = build_tool_executor(domain)

        agent = AgentRunner(tools=schemas, tool_executor=executor)
        evaluator = InteractiveEvaluator(agent=agent, max_rounds=args.max_rounds)

        print(f"--- Task {i+1}/{len(tasks)}: {task['id']} ---")
        print(f"   Prompt: {task['prompt'][:120]}...")
        report = evaluator.evaluate([task])
        d = report.to_dict()
        print(f"   Result: completed={d['completed_tasks']}/{d['total_tasks']}, "
              f"rounds={d['avg_rounds']}, first_try={d['first_try_rate']:.0%}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
