"""Run one interactive eval task on remote (Qwen3-4B via nanobot, DeepSeek as simulated user).

Usage (on remote):
    /data/anaconda3/bin/python ai_scripts/remote_eval_one.py
    /data/anaconda3/bin/python ai_scripts/remote_eval_one.py --task 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainable_openclaw.evaluation import QwenAgentRunner, InteractiveEvaluator

# ── config ──────────────────────────────────────────────────────────────────
AGENT_MODEL = "qwen3-4b"
AGENT_BASE_URL = "http://localhost:8000/v1"
AGENT_API_KEY = "no-key"
AGENT_USE_NATIVE_TOOLS = False  # Qwen3-4B uses text-based tool calling

# DeepSeek for simulated user
SIM_USER_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
SIM_USER_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
SIM_USER_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

PROMPTS_PATH = Path(__file__).resolve().parent.parent / "data" / "tau_bench" / "test_prompts_augmented.jsonl"


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
    parser = argparse.ArgumentParser(description="Remote single-task eval")
    parser.add_argument("--task", type=int, default=0, help="Task index (0-based)")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max rounds")
    args = parser.parse_args()

    tasks = load_tasks()
    task = tasks[args.task]

    task_id = task["id"]
    domain = task["domain"]
    print(f"Task: {task_id} ({domain})")
    print(f"Prompt: {task['prompt'][:300]}")
    print(f"Tools count: {len(task.get('tools', []))}")
    print()

    executor, schemas = build_tool_executor(domain)
    print(f"Loaded {len(schemas)} tools for {domain}")
    print()

    agent = QwenAgentRunner(
        tools=schemas,
        tool_executor=executor,
        model=AGENT_MODEL,
        api_key=AGENT_API_KEY,
        base_url=AGENT_BASE_URL,
    )
    evaluator = InteractiveEvaluator(
        agent=agent,
        model=SIM_USER_MODEL,
        api_key=SIM_USER_API_KEY,
        base_url=SIM_USER_BASE_URL,
        max_rounds=args.max_rounds,
    )

    print(f"Agent: {AGENT_MODEL} @ {AGENT_BASE_URL}")
    print(f"SimUser: {SIM_USER_MODEL}")
    print(f"Max rounds: {args.max_rounds}")
    print()

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
        print(f"--- Round {i // 2 + 1} [{role}] ---")
        print(content[:600])
        if tools:
            print(f"  [Tools called: {len(tools)}]")
            for t in tools:
                tc = str(t.get("content", ""))
                print(f"  - {t['name']}: {tc[:300]}")
        print()


if __name__ == "__main__":
    main()
