"""Full interactive evaluation on all 50 augmented test prompts.

Reads test_prompts_augmented.jsonl, runs agent-vs-simulated-user loop
for each task, saves aggregated report and per-task trajectories.

Usage:
    python scripts/eval/run_full_eval.py [--max-rounds N]

API config: deepseek-chat via DeepSeek API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainable_openclaw.agent.tau_bench_tools.registry import register_tau_bench_tools
from trainable_openclaw.agent.tau_bench_tools.mock_db import MockDatabase
from trainable_openclaw.evaluation import AgentRunner, InteractiveEvaluator

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

PROMPTS_PATH = Path("data/tau_bench/test_prompts_augmented.jsonl")
OUTPUT_DIR = Path("data/eval")
REPORT_PATH = OUTPUT_DIR / "baseline_report.json"
TRAJ_PATH = OUTPUT_DIR / "baseline_trajectories.jsonl"


def build_tool_executor(scenario: str):
    """Create a tool executor and schema list for the given domain."""
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
    """Load all 50 tasks from augmented test prompts file."""
    tasks = []
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def run_evaluation(max_rounds: int = 10) -> tuple[dict, list[dict]]:
    """Run all 50 tasks. Returns (report_dict, trajectories_list)."""
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks from {PROMPTS_PATH}\n")

    # Pre-build tool executors for both domains (reuse DB across tasks)
    tool_execs: dict[str, tuple] = {}
    for domain in ("retail", "airline"):
        tool_execs[domain] = build_tool_executor(domain)
        print(f"Built tool executor for {domain} ({len(tool_execs[domain][1])} tools)")

    print()

    all_trajectories: list[dict] = []
    task_results_data: list[dict] = []

    for i, task in enumerate(tasks):
        task_id = task.get("id", f"task_{i}")
        domain = task.get("domain", "retail")
        print(f"[{i+1:2d}/{len(tasks)}] {task_id} ({domain}) ... ", end="", flush=True)

        executor, schemas = tool_execs[domain]

        agent = AgentRunner(
            tools=schemas,
            tool_executor=executor,
            model=MODEL,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        evaluator = InteractiveEvaluator(
            agent=agent,
            model=MODEL,
            api_key=API_KEY,
            base_url=BASE_URL,
            max_rounds=max_rounds,
        )

        t_start = time.time()

        try:
            report = evaluator.evaluate([task])
            result = report.results[0]
            elapsed = time.time() - t_start
            status = "OK" if result.completed else ("GIVEUP" if result.status == "give_up" else "TIMEOUT")
            print(f"{status} | rounds={result.rounds} | sat={result.satisfaction:.1f} | {elapsed:.0f}s")
        except Exception as e:
            elapsed = time.time() - t_start
            print(f"ERROR | {elapsed:.0f}s | {e}")
            # Create a synthetic failure result
            from trainable_openclaw.evaluation.interactive_eval import TaskResult
            result = TaskResult(
                task_id=task_id,
                domain=domain,
                completed=False,
                rounds=0,
                satisfaction=0.0,
                status="error",
                conversation=[],
            )

        # Collect task-level result
        task_results_data.append({
            "task_id": task_id,
            "domain": domain,
            "completed": result.completed,
            "rounds": result.rounds,
            "satisfaction": result.satisfaction,
            "status": result.status,
            "first_try": result.completed and result.rounds == 1,
        })

        # Collect trajectory
        trajectory_entry = {
            "task_id": task_id,
            "domain": domain,
            "prompt": task.get("prompt", ""),
            "evaluation": task.get("evaluation", {}),
            "result": {
                "completed": result.completed,
                "rounds": result.rounds,
                "satisfaction": result.satisfaction,
                "status": result.status,
            },
            "conversation": result.conversation,
        }
        all_trajectories.append(trajectory_entry)

    # ---- Build final report ----
    total = len(task_results_data)
    completed = [r for r in task_results_data if r["completed"]]
    n_completed = len(completed)
    n_first_try = sum(1 for r in task_results_data if r["first_try"])
    n_recovery = sum(1 for r in completed if r["rounds"] > 1)
    n_giveup = sum(1 for r in task_results_data if r["status"] == "give_up")
    n_timeout = sum(1 for r in task_results_data if r["status"] == "timeout")
    n_error = sum(1 for r in task_results_data if r["status"] == "error")
    avg_rounds = sum(r["rounds"] for r in task_results_data) / total if total else 0

    # Domain breakdown
    domain_breakdown = {}
    for domain in ("retail", "airline"):
        dt = [r for r in task_results_data if r["domain"] == domain]
        dc = [r for r in dt if r["completed"]]
        domain_breakdown[domain] = {
            "total": len(dt),
            "completed": len(dc),
            "completion_rate": round(len(dc) / len(dt), 3) if dt else 0,
            "avg_rounds": round(sum(r["rounds"] for r in dt) / len(dt), 2) if dt else 0,
        }

    report_dict = {
        "total_tasks": total,
        "completed_tasks": n_completed,
        "completion_rate": round(n_completed / total, 3) if total else 0,
        "avg_rounds": round(avg_rounds, 2),
        "first_try_rate": round(n_first_try / total, 3) if total else 0,
        "recovery_rate": round(n_recovery / n_completed, 3) if n_completed else 0,
        "abandonment_rate": round(n_giveup / total, 3) if total else 0,
        "timeout_rate": round(n_timeout / total, 3) if total else 0,
        "error_count": n_error,
        "domain_breakdown": domain_breakdown,
        "per_task": task_results_data,
    }

    return report_dict, all_trajectories


def main():
    parser = argparse.ArgumentParser(description="Full interactive evaluation on 50 test prompts")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max rounds per task")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  FULL INTERACTIVE EVALUATION — 50 test prompts")
    print(f"  Model: {MODEL} @ {BASE_URL}")
    print(f"  Max rounds: {args.max_rounds}")
    print("=" * 60)
    print()

    total_start = time.time()
    report_dict, trajectories = run_evaluation(max_rounds=args.max_rounds)
    total_elapsed = time.time() - total_start

    # Save report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {REPORT_PATH}")

    # Save trajectories
    with open(TRAJ_PATH, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    print(f"Trajectories saved to {TRAJ_PATH}")

    # Print final summary
    print()
    print("=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Tasks:         {report_dict['total_tasks']}")
    print(f"  Completed:     {report_dict['completed_tasks']} ({report_dict['completion_rate']:.1%})")
    print(f"  First-try:     {report_dict['first_try_rate']:.1%}")
    print(f"  Recovery:      {report_dict['recovery_rate']:.1%}")
    print(f"  Abandoned:     {report_dict['abandonment_rate']:.1%}")
    print(f"  Timeout:       {report_dict['timeout_rate']:.1%}")
    print(f"  Errors:        {report_dict['error_count']}")
    print(f"  Avg rounds:    {report_dict['avg_rounds']}")
    print()
    print("  Domain breakdown:")
    for domain, stats in report_dict["domain_breakdown"].items():
        print(f"    {domain}: {stats['completed']}/{stats['total']} completed ({stats['completion_rate']:.1%}), "
              f"avg {stats['avg_rounds']} rounds")
    print()
    print(f"  Total time:    {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print("=" * 60)


if __name__ == "__main__":
    main()
