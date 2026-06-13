"""Train/test split & test set interactive evaluation.

Step 1 — Clean data split:
    Reads 164 prompts from ``data/tau_bench/grpo_prompts.jsonl``, groups by
    (domain, task_id) composite key, then produces a stratified ~80/20 split
    with zero task_id leakage.

    Outputs:
        data/tau_bench/train_prompts.jsonl
        data/tau_bench/test_prompts.jsonl

Step 2 — Interactive baseline evaluation:
    Runs the agent-simulated-user loop on every test prompt using DeepSeek
    API + tau-bench mock tools.  Collects rounds-to-completion metrics.

    Requires DEEPSEEK_API_KEY environment variable.

    Outputs:
        data/eval/baseline_report.json
        data/eval/baseline_trajectories.jsonl

Usage:
    python scripts/split_and_evaluate.py [--seed 42] [--max-rounds 8]
                                        [--test-ratio 0.2] [--eval-limit N]
                                        [--step1-only]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "tau_bench"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("split_and_eval")


# ===================================================================
# Step 1: Clean train/test split
# ===================================================================

def load_prompts(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_prompts(
    prompts: list[dict],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split prompts into train/test by (domain, task_id) composite key.

    Stratifies by domain to maintain the original domain distribution in
    each split.  All prompts sharing the same (domain, task_id) go to the
    same split — **zero task_id leakage**.

    Returns:
        (train_prompts, test_prompts)
    """
    rng = random.Random(seed)

    # Group by (domain, task_id) composite key
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in prompts:
        key = (p["domain"], p["task_id"])
        groups[key].append(p)

    # Partition groups by domain for stratified sampling
    domain_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (domain, tid) in groups:
        domain_groups[domain].append((domain, tid))

    train_keys: set[tuple[str, str]] = set()
    test_keys: set[tuple[str, str]] = set()

    for domain, keys in sorted(domain_groups.items()):
        keys_sorted = sorted(keys)
        rng.shuffle(keys_sorted)
        n_test = max(1, round(len(keys_sorted) * test_ratio))
        test_keys.update(keys_sorted[:n_test])
        train_keys.update(keys_sorted[n_test:])

    # Assemble output lists
    train_prompts = [p for key in train_keys for p in groups[key]]
    test_prompts = [p for key in test_keys for p in groups[key]]

    return train_prompts, test_prompts


def _print_split_report(
    all_prompts: list[dict],
    train_prompts: list[dict],
    test_prompts: list[dict],
):
    """Print a human-readable split report."""
    def _stats(entries):
        total = len(entries)
        domains = defaultdict(int)
        task_ids = set()
        for e in entries:
            domains[e["domain"]] += 1
            task_ids.add((e["domain"], e["task_id"]))
        return total, dict(domains), len(task_ids)

    a_total, a_dom, a_tasks = _stats(all_prompts)
    t_total, t_dom, t_tasks = _stats(train_prompts)
    v_total, v_dom, v_tasks = _stats(test_prompts)

    # Verify zero leakage
    train_ids = {(e["domain"], e["task_id"]) for e in train_prompts}
    test_ids = {(e["domain"], e["task_id"]) for e in test_prompts}
    overlap = train_ids & test_ids
    all_ids = {(e["domain"], e["task_id"]) for e in all_prompts}
    unassigned = all_ids - train_ids - test_ids

    print()
    print("=" * 60)
    print("  TRAIN / TEST SPLIT REPORT")
    print("=" * 60)
    print(f"  Total prompts:       {a_total:>4}")
    print(f"  Unique (domain,task_id): {a_tasks:>4}")
    print()
    print(f"  {'Split':<12} {'Prompts':>7}  {'Tasks':>5}  {'Airline':>7}  {'Retail':>7}")
    print(f"  {'-'*12} {'-'*7}  {'-'*5}  {'-'*7}  {'-'*7}")
    print(f"  {'train':<12} {t_total:>7}  {t_tasks:>5}  {t_dom.get('airline', 0):>7}  {t_dom.get('retail', 0):>7}")
    print(f"  {'test':<12} {v_total:>7}  {v_tasks:>5}  {v_dom.get('airline', 0):>7}  {v_dom.get('retail', 0):>7}")
    print()

    def _pct(part, whole):
        return f"{part / whole:.1%}" if whole else "N/A"

    print(f"  Airline ratio — train: {_pct(t_dom.get('airline', 0), a_dom.get('airline', 1))}  "
          f"test: {_pct(v_dom.get('airline', 0), a_dom.get('airline', 1))}")
    print(f"  Retail  ratio — train: {_pct(t_dom.get('retail', 0), a_dom.get('retail', 1))}  "
          f"test: {_pct(v_dom.get('retail', 0), a_dom.get('retail', 1))}")

    checks = [
        ("Zero task_id leakage (overlap)", len(overlap) == 0, f"{len(overlap)} overlapping"),
        ("All prompts assigned", len(unassigned) == 0, f"{len(unassigned)} unassigned"),
        ("Total count matches", t_total + v_total == a_total, f"{t_total + v_total} vs {a_total}"),
    ]
    print()
    for label, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {label}" + (f" ({detail})" if not ok else ""))

    print("=" * 60)

    if not all(ok for _, ok, _ in checks):
        raise RuntimeError("Split verification failed — check the report above.")


def write_jsonl(path: Path, entries: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Wrote %d entries to %s", len(entries), path)


# ===================================================================
# Step 2: Interactive evaluation
# ===================================================================

def _build_tool_executor(domain: str):
    """Return (tool_executor_callable, tool_schemas) for a given domain."""
    from trainable_openclaw.agent.tau_bench_tools.mock_db import MockDatabase
    from trainable_openclaw.agent.tau_bench_tools.registry import register_tau_bench_tools

    db = MockDatabase(domain)
    tools = register_tau_bench_tools(domain)

    def _execute(name: str, args: dict) -> dict:
        for t in tools:
            if t.name == name:
                try:
                    return db.execute(t, args)
                except Exception as exc:
                    return {"error": str(exc)}
        return {"error": f"Unknown tool: {name}"}

    schemas = [
        {"type": "function", "function": t.to_schema()["function"]} for t in tools
    ]
    return _execute, schemas


def run_evaluation(
    test_prompts: list[dict],
    max_rounds: int = 8,
    limit: int = 0,
) -> tuple[dict, list[dict]]:
    """Run interactive evaluation on test prompts.

    Args:
        test_prompts: List of task dicts with ``id``, ``domain``, ``prompt``,
            ``evaluation``, ``tools`` keys.
        max_rounds: Maximum conversation rounds per task.
        limit: If > 0, only evaluate the first N tasks.

    Returns:
        (report_dict, trajectory_list)
    """
    from trainable_openclaw.evaluation import AgentRunner, InteractiveEvaluator

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    tasks = test_prompts[:limit] if limit > 0 else test_prompts
    n_total = len(tasks)

    # Each task gets a FRESH MockDatabase to prevent cross-task state
    # contamination (tool calls in task A must not affect task B's data).

    all_results: list[Any] = []
    trajectories: list[dict] = []
    report_dict: dict = {}

    for i, task in enumerate(tasks):
        domain = task.get("domain", "retail")
        task_id = task.get("id", task.get("task_id", "?"))

        logger.info("[%d/%d] Evaluating %s (domain=%s)", i + 1, n_total, task_id, domain)

        # Fresh DB + tools per task (no cross-task state leakage)
        executor, schemas = _build_tool_executor(domain)
        agent = AgentRunner(tools=schemas, tool_executor=executor)
        evaluator = InteractiveEvaluator(agent=agent, max_rounds=max_rounds)

        t0 = time.perf_counter()
        report = evaluator.evaluate([task])
        elapsed = time.perf_counter() - t0

        tr = report.results[0]  # single task
        d = tr.to_dict()
        d["elapsed_sec"] = round(elapsed, 1)
        all_results.append(d)

        # Trajectory
        traj = {
            "task_id": task_id,
            "domain": domain,
            "result": tr.to_dict(),
            "conversation": tr.conversation,
        }
        trajectories.append(traj)

        status_icon = "OK" if tr.completed else ("X" if tr.status == "give_up" else "~")
        logger.info(
            "  %s  rounds=%d  satisfaction=%.2f  elapsed=%.1fs",
            status_icon,
            tr.rounds,
            tr.satisfaction,
            elapsed,
        )

    # Aggregate report
    report_dict = _aggregate(all_results)
    return report_dict, trajectories


def _aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"total_tasks": 0}

    completed = [r for r in results if r["completed"]]
    n_completed = len(completed)
    avg_rounds = sum(r["rounds"] for r in results) / n
    first_try = sum(1 for r in results if r.get("first_try")) / n
    recovery = (
        sum(1 for r in completed if r["rounds"] > 1) / n_completed if n_completed else 0.0
    )
    abandoned = sum(1 for r in results if r["status"] == "give_up") / n
    avg_satisfaction = sum(r["satisfaction"] for r in results) / n

    # Per-domain breakdown
    domain_stats = defaultdict(lambda: {"total": 0, "completed": 0, "avg_rounds": 0.0, "rounds_sum": 0})
    for r in results:
        ds = domain_stats[r["domain"]]
        ds["total"] += 1
        if r["completed"]:
            ds["completed"] += 1
        ds["rounds_sum"] += r["rounds"]
    for d, ds in domain_stats.items():
        ds["completion_rate"] = round(ds["completed"] / ds["total"], 3) if ds["total"] else 0
        ds["avg_rounds"] = round(ds["rounds_sum"] / ds["total"], 2) if ds["total"] else 0
        del ds["rounds_sum"]

    return {
        "total_tasks": n,
        "completed_tasks": n_completed,
        "completion_rate": round(n_completed / n, 3) if n else 0,
        "avg_rounds": round(avg_rounds, 2),
        "first_try_rate": round(first_try, 3),
        "recovery_rate": round(recovery, 3),
        "abandonment_rate": round(abandoned, 3),
        "avg_satisfaction": round(avg_satisfaction, 3),
        "by_domain": dict(domain_stats),
    }


def _print_eval_report(report: dict):
    print()
    print("=" * 60)
    print("  INTERACTIVE EVALUATION REPORT (BASELINE)")
    print("=" * 60)
    print(f"  Tasks evaluated:    {report.get('total_tasks', 0)}")
    print(f"  Completed:          {report.get('completed_tasks', 0)} "
          f"({report.get('completion_rate', 0):.1%})")
    print(f"  Avg rounds:         {report.get('avg_rounds', 0)}")
    print(f"  First-try rate:     {report.get('first_try_rate', 0):.1%}")
    print(f"  Recovery rate:      {report.get('recovery_rate', 0):.1%}")
    print(f"  Abandonment rate:   {report.get('abandonment_rate', 0):.1%}")
    print(f"  Avg satisfaction:   {report.get('avg_satisfaction', 0):.3f}")
    by_domain = report.get("by_domain", {})
    if by_domain:
        print()
        print(f"  {'Domain':<10} {'Tasks':>5} {'Completed':>10} {'Rate':>8} {'AvgRnd':>6}")
        print(f"  {'-'*10} {'-'*5} {'-'*10} {'-'*8} {'-'*6}")
        for domain, ds in sorted(by_domain.items()):
            print(
                f"  {domain:<10} {ds['total']:>5} {ds['completed']:>10} "
                f"{ds['completion_rate']:>7.1%} {ds['avg_rounds']:>6}"
            )
    print("=" * 60)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train/test split & interactive baseline evaluation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--max-rounds", type=int, default=8, help="Max rounds per task")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--eval-limit", type=int, default=0,
                        help="Max test tasks to evaluate (0 = all)")
    parser.add_argument("--step1-only", action="store_true",
                        help="Only run Step 1 (split), skip evaluation")
    parser.add_argument("--input", type=str, default=None,
                        help="Override input prompts file")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else DATA_DIR / "grpo_prompts.jsonl"
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1: Split
    # ------------------------------------------------------------------

    logger.info("=== Step 1: Clean train/test split ===")
    logger.info("Loading prompts from %s", input_path)
    all_prompts = load_prompts(input_path)
    logger.info("Loaded %d prompts", len(all_prompts))

    train, test = split_prompts(all_prompts, test_ratio=args.test_ratio, seed=args.seed)

    train_path = DATA_DIR / "train_prompts.jsonl"
    test_path = DATA_DIR / "test_prompts.jsonl"

    write_jsonl(train_path, train)
    write_jsonl(test_path, test)

    _print_split_report(all_prompts, train, test)

    if args.step1_only:
        logger.info("Step 1 complete (--step1-only).")
        return

    # ------------------------------------------------------------------
    # Step 2: Evaluate
    # ------------------------------------------------------------------

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print()
        print("=" * 60)
        print("  WARNING: DEEPSEEK_API_KEY not set.")
        print("  Step 2 (interactive evaluation) requires this environment variable.")
        print("  Train/test split files have been written.")
        print("  Re-run with DEEPSEEK_API_KEY to run the evaluation.")
        print("=" * 60)
        print()
        return

    logger.info("=== Step 2: Interactive evaluation ===")
    logger.info("Evaluating %d test prompts (max %d rounds/task)",
                len(test) if args.eval_limit == 0 else min(args.eval_limit, len(test)),
                args.max_rounds)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    report_dict, trajectories = run_evaluation(
        test, max_rounds=args.max_rounds, limit=args.eval_limit,
    )

    report_path = EVAL_DIR / "baseline_report.json"
    traj_path = EVAL_DIR / "baseline_trajectories.jsonl"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    logger.info("Wrote report to %s", report_path)

    with open(traj_path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    logger.info("Wrote %d trajectories to %s", len(trajectories), traj_path)

    _print_eval_report(report_dict)


if __name__ == "__main__":
    main()
