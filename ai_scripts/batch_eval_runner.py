"""Batch eval runner — concurrent, with early-stop and 5-dimension scoring.

Usage:
    python ai_scripts/batch_eval_runner.py                          # defaults
    python ai_scripts/batch_eval_runner.py --max-tasks 30 --concurrency 4
    python ai_scripts/batch_eval_runner.py --max-rounds 7 --no-early-stop
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainable_openclaw.evaluation.simulated_user import SimulatedUser
from trainable_openclaw.evaluation.interactive_eval import TaskResult
import requests

NANOBOT_URL = os.environ.get("NANOBOT_URL", "http://localhost:8900")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"
PROMPTS_PATH = Path(__file__).resolve().parent.parent / "data" / "tau_bench" / "test_prompts_augmented.jsonl"

_print_lock = Lock()
_results_lock = Lock()

# Configurable globals
MAX_ROUNDS = 7
HTTP_TIMEOUT = 300
EARLY_STOP = True
EARLY_STOP_CONSECUTIVE = 3
EARLY_STOP_SAT_THRESHOLD = 0.3


def load_tasks():
    tasks = []
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def call_nanobot(message, session_id):
    resp = requests.post(
        f"{NANOBOT_URL}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": message}],
            "session_id": session_id,
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str):
    with _print_lock:
        print(f"[{_ts()}] {msg}", flush=True)


def evaluate_one(task, task_idx):
    task_id = task.get("id", "?")
    domain = task.get("domain", "")
    _log(f"T{task_idx} START: {task_id} ({domain})")

    sim_user = SimulatedUser(
        task,
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        mode="teacher",
    )

    session_id = f"eval-{task_id}"
    conversation = []
    user_msg = sim_user.initial_message
    low_sat_streak = 0

    for r in range(1, MAX_ROUNDS + 1):
        t0 = time.time()
        try:
            agent_msg = call_nanobot(user_msg, session_id)
        except Exception as exc:
            _log(f"T{task_idx} R{r}: AGENT ERROR: {exc}")
            return TaskResult(
                task_id=task_id, domain=domain,
                completed=False, rounds=r, satisfaction=0.0,
                status="agent_error", conversation=conversation,
            )
        dt = time.time() - t0
        conversation.append({"role": "agent", "content": agent_msg})
        _log(f"T{task_idx} R{r}: agent ({dt:.0f}s, {len(agent_msg)}c)")

        try:
            feedback = sim_user.respond(agent_msg, [])
        except Exception as exc:
            _log(f"T{task_idx} R{r}: USER ERROR (DeepSeek): {exc}")
            return TaskResult(
                task_id=task_id, domain=domain,
                completed=False, rounds=r, satisfaction=0.0,
                status="agent_error", conversation=conversation,
            )
        conversation.append({"role": "user", "content": feedback.message})
        _log(f"T{task_idx} R{r}: user {feedback.status} sat={feedback.satisfaction}")

        if feedback.status == "complete":
            _log(f"T{task_idx} DONE: complete r={r} sat={feedback.satisfaction}")
            return TaskResult(
                task_id=task_id, domain=domain,
                completed=True, rounds=r, satisfaction=feedback.satisfaction,
                status="complete", conversation=conversation,
            )
        elif feedback.status == "give_up":
            _log(f"T{task_idx} DONE: give_up r={r}")
            return TaskResult(
                task_id=task_id, domain=domain,
                completed=False, rounds=r, satisfaction=feedback.satisfaction,
                status="give_up", conversation=conversation,
            )

        # Early stop: N consecutive low-satisfaction rounds
        if EARLY_STOP and feedback.satisfaction < EARLY_STOP_SAT_THRESHOLD:
            low_sat_streak += 1
            if low_sat_streak >= EARLY_STOP_CONSECUTIVE:
                _log(f"T{task_idx} EARLY STOP: {low_sat_streak} consecutive sat<{EARLY_STOP_SAT_THRESHOLD}")
                return TaskResult(
                    task_id=task_id, domain=domain,
                    completed=False, rounds=r, satisfaction=feedback.satisfaction,
                    status="give_up", conversation=conversation,
                )
        else:
            low_sat_streak = 0

        user_msg = feedback.message

    _log(f"T{task_idx} DONE: timeout ({MAX_ROUNDS}r)")
    return TaskResult(
        task_id=task_id, domain=domain,
        completed=False, rounds=MAX_ROUNDS, satisfaction=0.0,
        status="timeout", conversation=conversation,
    )


def compute_score(results: dict) -> dict:
    tasks = list(results.values()) if isinstance(results, dict) else list(results)
    n = len(tasks)
    if n == 0:
        return {"error": "no tasks"}

    completed = sum(1 for t in tasks if t["completed"])
    avg_sat = sum(t["satisfaction"] for t in tasks) / n
    first_try = sum(1 for t in tasks if t["completed"] and t["rounds"] <= 3) / n
    reliability = 1.0 - sum(1 for t in tasks if t["status"] == "agent_error") / n
    persistence = 1.0 - sum(1 for t in tasks if t["status"] == "give_up") / n

    completion_rate = completed / n
    score = 0.4 * completion_rate + 0.3 * avg_sat + 0.1 * first_try + 0.1 * reliability + 0.1 * persistence

    def domain_stats(domain_tasks):
        if not domain_tasks:
            return {}
        dn = len(domain_tasks)
        dc = sum(1 for t in domain_tasks if t["completed"])
        return {
            "count": dn,
            "completed": dc,
            "completion_rate": round(dc / dn, 4),
            "avg_satisfaction": round(sum(t["satisfaction"] for t in domain_tasks) / dn, 4),
            "first_try_rate": round(sum(1 for t in domain_tasks if t["completed"] and t["rounds"] <= 3) / dn, 4),
        }

    airline = [t for t in tasks if t.get("domain", "").startswith("airline")]
    retail = [t for t in tasks if t.get("domain", "").startswith("retail")]

    return {
        "total_tasks": n,
        "completed": completed,
        "completion_rate": round(completion_rate, 4),
        "avg_satisfaction": round(avg_sat, 4),
        "first_try_rate": round(first_try, 4),
        "reliability": round(reliability, 4),
        "persistence": round(persistence, 4),
        "overall_score": round(score, 4),
        "by_domain": {
            "airline": domain_stats(airline),
            "retail": domain_stats(retail),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Batch tau-bench evaluation")
    parser.add_argument("--max-tasks", type=int, default=30, help="Number of tasks to run (default 30)")
    parser.add_argument("--max-rounds", type=int, default=7, help="Max conversation rounds (default 7)")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent tasks (default 4)")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stop on low satisfaction")
    parser.add_argument("--http-timeout", type=int, default=300, help="HTTP timeout seconds (default 300)")
    parser.add_argument("--domain", type=str, default="", help="Only run tasks from this domain (retail/airline)")
    args = parser.parse_args()

    global MAX_ROUNDS, HTTP_TIMEOUT, EARLY_STOP
    MAX_ROUNDS = args.max_rounds
    HTTP_TIMEOUT = args.http_timeout
    EARLY_STOP = not args.no_early_stop

    print(f"Starting eval: {args.max_tasks} tasks, {args.max_rounds} max rounds, "
          f"{args.concurrency} concurrent, early_stop={EARLY_STOP}", flush=True)
    print(f"Nanobot: {NANOBOT_URL}", flush=True)
    print(f"SimUser: {DEEPSEEK_MODEL} @ {DEEPSEEK_BASE_URL}", flush=True)

    try:
        r = requests.get(f"{NANOBOT_URL}/health", timeout=5)
        print(f"nanobot health: {r.json()}", flush=True)
    except Exception as exc:
        print(f"ERROR: Cannot reach nanobot: {exc}", flush=True)
        sys.exit(1)

    tasks = load_tasks()
    if args.domain:
        tasks = [t for t in tasks if t.get("domain") == args.domain]
        print(f"Filtered to {args.domain}: {len(tasks)} tasks", flush=True)
    tasks = tasks[:args.max_tasks]
    n_tasks = len(tasks)
    print(f"Running {n_tasks} tasks with {args.concurrency} workers", flush=True)

    results = {}
    t0_total = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(evaluate_one, task, i): i
            for i, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            i = futures[future]
            tid = tasks[i]["id"]
            domain = tasks[i].get("domain", "")
            try:
                tr = future.result()
            except Exception as exc:
                _log(f"T{i} CRASH: unhandled exception in thread: {exc}")
                tr = TaskResult(
                    task_id=tid, domain=domain,
                    completed=False, rounds=0, satisfaction=0.0,
                    status="agent_error", conversation=[],
                )
            elapsed = time.time() - t0_total
            status = "[OK]" if tr.completed else "[FAIL]"
            with _results_lock:
                results[f"task_{i}_{tid}"] = {
                    "completed": tr.completed,
                    "rounds": tr.rounds,
                    "satisfaction": tr.satisfaction,
                    "status": tr.status,
                    "domain": tr.domain,
                }
            done = len(results)
            _log(f"PROGRESS: {done}/{n_tasks} ({done*100//n_tasks}%), {status} "
                 f"{tid}: completed={tr.completed} r={tr.rounds} sat={tr.satisfaction} ({elapsed:.0f}s elapsed)")

            # Checkpoint every 5 tasks
            if done % 5 == 0:
                with _results_lock:
                    ckpt = {"done": done, "total": n_tasks, "results_so_far": dict(results)}
                with open("/tmp/batch_eval_checkpoint.json", "w") as f:
                    json.dump(ckpt, f, indent=2)
                _log(f"CHECKPOINT: {done}/{n_tasks} saved")

    total_time = time.time() - t0_total

    scoring = compute_score(results)
    out = {"scoring": scoring, "config": {
        "max_rounds": MAX_ROUNDS, "max_tasks": args.max_tasks,
        "concurrency": args.concurrency, "early_stop": EARLY_STOP,
        "http_timeout": HTTP_TIMEOUT,
    }, "tasks": results}

    with open("/tmp/batch_eval_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== FINAL SCORES ===", flush=True)
    for k, v in scoring.items():
        if k != "by_domain":
            print(f"  {k}: {v}", flush=True)
    for domain, ds in scoring.get("by_domain", {}).items():
        if ds:
            print(f"  [{domain}] {ds['count']} tasks, completion={ds['completion_rate']}, "
                  f"avg_sat={ds['avg_satisfaction']}", flush=True)
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)
    print("Results: /tmp/batch_eval_results.json", flush=True)


if __name__ == "__main__":
    main()
