#!/usr/bin/env python3
"""Baseline evaluation: Qwen3.5-4B via nanobot + DeepSeek simulated user.
Runs retail Test 23 tasks (37 entries) — used ONLY for final evaluation.
Usage: python /tmp/baseline_eval.py [--limit N] [--output PATH]
"""

import json, os, sys, time, argparse
import requests

NANOBOT_URL = "http://localhost:8900"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"

sys.path.insert(0, "/data/wangye/trainable-openclaw")
from trainable_openclaw.evaluation.simulated_user import SimulatedUser


def load_retail_tasks(path):
    tasks = []
    with open(path) as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                if t.get("domain") == "retail":
                    tasks.append(t)
    return tasks


def call_nanobot(message, session_id, timeout=600):
    """Send message to nanobot, get final text response after tool calling."""
    resp = requests.post(
        f"{NANOBOT_URL}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": message}], "session_id": session_id},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def evaluate_one(task, max_rounds=10):
    sim = SimulatedUser(
        task, model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
    session_id = f"baseline-{task['id']}"
    conversation = []
    user_msg = sim.initial_message

    for r in range(1, max_rounds + 1):
        try:
            agent_msg = call_nanobot(user_msg, session_id)
        except Exception as exc:
            return {
                "task_id": task["id"], "domain": task["domain"],
                "completed": False, "rounds": r, "satisfaction": 0.0,
                "status": "agent_error", "error": str(exc),
            }

        conversation.append({"role": "agent", "content": agent_msg})
        feedback = sim.respond(agent_msg, [])
        conversation.append({"role": "user", "content": feedback.message})

        if feedback.status == "complete":
            return {
                "task_id": task["id"], "domain": task["domain"],
                "completed": True, "rounds": r,
                "satisfaction": feedback.satisfaction, "status": "complete",
            }
        elif feedback.status == "give_up":
            return {
                "task_id": task["id"], "domain": task["domain"],
                "completed": False, "rounds": r,
                "satisfaction": feedback.satisfaction, "status": "give_up",
            }
        user_msg = feedback.message

    return {
        "task_id": task["id"], "domain": task["domain"],
        "completed": False, "rounds": max_rounds, "satisfaction": 0.0,
        "status": "timeout",
    }


def build_report(results):
    total = len(results)
    if total == 0:
        return {}
    n_completed = sum(1 for r in results if r["completed"])
    avg_rounds = sum(r["rounds"] for r in results) / total
    first_try = sum(1 for r in results if r["completed"] and r["rounds"] == 1) / total
    recovery = sum(1 for r in results if r["completed"] and r["rounds"] > 1) / n_completed if n_completed else 0
    abandoned = sum(1 for r in results if r["status"] == "give_up") / total
    return {
        "total_tasks": total,
        "completed_tasks": n_completed,
        "completion_rate": round(n_completed / total, 4),
        "avg_rounds": round(avg_rounds, 2),
        "first_try_rate": round(first_try, 4),
        "recovery_rate": round(recovery, 4),
        "abandonment_rate": round(abandoned, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max tasks to evaluate")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--output", type=str, default="/data/wangye/trainable-openclaw/evaluation_results/baseline_4b_retail.json")
    parser.add_argument("--test-file", type=str, default="/data/wangye/trainable-openclaw/data/tau_bench/test_prompts_augmented.jsonl")
    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    # Verify nanobot
    try:
        r = requests.get(f"{NANOBOT_URL}/health", timeout=5)
        print(f"nanobot: {r.json()}")
    except Exception as exc:
        print(f"ERROR: Cannot reach nanobot at {NANOBOT_URL}: {exc}")
        sys.exit(1)

    tasks = load_retail_tasks(args.test_file)
    if args.limit:
        tasks = tasks[:args.limit]

    print(f"Baseline eval: {len(tasks)} retail tasks")
    print(f"Agent: Qwen3.5-4B via nanobot @ {NANOBOT_URL}")
    print(f"SimUser: {DEEPSEEK_MODEL}")
    print()

    results = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] {task['id']} ... ", end="", flush=True)
        t_start = time.time()
        r = evaluate_one(task, max_rounds=args.max_rounds)
        elapsed = time.time() - t_start
        results.append(r)
        status = "OK" if r["completed"] else r["status"].upper()
        print(f"{status} | rounds={r['rounds']} | sat={r['satisfaction']} | {elapsed:.0f}s")

    total_time = time.time() - t0
    report = build_report(results)

    print(f"\n{'='*60}")
    print(f"  BASELINE EVALUATION REPORT")
    print(f"{'='*60}")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"  total_time: {total_time:.0f}s")
    print(f"{'='*60}")

    # Save results
    output = {
        "config": {
            "model": "Qwen3.5-4B",
            "backend": "nanobot + vllm",
            "sim_user_model": DEEPSEEK_MODEL,
            "max_rounds": args.max_rounds,
            "test_file": args.test_file,
        },
        "report": report,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
