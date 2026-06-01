#!/usr/bin/env python
"""
Phase 3: 初期训练闭环 (Initial Training Closed Loop)

End-to-end pipeline that integrates rubric-based GRPO training with
correction-rate evaluation. Runs the full cycle:

  1. Load rubrics + training data
  2. Pre-training correction rate baseline
  3. Train (rubric-scored GRPO via serve_ppo)
  4. Post-training correction rate evaluation
  5. Δ纠错率 report
  6. (Optional) Update rubrics from new conversation logs

Usage:
    python scripts/run_phase3_pipeline.py \
        --train-data data/training_pairs.jsonl \
        --test-data data/test_eval/training_pairs.jsonl \
        --rubrics data/rubrics.json \
        --model-server http://localhost:8000/v1

Or as a dry-run (no API calls):
    python scripts/run_phase3_pipeline.py --dry-run --stats-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase3_pipeline")


# ---------------------------------------------------------------------------
# Step 1: Ensure rubrics exist
# ---------------------------------------------------------------------------

def ensure_rubrics(
    rubrics_path: str = "data/rubrics.json",
    rubric_seeds_path: str = "data/rubric_seeds.json",
    api_key: str = "",
    simple: bool = False,
) -> list:
    """Ensure rubrics file exists. Generate if missing."""
    rubrics_file = Path(rubrics_path)
    if rubrics_file.exists():
        with open(rubrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        active = [r for r in data if r.get("状态") == "活跃"]
        logger.info("Loaded %d active rubrics from %s", len(active), rubrics_path)
        if active:
            return active

    # Generate from rubric seeds
    seeds_file = Path(rubric_seeds_path)
    if not seeds_file.exists():
        logger.warning("No rubric seeds found at %s — rubrics cannot be generated", rubric_seeds_path)
        return []

    with open(seeds_file, "r", encoding="utf-8") as f:
        seeds = json.load(f)

    logger.info("Generating rubrics from %d seeds...", len(seeds))

    from trainable_openclaw.evaluation.feedback import FeedbackAnalyzer
    from trainable_openclaw.evaluation.rubric import RubricGenerator

    analyzer = FeedbackAnalyzer(api_key=api_key)
    patterns = analyzer.analyze_simple(seeds)

    gen = RubricGenerator(api_key=api_key)
    if simple:
        rubrics = gen.generate_simple(patterns)
    else:
        rubrics = asyncio.run(gen.generate_all(patterns))

    logger.info("Generated %d rubrics → %s", len(rubrics), rubrics_path)
    return rubrics


# ---------------------------------------------------------------------------
# Step 2: Pre-training correction rate baseline
# ---------------------------------------------------------------------------

async def run_baseline_eval(
    test_data_path: str,
    model_server_url: str,
    api_key: str,
    max_prompts: int = 0,
) -> dict:
    """Run pre-training correction rate evaluation."""
    from trainable_openclaw.evaluation.correction_rate import (
        CorrectionRateEvaluator,
        load_test_prompts,
    )

    prompts = load_test_prompts(test_data_path, max_prompts=max_prompts)
    if not prompts:
        logger.warning("No test prompts found — skipping baseline")
        return {"status": "skipped", "reason": "no test prompts"}

    evaluator = CorrectionRateEvaluator(
        model_server_url=model_server_url,
        api_key=api_key,
    )

    logger.info("Running pre-training baseline on %d prompts...", len(prompts))
    result = await evaluator.evaluate(prompts)
    d = result.to_dict()
    d["status"] = "ok"
    d["num_prompts"] = len(prompts)
    return d


# ---------------------------------------------------------------------------
# Step 3: Training config generation
# ---------------------------------------------------------------------------

def generate_training_config(
    rubrics_path: str,
    train_data_path: str,
    api_key: str,
    output_path: str = "",
) -> str:
    """Generate Hydra config overrides for trajectory-based training."""
    config_lines = [
        "+trainer.trajectory.enabled=true",
        f"+trainer.trajectory.data_path={train_data_path}",
        f"+trainer.trajectory.rubrics_path={rubrics_path}",
        f"+trainer.trajectory.api_key={api_key}",
        "+trainer.trajectory.max_rubrics=8",
        "+trainer.trajectory.reward_mode=mean",
        "+trainer.idle_timeout=30",
        "+trainer.min_samples=0",
        "+trainer.train_steps_per_cycle=3",
        "+trainer.prompts_per_step=4",
    ]

    config_text = "\n".join(config_lines)
    if output_path:
        with open(output_path, "w") as f:
            f.write(config_text)
        logger.info("Training config written to %s", output_path)

    return config_text


# ---------------------------------------------------------------------------
# Step 4: Post-training evaluation + comparison
# ---------------------------------------------------------------------------

async def run_post_eval(
    test_data_path: str,
    model_server_url: str,
    api_key: str,
    pre_result: dict,
    max_prompts: int = 0,
) -> dict:
    """Run post-training evaluation and compute Δ."""
    from trainable_openclaw.evaluation.correction_rate import (
        CorrectionRateEvaluator,
        CorrectionRateResult,
        load_test_prompts,
    )

    prompts = load_test_prompts(test_data_path, max_prompts=max_prompts)
    if not prompts:
        return {"status": "skipped"}

    evaluator = CorrectionRateEvaluator(
        model_server_url=model_server_url,
        api_key=api_key,
    )

    logger.info("Running post-training evaluation on %d prompts...", len(prompts))

    # Build pre_result object if available
    pre_obj = None
    if pre_result.get("status") == "ok":
        pre_obj = CorrectionRateResult(
            total=pre_result["总数"],
            直接通过=pre_result["直接通过"],
            纠错后通过=pre_result["纠错后通过"],
            失败=pre_result["失败"],
            纠错率=pre_result["纠错率"],
        )

    result = await evaluator.evaluate_delta(prompts, pre_obj)
    d = result.to_dict()
    d["status"] = "ok"
    d["Δ纠错率"] = round(result.Δ纠错率, 4)
    d["pre_纠错率"] = round(result.pre_纠错率, 4)
    return d


# ---------------------------------------------------------------------------
# Step 6: Rubric update from logs
# ---------------------------------------------------------------------------

def update_rubrics_from_logs(
    db_path: str = "data/conversations.db",
    rubrics_path: str = "data/rubrics.json",
    api_key: str = "",
    min_new_messages: int = 50,
    simple: bool = True,
) -> dict:
    """Extract new corrections from conversation logs → update rubrics.

    Reads recent conversations from ConversationStore, runs User Sim
    Agent to identify corrections, extracts new rubric seeds, and
    merges them with existing rubrics.
    """
    from trainable_openclaw.logging.conversation_store import ConversationStore

    store = ConversationStore(db_path)
    stats = store.get_statistics()

    if stats["总消息数"] < min_new_messages:
        logger.info(
            "Only %d messages in store (need %d) — skipping rubric update",
            stats["总消息数"], min_new_messages,
        )
        return {"status": "skipped", "messages": stats["总消息数"]}

    # Get recent messages
    sessions = store.list_sessions(limit=50)
    logger.info("Analyzing %d sessions for new correction patterns...", len(sessions))

    # For each session with corrections, extract pattern
    new_seeds = []
    for sess in sessions:
        messages = store.get_messages(sess["id"])
        for i, msg in enumerate(messages):
            if msg["role"] != "user":
                continue
            content = msg.get("content", "")
            if "纠正" in content or "问题" in content or "错误" in content:
                # Extract correction seed
                new_seeds.append({
                    "维度": "log_extracted",
                    "频次": 1,
                    "示例": [{"纠错意见": content[:300]}],
                })

    if not new_seeds:
        logger.info("No new correction patterns found")
        return {"status": "no_new_patterns"}

    # Merge: generate new rubrics, combine with existing
    from trainable_openclaw.evaluation.feedback import FeedbackAnalyzer
    from trainable_openclaw.evaluation.rubric import RubricGenerator, RubricStore

    analyzer = FeedbackAnalyzer(api_key=api_key)
    patterns = analyzer.analyze_simple(new_seeds)

    store_rubrics = RubricStore(rubrics_path)
    gen = RubricGenerator(api_key=api_key, store=store_rubrics)

    new_count = 0
    for p in patterns:
        existing = store_rubrics.match(p.模式名称)
        if existing:
            store_rubrics.update(existing)
        else:
            rubric = gen._generate_fallback(p)
            new_count += 1

    logger.info(
        "Rubric update: %d new added, %d existing updated → %s",
        new_count, len(patterns) - new_count, rubrics_path,
    )
    return {"status": "ok", "new_rubrics": new_count, "total_rubrics": len(store_rubrics.rubrics)}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

async def run_full_pipeline(args):
    """Execute the complete Phase 3 closed loop."""
    results = {"steps": {}, "start_time": time.strftime("%Y-%m-%d %H:%M:%S")}

    # --- Step 1: Ensure rubrics ---
    logger.info("=" * 60)
    logger.info("Step 1: Ensuring rubrics exist...")
    rubrics = ensure_rubrics(
        rubrics_path=args.rubrics,
        rubric_seeds_path=args.rubric_seeds,
        api_key=args.api_key,
        simple=args.simple,
    )
    results["steps"]["ensure_rubrics"] = {
        "status": "ok", "active_rubrics": len(rubrics),
    }
    if not rubrics:
        logger.error("No rubrics available — cannot proceed")
        return results

    # --- Step 2: Pre-training baseline ---
    logger.info("=" * 60)
    logger.info("Step 2: Pre-training correction rate baseline...")
    if args.skip_eval:
        results["steps"]["pre_baseline"] = {"status": "skipped"}
    else:
        pre = await run_baseline_eval(
            test_data_path=args.test_data,
            model_server_url=args.model_server,
            api_key=args.api_key,
            max_prompts=args.max_test_prompts,
        )
        results["steps"]["pre_baseline"] = pre
        if pre.get("status") == "ok":
            print(f"\n  Pre-training 纠错率: {pre['纠错率']:.2%} "
                  f"({pre['直接通过']}/{pre['总数']} direct pass)\n")

    # --- Step 3: Training config ---
    logger.info("=" * 60)
    logger.info("Step 3: Training configuration...")
    config_text = generate_training_config(
        rubrics_path=args.rubrics,
        train_data_path=args.train_data,
        api_key=args.api_key,
        output_path=args.config_output,
    )
    results["steps"]["training_config"] = {"status": "ok"}
    print("Training config overrides:")
    for line in config_text.split("\n"):
        print(f"  {line}")
    print()

    # --- Step 4: Training (delegated to serve_ppo) ---
    logger.info("=" * 60)
    logger.info("Step 4: Training (requires GPU server running serve_ppo)...")
    if args.skip_training:
        results["steps"]["training"] = {"status": "skipped"}
        logger.info("Training skipped (--skip-training)")
    else:
        logger.info(
            "Start serve_ppo with the config above. Training will trigger "
            "automatically on idle timeout."
        )
        logger.info(
            "Example: python -m verl.trainer.serve_ppo "
            "config=ppo_trainer "
            "actor_rollout_ref.model.path=/path/to/model "
            "%s",
            " ".join(f"\"{l}\"" for l in config_text.split("\n")),
        )
        results["steps"]["training"] = {
            "status": "pending_manual",
            "note": "Start serve_ppo with the generated config",
        }

    # --- Step 5: Post-training evaluation ---
    logger.info("=" * 60)
    logger.info("Step 5: Post-training correction rate evaluation...")
    if args.skip_eval or args.skip_training:
        results["steps"]["post_eval"] = {"status": "skipped"}
    else:
        pre_result = results["steps"].get("pre_baseline", {})
        post = await run_post_eval(
            test_data_path=args.test_data,
            model_server_url=args.model_server,
            api_key=args.api_key,
            pre_result=pre_result,
            max_prompts=args.max_test_prompts,
        )
        results["steps"]["post_eval"] = post

        # Print comparison
        if post.get("status") == "ok":
            print("\n" + "=" * 60)
            print("  Phase 3 Results: 纠错率 Comparison")
            print("=" * 60)
            pre_rate = post.get("pre_纠错率", 0)
            post_rate = post.get("纠错率", 0)
            delta = post.get("Δ纠错率", 0)
            print(f"  Pre-training:  {pre_rate:.2%}")
            print(f"  Post-training: {post_rate:.2%}")
            print(f"  Δ纠错率:       {delta:+.2%}")
            if delta < 0:
                print(f"  Verdict:       IMPROVEMENT ({abs(delta):.1%} fewer corrections)")
            elif delta > 0:
                print(f"  Verdict:       REGRESSION ({delta:.1%} more corrections)")
            else:
                print(f"  Verdict:       NO CHANGE")
            print("=" * 60)

    # --- Step 6: Rubric update from logs ---
    logger.info("=" * 60)
    logger.info("Step 6: Updating rubrics from conversation logs...")
    if args.skip_rubric_update:
        results["steps"]["rubric_update"] = {"status": "skipped"}
    else:
        update_result = update_rubrics_from_logs(
            db_path=args.db_path,
            rubrics_path=args.rubrics,
            api_key=args.api_key,
            simple=args.simple,
        )
        results["steps"]["rubric_update"] = update_result

    # --- Summary ---
    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info("Phase 3 pipeline complete.")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Results written to %s", args.output)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: 初期训练闭环 — Rubric-based GRPO training pipeline",
    )
    # Data paths
    parser.add_argument("--train-data", default="data/training_pairs.jsonl")
    parser.add_argument("--test-data", default="data/test_eval/training_pairs.jsonl")
    parser.add_argument("--rubrics", default="data/rubrics.json")
    parser.add_argument("--rubric-seeds", default="data/rubric_seeds.json")
    parser.add_argument("--db-path", default="data/conversations.db")
    # API
    parser.add_argument("--api-key", default="", help="DeepSeek API key (or $DEEPSEEK_API_KEY)")
    parser.add_argument("--model-server", default="http://localhost:8000/v1")
    # Modes
    parser.add_argument("--simple", action="store_true", help="Use simple (template) rubric generation")
    parser.add_argument("--skip-eval", action="store_true", help="Skip correction rate evaluation")
    parser.add_argument("--skip-training", action="store_true", help="Skip training (eval only)")
    parser.add_argument("--skip-rubric-update", action="store_true", help="Skip rubric update from logs")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without execution")
    # Limits
    parser.add_argument("--max-test-prompts", type=int, default=0, help="Max test prompts (0=all)")
    # Output
    parser.add_argument("--config-output", default="", help="Write training config to file")
    parser.add_argument("--output", "-o", default="", help="Write pipeline results to JSON")

    args = parser.parse_args()

    # Resolve API key
    if not args.api_key:
        args.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not args.api_key:
        logger.warning("No API key provided — rubric generation and evaluation will fail")
        logger.warning("Set DEEPSEEK_API_KEY environment variable or pass --api-key")

    if args.dry_run:
        print("Dry run — would execute:")
        print(f"  1. Ensure rubrics: {args.rubrics}")
        print(f"  2. Pre-training baseline: {args.test_data} ({args.max_test_prompts or 'all'} prompts)")
        print(f"  3. Generate training config")
        print(f"  4. Train (rubric-scored GRPO)")
        print(f"  5. Post-training evaluation + Δ纠错率")
        print(f"  6. Update rubrics from logs: {args.db_path}")
        return

    asyncio.run(run_full_pipeline(args))


if __name__ == "__main__":
    main()
