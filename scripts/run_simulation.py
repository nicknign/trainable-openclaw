#!/usr/bin/env python3
"""
Run the Phase 1.5 S2 simulation pipeline.

Takes seed prompts (from S1) and runs User Sim Agent ↔ Qwen3-4B
multi-turn correction dialogues to generate training data.

Usage:
    # Mock mode (DeepSeek simulates both sides, no GPU needed)
    python scripts/run_simulation.py --seed-file data/seed_prompts.jsonl --mock

    # Live mode (real Qwen3-4B API)
    python scripts/run_simulation.py --seed-file data/seed_prompts.jsonl --no-mock --qwen-url http://localhost:8000

    # Dry run (validate structure without API calls)
    python scripts/run_simulation.py --seed-file data/seed_prompts.jsonl --dry-run

API key:
    Set DEEPSEEK_API_KEY environment variable, or pass --api-key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.5 S2: User Sim Correction Dialogue Pipeline"
    )
    parser.add_argument(
        "--seed-file", default="data/seed_prompts.jsonl",
        help="Seed prompts JSONL from S1 (default: data/seed_prompts.jsonl)",
    )
    parser.add_argument(
        "--output", default="data/correction_trajectories.jsonl",
        help="Output trajectories JSONL (default: data/correction_trajectories.jsonl)",
    )
    parser.add_argument(
        "--api-key", default="",
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url", default="https://api.deepseek.com",
        help="DeepSeek API base URL (default: https://api.deepseek.com)",
    )
    parser.add_argument(
        "--model", default="deepseek-v4-flash",
        help="Model for User Sim Agent (default: deepseek-v4-flash)",
    )
    parser.add_argument(
        "--mock", action="store_true", default=True,
        help="Use DeepSeek to simulate Qwen3-4B responses (default: True)",
    )
    parser.add_argument(
        "--no-mock", dest="mock", action="store_false",
        help="Use real Qwen3-4B API instead of mock",
    )
    parser.add_argument(
        "--qwen-url", default="http://localhost:8000",
        help="Qwen3-4B API URL when --no-mock (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--max-prompts", type=int, default=0,
        help="Max prompts to process (0 = all, useful for testing)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=5,
        help="Max correction turns per prompt (default: 5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate structure without making API calls",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Print stats from existing output file",
    )
    args = parser.parse_args()

    # Stats-only mode
    if args.stats_only:
        print_stats(args.output)
        return

    # Validate seed file
    if not os.path.exists(args.seed_file):
        print(f"ERROR: Seed file not found: {args.seed_file}")
        print(f"  Run S1 first: python scripts/extract_seed_prompts.py")
        sys.exit(1)

    # Dry run mode
    if args.dry_run:
        run_dry_run(args.seed_file)
        return

    # Resolve API key
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: No API key provided.")
        print("  Set DEEPSEEK_API_KEY environment variable or pass --api-key")
        print("  Or use --dry-run to validate without API calls")
        sys.exit(1)

    # Run simulation
    from trainable_openclaw.simulation.engine import run_seed_prompts

    asyncio.run(run_seed_prompts(
        seed_file=args.seed_file,
        output_file=args.output,
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        mock=args.mock,
        max_prompts=args.max_prompts,
        max_turns=args.max_turns,
        qwen_url=args.qwen_url,
    ))

    print_stats(args.output)


def run_dry_run(seed_file: str) -> None:
    """Validate seed file structure without API calls."""
    print(f"Dry-run validation of {seed_file}")
    print()

    seeds = []
    with open(seed_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seeds.append(json.loads(line))

    print(f"Total seeds: {len(seeds)}")

    # Check required fields
    required = ["prompt", "category"]
    missing = []
    for i, s in enumerate(seeds):
        for field in required:
            if field not in s:
                missing.append((i, field))
    if missing:
        print(f"ERROR: Missing fields in {len(missing)} seeds:")
        for i, field in missing[:10]:
            print(f"  Seed {i}: missing '{field}'")
    else:
        print("All seeds have required fields: OK")

    # Category distribution
    from collections import Counter
    cats = Counter(s.get("category", "unknown") for s in seeds)
    print(f"\nCategory distribution ({len(cats)} categories):")
    for cat, cnt in cats.most_common():
        print(f"  {cat:<30s} {cnt:>5d}")

    # Persona mapping
    from trainable_openclaw.simulation.user_sim import select_persona
    persona_counts = Counter()
    for s in seeds:
        persona_counts[select_persona(s.get("category", "general"))] += 1
    print(f"\nPersona assignment:")
    for p, cnt in persona_counts.most_common():
        print(f"  {p:<30s} {cnt:>5d}")

    # Prompt length stats
    lengths = [s.get("char_count", len(s["prompt"])) for s in seeds]
    print(f"\nPrompt length stats:")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.0f}")
    print()
    print("Dry run complete. Use without --dry-run to execute with API.")


def print_stats(output_file: str) -> None:
    """Print statistics from trajectory output file."""
    if not os.path.exists(output_file):
        print(f"No output file yet: {output_file}")
        return

    trajectories = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))

    from collections import Counter
    verdicts = Counter(t.get("final_verdict", "unknown") for t in trajectories)
    personas = Counter(t.get("persona", "unknown") for t in trajectories)

    n = len(trajectories)
    avg_corrections = sum(t.get("correction_count", 0) for t in trajectories) / n if n else 0
    avg_turns = sum(t.get("total_turns", 0) for t in trajectories) / n if n else 0

    direct_pass = verdicts.get("direct_pass", 0)
    corrected = verdicts.get("corrected", 0)

    print(f"\n{'='*60}")
    print(f"  Simulation Results")
    print(f"{'='*60}")
    print(f"  Total trajectories:  {n}")
    print(f"  Avg corrections:     {avg_corrections:.2f}")
    print(f"  Avg turns:           {avg_turns:.2f}")
    print(f"  Direct pass:         {direct_pass} ({direct_pass/n*100:.1f}%)" if n else "")
    print(f"  Corrected pass:      {corrected} ({corrected/n*100:.1f}%)" if n else "")
    print(f"  Correction close rate: {(direct_pass + corrected)/n*100:.1f}%" if n else "")

    print(f"\n  Verdict distribution:")
    for v, c in verdicts.most_common():
        print(f"    {v:<20s} {c:>5d}")

    print(f"\n  Persona distribution:")
    for p, c in personas.most_common():
        print(f"    {p:<20s} {c:>5d}")


if __name__ == "__main__":
    main()
