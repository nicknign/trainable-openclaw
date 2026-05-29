#!/usr/bin/env python3
"""
Extract seed prompts from LMSYS-Chat-1M dataset in ConversationStore.

Queries the LMSYS data already imported into data/conversations.db,
performs stratified sampling by category, filters low-quality prompts,
and outputs a JSONL file of seed prompts for the simulation pipeline (S1).

Usage:
    python scripts/extract_seed_prompts.py
    python scripts/extract_seed_prompts.py --db data/conversations.db --output data/seed_prompts.jsonl
    python scripts/extract_seed_prompts.py --max-per-category 400 --min-length 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from random import Random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainable_openclaw.logging.conversation_store import ConversationStore

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

# Minimum prompt length in characters (after stripping whitespace)
DEFAULT_MIN_LENGTH = 15

# Conversational noise patterns — prompts that are purely chat/openers/thanks.
# Matched case-insensitively after stripping.
CONVERSATIONAL_BLACKLIST = {
    "hello", "hi", "hey", "yo", "sup", "hola",
    "thanks", "thank you", "thx", "ty", "thank you very much",
    "ok", "okay", "k", "fine", "sure", "yes", "no", "nope", "yep",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how are you doing", "what's up", "whats up",
    "bye", "goodbye", "see you", "cya", "later",
    "nice", "cool", "great", "awesome",
    "lol", "lmao", "haha", "hehe",
    "...", "？", "？?", "。。。",
    "你好", "谢谢", "好的", "嗯", "哦",
}


def _is_conversational_noise(text: str) -> bool:
    """Return True if the prompt is purely conversational chit-chat."""
    cleaned = text.strip().lower().rstrip("!。？?!,.，")
    if not cleaned:
        return True
    if cleaned in CONVERSATIONAL_BLACKLIST:
        return True
    # Also catch single-character or purely punctuation
    if len(cleaned) <= 2 and cleaned.isascii():
        return any(c.isalpha() for c in cleaned)  # single letter like "k", "a"
    return False


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def query_lmsys_sessions(store: ConversationStore) -> list[dict]:
    """Return all LMSYS sessions with category metadata.

    Each result includes: session_id, category, conv_id, metadata (raw JSON).
    """
    rows = store.conn.execute(
        "SELECT id AS session_id, "
        "  json_extract(metadata, '$.category') AS category, "
        "  json_extract(metadata, '$.conv_id') AS conv_id, "
        "  metadata "
        "FROM sessions "
        "WHERE json_extract(metadata, '$.source') = 'lmsys_chat_1m_clean' "
        "ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


def get_first_user_message(store: ConversationStore, session_id: str) -> dict | None:
    """Return the first user message for a session, or None."""
    row = store.conn.execute(
        "SELECT id, content, created_at, metadata "
        "FROM messages "
        "WHERE session_id = ? AND role = 'user' "
        "ORDER BY created_at ASC LIMIT 1",
        (session_id,),
    ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Extraction & sampling
# ---------------------------------------------------------------------------


def extract_prompts(
    db_path: str,
    min_length: int = DEFAULT_MIN_LENGTH,
    rng_seed: int = 42,
) -> list[dict]:
    """Extract all valid LMSYS user prompts, grouped by category.

    Returns a list of prompt dicts, each with:
      prompt_text, category, session_id, conv_id, msg_id, char_count
    """
    store = ConversationStore(db_path)

    sessions = query_lmsys_sessions(store)
    print(f"Querying {len(sessions)} LMSYS sessions...")

    # Collect by category
    by_category: dict[str, list[dict]] = defaultdict(list)
    seen_texts: set[str] = set()
    skipped_short = 0
    skipped_noise = 0
    skipped_duplicate = 0
    skipped_no_user_msg = 0

    for i, sess in enumerate(sessions):
        if (i + 1) % 50000 == 0:
            print(f"  Processed {i + 1}/{len(sessions)} sessions...")

        msg = get_first_user_message(store, session_id=sess["session_id"])
        if msg is None:
            skipped_no_user_msg += 1
            continue

        text = msg["content"].strip()
        if len(text) < min_length:
            skipped_short += 1
            continue

        if _is_conversational_noise(text):
            skipped_noise += 1
            continue

        # Deduplicate by exact match (lowercased)
        dedup_key = text.lower()
        if dedup_key in seen_texts:
            skipped_duplicate += 1
            continue
        seen_texts.add(dedup_key)

        category = sess.get("category") or "unknown"
        by_category[category].append({
            "prompt": text,
            "category": category,
            "session_id": sess["session_id"],
            "conv_id": sess.get("conv_id"),
            "msg_id": msg["id"],
            "char_count": len(text),
        })

    store.close()

    print(f"\nFiltering summary:")
    print(f"  Skipped — no user message: {skipped_no_user_msg}")
    print(f"  Skipped — too short (<{min_length}): {skipped_short}")
    print(f"  Skipped — conversational noise: {skipped_noise}")
    print(f"  Skipped — duplicate (exact): {skipped_duplicate}")
    print(f"  Kept: {sum(len(v) for v in by_category.values())} prompts across {len(by_category)} categories")

    return by_category


def stratified_sample(
    by_category: dict[str, list[dict]],
    max_per_category: int = 400,
    rng_seed: int = 42,
) -> list[dict]:
    """Sample up to *max_per_category* prompts from each category.

    Uses deterministic shuffle so results are reproducible.
    """
    rng = Random(rng_seed)
    sampled: list[dict] = []

    for cat in sorted(by_category.keys()):
        pool = by_category[cat]
        rng.shuffle(pool)
        taken = pool[:max_per_category]
        sampled.extend(taken)

    return sampled


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def print_statistics(sampled: list[dict]) -> None:
    """Print extraction statistics."""
    by_category: dict[str, list[dict]] = defaultdict(list)
    for p in sampled:
        by_category[p["category"]].append(p)

    print(f"\n{'=' * 60}")
    print(f"  Seed Prompt Extraction Results")
    print(f"{'=' * 60}")
    print(f"  Total unique prompts: {len(sampled)}")
    print(f"  Categories: {len(by_category)}")
    print()

    # Per-category counts (sorted by count descending)
    cat_counts = [(cat, len(prompts)) for cat, prompts in by_category.items()]
    cat_counts.sort(key=lambda x: -x[1])

    print(f"  {'Category':<30s} {'Count':>6s}  {'Avg Len':>8s}")
    print(f"  {'-' * 47}")
    for cat, cnt in cat_counts:
        avg_len = sum(p["char_count"] for p in by_category[cat]) / cnt if cnt else 0
        print(f"  {cat:<30s} {cnt:>6d}  {avg_len:>7.1f}")

    print(f"  {'-' * 47}")
    total = sum(cnt for _, cnt in cat_counts)
    overall_avg = sum(p["char_count"] for p in sampled) / total if total else 0
    print(f"  {'TOTAL':<30s} {total:>6d}  {overall_avg:>7.1f}")
    print()

    # Example prompts from top 5 categories
    print(f"  Example prompts (top {min(5, len(cat_counts))} categories):")
    print()
    for cat, _ in cat_counts[:5]:
        examples = by_category[cat][:3]
        print(f"  [{cat}]")
        for ex in examples:
            preview = ex["prompt"][:120]
            if len(ex["prompt"]) > 120:
                preview += "..."
            print(f"    • {preview}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract seed prompts from LMSYS-Chat-1M"
    )
    parser.add_argument(
        "--db", default="data/conversations.db",
        help="Path to ConversationStore database",
    )
    parser.add_argument(
        "--output", default="data/seed_prompts.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--max-per-category", type=int, default=400,
        help="Maximum prompts to sample per category (default: 400)",
    )
    parser.add_argument(
        "--min-length", type=int, default=DEFAULT_MIN_LENGTH,
        help=f"Minimum prompt length in characters (default: {DEFAULT_MIN_LENGTH})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Print stats from existing output file, skip extraction",
    )
    args = parser.parse_args()

    if args.stats_only:
        if not os.path.exists(args.output):
            print(f"ERROR: Output file not found: {args.output}")
            sys.exit(1)
        with open(args.output, "r", encoding="utf-8") as f:
            sampled = [json.loads(line) for line in f if line.strip()]
        print_statistics(sampled)
        return

    if not os.path.exists(args.db):
        print(f"ERROR: Database not found: {args.db}")
        print(f"  Run 'python scripts/prepare_lmsys.py' first to import the LMSYS dataset.")
        sys.exit(1)

    print("Extracting seed prompts from LMSYS-Chat-1M...")
    print(f"  Database: {args.db}")
    print(f"  Max per category: {args.max_per_category}")
    print(f"  Min prompt length: {args.min_length}")
    print()

    # Step 1: Extract all valid prompts
    by_category = extract_prompts(
        db_path=args.db,
        min_length=args.min_length,
        rng_seed=args.seed,
    )

    # Step 2: Stratified sampling
    sampled = stratified_sample(
        by_category,
        max_per_category=args.max_per_category,
        rng_seed=args.seed,
    )

    # Step 3: Output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in sampled:
            line = {
                "prompt": entry["prompt"],
                "category": entry["category"],
                "session_id": entry["session_id"],
                "conv_id": entry["conv_id"],
                "char_count": entry["char_count"],
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\nOutput: {len(sampled)} prompts → {args.output}")

    # Step 4: Statistics
    print_statistics(sampled)


if __name__ == "__main__":
    main()
