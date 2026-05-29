#!/usr/bin/env python3
"""
OASST2 Data Preparation — download, parse, split, import, and evaluate.

Pipeline:
  1. Download OpenAssistant/oasst2 from HuggingFace
  2. Parse flat messages into conversation trees (multi-turn sessions)
  3. Split into train (80%) and test (20%) by message_tree_id
  4. Train set → import into ConversationStore (for B1/B2 rubric evolution)
  5. Test set → export as JSONL (for baseline evaluation)
  6. Print dataset statistics

Usage::

    python scripts/prepare_oasst2.py                    # full pipeline
    python scripts/prepare_oasst2.py --split-only       # split only, skip store import
    python scripts/prepare_oasst2.py --test-size 0.1    # 10% test split
    python scripts/prepare_oasst2.py --stats-only       # print stats from existing splits

Requires: ``datasets`` library (``pip install datasets``)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_time(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


# ---------------------------------------------------------------------------
# Step 1 — Download
# ---------------------------------------------------------------------------


def download_oasst2(split: str = "train+validation", max_rows: int = 0) -> list[dict]:
    """Download OASST2 and return flat message list.

    Args:
        split: dataset split name (e.g. "train+validation")
        max_rows: if > 0, limit to first N rows (streaming mode for speed)

    Returns list of dicts with keys: message_id, parent_id, message_tree_id,
    role, text, rank, labels, model_name, lang, review_result, ...
    """
    from datasets import load_dataset

    # Auto-detect mirror for users behind GFW
    mirror = os.environ.get("HF_ENDPOINT", os.environ.get("HF_MIRROR", ""))
    if not mirror:
        # Try common mirrors silently
        for candidate in ["https://hf-mirror.com"]:
            try:
                import urllib.request
                urllib.request.urlopen(candidate, timeout=3)
                mirror = candidate
                break
            except Exception:
                continue
    if mirror:
        os.environ.setdefault("HF_ENDPOINT", mirror)
        print(f"  Using HF mirror: {mirror}")

    if max_rows > 0:
        print(f"Downloading OpenAssistant/oasst2 (streaming, max {max_rows} rows)...")
        messages = []
        for s in split.split("+"):
            print(f"  Loading split: {s}...")
            ds = load_dataset("OpenAssistant/oasst2", split=s, streaming=True)
            for row in ds:
                messages.append(dict(row))
                if len(messages) >= max_rows:
                    break
            if len(messages) >= max_rows:
                break
        print(f"  Downloaded {len(messages)} messages (streaming)")
    else:
        print(f"Downloading OpenAssistant/oasst2 ({split})...")
        ds = load_dataset("OpenAssistant/oasst2", split=split)
        messages = [dict(row) for row in ds]
        print(f"  Downloaded {len(messages)} messages")
    return messages


# ---------------------------------------------------------------------------
# Step 2 — Parse trees
# ---------------------------------------------------------------------------

# Mapping from OASST2 label names to human-readable feedback descriptions
LABEL_FEEDBACK_MAP = {
    "quality": ("回答质量", lambda v: "高质量回答" if v > 0.7 else "质量一般" if v > 0.4 else "质量较低"),
    "creativity": ("创造力", lambda v: "很有创意" if v > 0.7 else "有想法" if v > 0.4 else "比较常规"),
    "humor": ("幽默感", lambda v: "很幽默" if v > 0.7 else "有点意思" if v > 0.4 else "比较严肃"),
    "helpfulness": ("实用性", lambda v: "非常有用" if v > 0.7 else "有所帮助" if v > 0.4 else "帮助有限"),
    "toxicity": ("毒性", lambda v: "内容不当" if v > 0.3 else "内容正常"),
    "violence": ("暴力内容", lambda v: "含有暴力" if v > 0.3 else "无暴力"),
    "spam": ("垃圾信息", lambda v: "疑似垃圾" if v > 0.3 else "正常"),
}


def _simulate_feedback(msg: dict) -> str:
    """Generate simulated user feedback from OASST2 labels.

    Combines multiple label dimensions into a natural-language feedback sentence,
    similar to what a real user might say when reviewing an assistant response.
    """
    if msg["role"] != "assistant":
        return ""

    parts = []
    labels = msg.get("labels") or []

    for label in labels:
        name = label.get("name", "")
        value = label.get("value", 0.0)
        if name in LABEL_FEEDBACK_MAP:
            title, fn = LABEL_FEEDBACK_MAP[name]
            desc = fn(value)
            parts.append(f"{title}: {desc}")

    # Rank provides overall quality signal (0 = best)
    rank = msg.get("rank")
    if rank is not None and rank > 0:
        if rank <= 2:
            parts.insert(0, f"总体评价: 优秀 (rank={rank})")
        elif rank <= 5:
            parts.insert(0, f"总体评价: 良好 (rank={rank})")
        else:
            parts.insert(0, f"总体评价: 需要改进 (rank={rank})")

    return "; ".join(parts) if parts else ""


def _msg_to_label_score(msg: dict) -> float | None:
    """Extract overall quality score from OASST2 labels.

    Uses rank (inverted) and quality label to produce a single 0-1 score
    suitable as a baseline reward signal.
    """
    labels = msg.get("labels") or []
    quality_scores = []
    for label in labels:
        if label.get("name") in ("quality", "helpfulness"):
            quality_scores.append(label.get("value", 0.0))

    if quality_scores:
        base = sum(quality_scores) / len(quality_scores)
    else:
        base = 0.5  # neutral if no quality label

    # Invert rank: lower rank = better. Scale: max_rank=10 → mapped to 0-1
    rank = msg.get("rank")
    if rank is not None:
        rank_penalty = min(rank / 10.0, 1.0) * 0.3  # rank accounts for up to 30%
        base = base * (1.0 - rank_penalty) + 0.1 * (1.0 - rank_penalty)

    return round(max(0.0, min(1.0, base)), 4)


def build_conversations(messages: list[dict]) -> list[dict]:
    """Build multi-turn conversation sessions from flat OASST2 messages.

    Each conversation is a dict:
        {
            "tree_id": str,
            "user_id": str (derived from root prompter),
            "model": str or None,
            "created_at": float,
            "messages": [
                {"role": "prompter"|"assistant", "content": str, "rank": int,
                 "labels": [...], "quality_score": float, "feedback": str},
                ...
            ],
            "label_count": int,
            "avg_quality_score": float,
        }

    Only the main path of each tree is used (highest-ranked branch), and
    only trees with at least one prompter+assistant pair are included.
    """
    # Group messages by tree
    trees: dict[str, list[dict]] = defaultdict(list)
    for msg in messages:
        trees[msg["message_tree_id"]].append(msg)

    conversations = []
    skipped_empty = 0
    skipped_single = 0

    for tree_id, msgs in trees.items():
        msgs.sort(key=lambda m: m.get("created_date", ""))

        # Build parent→children index
        children: dict[str | None, list[dict]] = defaultdict(list)
        for m in msgs:
            children[m.get("parent_id")].append(m)

        # Walk main path (choose highest-ranked branch at each fork)
        path = []
        current_list = children.get(None, [])  # root messages
        while current_list:
            # Pick best child: lowest rank first, then by review result
            current_list.sort(key=lambda m: (m.get("rank", 999), not m.get("review_result", False)))
            current = current_list[0]
            path.append(current)
            current_list = children.get(current["message_id"], [])

        # Filter: need at least 1 user + 1 assistant message
        has_prompter = any(m["role"] == "prompter" for m in path)
        has_assistant = any(m["role"] == "assistant" for m in path)
        if not has_prompter or not has_assistant:
            skipped_empty += 1
            continue
        if len(path) < 2:
            skipped_single += 1
            continue

        # Build conversation record
        conv_msgs = []
        total_quality = 0.0
        total_labels = 0

        for m in path:
            quality_score = _msg_to_label_score(m)
            feedback = _simulate_feedback(m)
            conv_msgs.append({
                "role": "user" if m["role"] == "prompter" else "assistant",
                "content": m["text"],
                "rank": m.get("rank"),
                "lang": m.get("lang"),
                "model_name": m.get("model_name"),
                "labels": m.get("labels") or [],
                "quality_score": quality_score,
                "feedback": feedback,
                "review_result": m.get("review_result"),
            })
            if m["role"] == "assistant":
                total_quality += quality_score
                total_labels += 1

        conversations.append({
            "tree_id": tree_id,
            "user_id": f"oasst2_user_{hash(tree_id) % 10000:04d}",
            "model": path[0].get("model_name") if path[0]["role"] == "assistant" else (
                path[1].get("model_name") if len(path) > 1 else None
            ),
            "created_at": time.mktime(time.strptime(
                path[0].get("created_date", "2023-01-01T00:00:00")[:19],
                "%Y-%m-%dT%H:%M:%S",
            )),
            "messages": conv_msgs,
            "total_messages": len(conv_msgs),
            "label_count": total_labels,
            "avg_quality_score": round(total_quality / total_labels, 4) if total_labels else None,
        })

    print(f"  Built {len(conversations)} valid conversations")
    print(f"  Skipped: {skipped_empty} no-pair, {skipped_single} single-message")
    return conversations


# ---------------------------------------------------------------------------
# Step 3 — Train / Test split
# ---------------------------------------------------------------------------


def split_train_test(
    conversations: list[dict],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split conversations by tree_id (not by message) to avoid leakage.

    Returns (train, test) tuple.
    """
    import random
    random.seed(seed)

    tree_ids = list({c["tree_id"] for c in conversations})
    random.shuffle(tree_ids)

    n_test = max(1, int(len(tree_ids) * test_size))
    test_ids = set(tree_ids[:n_test])

    train = [c for c in conversations if c["tree_id"] not in test_ids]
    test = [c for c in conversations if c["tree_id"] in test_ids]

    print(f"  Train: {len(train)} conversations ({len(set(c['tree_id'] for c in train))} trees)")
    print(f"  Test:  {len(test)} conversations ({len(set(c['tree_id'] for c in test))} trees)")

    return train, test


# ---------------------------------------------------------------------------
# Step 4 — Import train set into ConversationStore
# ---------------------------------------------------------------------------


def import_to_store(conversations: list[dict], db_path: str = "data/conversations.db") -> dict:
    """Import training conversations into the ConversationStore.

    Each conversation tree becomes one session with multiple messages.
    The OASST2 tree_id is stored in session metadata for traceability.
    Simulated feedback is stored in message metadata.

    Returns import statistics.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from trainable_openclaw.logging.conversation_store import ConversationStore

    store = ConversationStore(db_path)

    # Batch import with transaction for speed
    total_sessions = 0
    total_messages = 0
    total_feedback = 0

    for conv in conversations:
        sid = store.create_session(
            user_id=conv["user_id"],
            model=conv["model"] or "oasst2-unknown",
            metadata={
                "source": "oasst2",
                "tree_id": conv["tree_id"],
                "avg_quality_score": conv["avg_quality_score"],
                "lang": conv["messages"][0].get("lang"),
            },
        )

        for msg in conv["messages"]:
            meta = {
                "rank": msg.get("rank"),
                "oasst2_labels": msg.get("labels"),
                "quality_score": msg["quality_score"],
                "lang": msg.get("lang"),
                "source_model": msg.get("model_name"),
                "review_result": msg.get("review_result"),
            }
            if msg["feedback"]:
                meta["simulated_feedback"] = msg["feedback"]

            store.add_message(
                session_id=sid,
                role=msg["role"],
                content=msg["content"],
                metadata=meta,
                stop_reason="stop" if msg["role"] == "assistant" else None,
            )

            if msg["feedback"]:
                total_feedback += 1

        total_sessions += 1
        total_messages += len(conv["messages"])

    store.close()

    print(f"  Imported: {total_sessions} sessions, {total_messages} messages")
    print(f"  Feedback records: {total_feedback}")

    return {
        "sessions": total_sessions,
        "messages": total_messages,
        "feedback_records": total_feedback,
    }


# ---------------------------------------------------------------------------
# Step 5 — Export test set as JSONL
# ---------------------------------------------------------------------------


def export_test_jsonl(conversations: list[dict], output_path: str = "data/oasst2_test.jsonl") -> str:
    """Export test conversations as JSONL for evaluation.

    Each line is a JSON object with the conversation structure,
    suitable for model inference and scoring.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            # Extract user prompts (we'll evaluate model responses on these)
            prompts = [m["content"] for m in conv["messages"] if m["role"] == "user"]
            assistant_msgs = [m for m in conv["messages"] if m["role"] == "assistant"]

            record = {
                "tree_id": conv["tree_id"],
                "user_id": conv["user_id"],
                "model": conv["model"],
                "prompts": prompts,
                "reference_responses": [
                    {
                        "content": a["content"],
                        "quality_score": a["quality_score"],
                        "feedback": a["feedback"],
                        "rank": a["rank"],
                    }
                    for a in assistant_msgs
                ],
                "avg_quality_score": conv["avg_quality_score"],
                "total_messages": conv["total_messages"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    file_size = os.path.getsize(output_path)
    print(f"  Exported {len(conversations)} test records to {output_path} ({file_size/1024:.1f} KB)")
    return output_path


# ---------------------------------------------------------------------------
# Step 6 — Statistics
# ---------------------------------------------------------------------------


def print_statistics(conversations: list[dict], label: str = "Dataset") -> None:
    """Print summary statistics for a conversation split."""
    if not conversations:
        print(f"\n{label}: (empty)")
        return

    total_msgs = sum(c["total_messages"] for c in conversations)
    user_msgs = sum(
        sum(1 for m in c["messages"] if m["role"] == "user") for c in conversations
    )
    asst_msgs = sum(
        sum(1 for m in c["messages"] if m["role"] == "assistant") for c in conversations
    )
    scores = [
        c["avg_quality_score"]
        for c in conversations
        if c["avg_quality_score"] is not None
    ]
    feedback_count = sum(
        sum(1 for m in c["messages"] if m.get("feedback")) for c in conversations
    )
    # Turns per conversation
    turns = [c["total_messages"] // 2 for c in conversations]
    avg_turns = sum(turns) / len(turns) if turns else 0

    # Language distribution (from first message)
    langs: dict[str, int] = defaultdict(int)
    for c in conversations:
        lang = c["messages"][0].get("lang", "unknown") if c["messages"] else "unknown"
        langs[lang] += 1

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Conversations:     {len(conversations)}")
    print(f"  Total messages:    {total_msgs}")
    print(f"  User messages:     {user_msgs}")
    print(f"  Assistant msgs:    {asst_msgs}")
    print(f"  Avg turns/conv:    {avg_turns:.1f}")
    print(f"  Feedback records:  {feedback_count}")
    if scores:
        print(f"  Avg quality score: {sum(scores)/len(scores):.3f}  (min={min(scores):.3f}, max={max(scores):.3f})")

    # Top languages
    top_langs = sorted(langs.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top languages:     {', '.join(f'{l}({c})' for l, c in top_langs)}")

    # Turn distribution
    turn_dist = defaultdict(int)
    for t in turns:
        if t <= 2:
            turn_dist["1-2 turns"] += 1
        elif t <= 5:
            turn_dist["3-5 turns"] += 1
        elif t <= 10:
            turn_dist["6-10 turns"] += 1
        else:
            turn_dist["10+ turns"] += 1
    dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(turn_dist.items()))
    print(f"  Turn distribution: {dist_str}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="OASST2 Data Preparation")
    parser.add_argument("--split-only", action="store_true",
                        help="Only download + split, skip store import")
    parser.add_argument("--stats-only", action="store_true",
                        help="Print statistics from existing JSONL splits")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set ratio (default: 0.2)")
    parser.add_argument("--max-rows", type=int, default=5000,
                        help="Max rows to download from HF (0=all, default: 5000)")
    parser.add_argument("--db-path", default="data/conversations.db",
                        help="ConversationStore database path")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory for splits")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.stats_only:
        # Read existing splits
        for name, path in [("Train", f"{args.output_dir}/oasst2_train.jsonl"),
                           ("Test", f"{args.output_dir}/oasst2_test.jsonl")]:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    convs = [json.loads(line) for line in f if line.strip()]
                # Convert back to internal format for stats
                print_statistics(convs, name)
            else:
                print(f"  {path} not found, skipping")
        return

    # Full pipeline
    print(f"\n{'='*60}")
    print(f"  OASST2 Data Preparation")
    print(f"  Test size: {args.test_size:.0%}")
    print(f"  DB path:   {args.db_path}")
    print(f"  Output:    {args.output_dir}/")
    print(f"{'='*60}\n")

    # 1. Download
    messages = download_oasst2(max_rows=args.max_rows)

    # 2. Parse trees
    print("\nBuilding conversation trees...")
    conversations = build_conversations(messages)

    # 3. Split
    print(f"\nSplitting train/test ({args.test_size:.0%} test)...")
    train, test = split_train_test(conversations, test_size=args.test_size)
    print_statistics(train, "Train Set")
    print_statistics(test, "Test Set")

    # 4. Import train into store
    if not args.split_only:
        print("\nImporting train set into ConversationStore...")
        import_stats = import_to_store(train, db_path=args.db_path)

    # 5. Export both splits as JSONL (for portability and evaluation)
    train_path = f"{args.output_dir}/oasst2_train.jsonl"
    test_path = f"{args.output_dir}/oasst2_test.jsonl"

    print("\nExporting splits...")
    for convs, path, name in [(train, train_path, "train"), (test, test_path, "test")]:
        export_test_jsonl(convs, path)
        # Also export as a list of dicts (for programmatic use)
        full_path = path.replace(".jsonl", "_full.json")
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(convs, f, ensure_ascii=False, indent=2)
        print(f"  {name} full dump: {full_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Data Preparation Complete")
    print(f"{'='*60}")
    print(f"  Train set:     {len(train)} convs → {args.db_path}")
    print(f"  Test set:      {len(test)} convs → {test_path}")
    print(f"  Train JSONL:   {train_path}")
    print(f"")
    print(f"  Next steps:")
    print(f"    1. View data:  python -m trainable_openclaw.logging.viewer stats")
    print(f"    2. Eval model: python scripts/eval_oasst2_baseline.py (TBD)")
    print()


if __name__ == "__main__":
    main()
