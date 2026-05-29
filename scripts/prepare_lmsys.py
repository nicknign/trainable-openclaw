#!/usr/bin/env python3
"""
LMSYS-Chat-1M (Clean) Data Preparation for Phase B.

Dataset: AI-ModelScope/lmsys_chat_1m_clean (ModelScope)
Features per row:
  - id: conversation UUID
  - conversations: [{from: "human"|"gpt", value: str}, ...]
  - category: topic label (explanation, coding, creative writing, ...)
  - grounded: bool
  - deepseek_response: {moralization: int, reward: float, value: str}
  - phi-3-mini_response: {moralization: int, reward: float, value: str}
  - flaw: str — description of response flaw (or "normal" if none)
  - agreement: bool | None — whether DeepSeek and Phi-3-mini agree

Pipeline:
  1. Download all parquet files from ModelScope
  2. Parse and import into ConversationStore
  3. Export test split as JSONL
  4. Print statistics

Usage:
    python scripts/prepare_lmsys.py                      # full pipeline
    python scripts/prepare_lmsys.py --max-rows 5000      # limit rows for dev
    python scripts/prepare_lmsys.py --stats-only         # print stats from existing DB
    python scripts/prepare_lmsys.py --no-import          # export JSONL only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 1 — Download from ModelScope
# ---------------------------------------------------------------------------


def download_parquet_files(
    output_dir: str = "data/lmsys_chat", expected_count: int = 4
) -> list[str]:
    """Download all parquet files via modelscope CLI, return list of file paths.

    Always runs the download command — modelscope CLI handles dedup for
    already-downloaded files, so partial downloads resume correctly.
    """
    os.makedirs(output_dir, exist_ok=True)

    existing = sorted(Path(output_dir).glob("data/train-*.parquet"))
    print(f"  Found {len(existing)} existing parquet file(s) in {output_dir}/data/")

    if len(existing) >= expected_count:
        print(f"  All {expected_count} files present, skipping download")
        return [str(p) for p in existing]

    print(f"  Downloading from ModelScope: AI-ModelScope/lmsys_chat_1m_clean")
    result = subprocess.run(
        [
            "modelscope", "download",
            "--dataset", "AI-ModelScope/lmsys_chat_1m_clean",
            "--local_dir", output_dir,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"modelscope download failed: {result.stderr[:500]}")

    files = sorted(Path(output_dir).glob("data/train-*.parquet"))
    print(f"  Downloaded, now have {len(files)} parquet file(s)")
    return [str(p) for p in files]


# ---------------------------------------------------------------------------
# Step 2 — Parse & import
# ---------------------------------------------------------------------------

# Map conversation roles from LMSYS to our convention
ROLE_MAP = {"human": "user", "gpt": "assistant"}


def _map_feedback(flaw: str | None, agreement: bool | None) -> str:
    """Convert flaw field into human-readable feedback text."""
    if not flaw or flaw == "normal":
        return ""
    return flaw


def _compute_quality_score(
    deepseek_reward: float | None,
    phi_reward: float | None,
) -> float | None:
    """Normalize reward to 0-1 quality score.

    The original rewards range roughly -20 to +20 from LLM judges.
    We normalize via sigmoid to map to [0, 1].
    """
    scores = []
    if deepseek_reward is not None:
        scores.append(deepseek_reward)
    if phi_reward is not None:
        scores.append(phi_reward)
    if not scores:
        return None

    # Average then sigmoid to [0, 1]
    avg = sum(scores) / len(scores)
    # Scale: typical range [-20, 20] → sigmoid(avg/5) maps well to [0, 1]
    import math
    try:
        return round(1.0 / (1.0 + math.exp(-avg / 5.0)), 4)
    except OverflowError:
        return 1.0 if avg > 0 else 0.0


def process_row(row: dict) -> dict | None:
    """Convert one dataset row into a conversation dict for import.

    Returns None if the row is invalid/malformed.
    """
    import numpy as np

    conv_id = row.get("id")
    turns = row.get("conversations", [])

    # Handle numpy arrays from parquet
    if isinstance(turns, np.ndarray):
        turns = turns.tolist()
    if turns is None or (hasattr(turns, "__len__") and len(turns) == 0):
        return None
    if not conv_id:
        return None

    # Build messages array
    messages = []
    for turn in turns:
        role = ROLE_MAP.get(turn.get("from", ""), turn.get("from", ""))
        content = turn.get("value", "")
        if not content:
            continue
        messages.append({"role": role, "content": content})

    if len(messages) < 2:
        return None  # Need at least one user + one assistant

    # Extract quality signals
    ds_resp = row.get("deepseek_response") or {}
    phi_resp = row.get("phi-3-mini_response") or {}

    ds_reward = ds_resp.get("reward") if isinstance(ds_resp, dict) else None
    phi_reward = phi_resp.get("reward") if isinstance(phi_resp, dict) else None

    quality_score = _compute_quality_score(ds_reward, phi_reward)

    flaw = row.get("flaw", "")
    agreement = row.get("agreement")
    feedback = _map_feedback(flaw, agreement)

    category = row.get("category", "")
    grounded = row.get("grounded", False)

    # Derive user_id from conversation ID (deterministic pseudo-user)
    user_id = f"lmsys_user_{hash(conv_id) % 50000:05d}"

    return {
        "conv_id": conv_id,
        "user_id": user_id,
        "model": "lmsys-chat",  # placeholder, real model name not in clean version
        "category": category,
        "messages": messages,
        "quality_score": quality_score,
        "ds_reward": ds_reward,
        "phi_reward": phi_reward,
        "flaw": flaw or "",
        "feedback": feedback,
        "agreement": agreement,
        "grounded": grounded,
    }


def import_to_store(
    parquet_files: list[str],
    db_path: str = "data/conversations.db",
    max_rows: int = 0,
    test_size: float = 0.1,
) -> dict:
    """Load parquet files, parse conversations, import to ConversationStore.

    Args:
        parquet_files: list of .parquet file paths
        db_path: SQLite database path
        max_rows: if > 0, limit total rows loaded (0 = all)
        test_size: fraction of conversations held out as test set

    Returns stats dict.
    """
    import pyarrow.parquet as pq

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from trainable_openclaw.logging.conversation_store import ConversationStore

    store = ConversationStore(db_path)

    total_rows = 0
    total_imported = 0
    total_messages = 0
    total_feedback = 0
    test_convs = []

    for fpath in parquet_files:
        print(f"  Processing {os.path.basename(fpath)} ...")
        pf = pq.ParquetFile(fpath)
        # Read in batches to manage memory
        for batch in pf.iter_batches(batch_size=10000):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                conv = process_row(row.to_dict())
                if conv is None:
                    continue

                total_rows += 1

                # Hold-out test set
                is_test = hash(conv["conv_id"]) % 100 < int(test_size * 100)
                if is_test:
                    test_convs.append(conv)
                    if max_rows > 0 and total_rows >= max_rows:
                        break
                    continue

                # Import into store
                sid = store.create_session(
                    user_id=conv["user_id"],
                    model=conv["model"],
                    metadata={
                        "source": "lmsys_chat_1m_clean",
                        "conv_id": conv["conv_id"],
                        "category": conv["category"],
                        "quality_score": conv["quality_score"],
                        "ds_reward": conv["ds_reward"],
                        "phi_reward": conv["phi_reward"],
                        "agreement": conv["agreement"],
                        "grounded": conv["grounded"],
                    },
                )

                for i, msg in enumerate(conv["messages"]):
                    meta = {"turn_index": i}
                    if msg["role"] == "assistant":
                        meta["quality_score"] = conv["quality_score"]
                        if conv["feedback"]:
                            meta["simulated_feedback"] = conv["feedback"]
                            total_feedback += 1

                    store.add_message(
                        session_id=sid,
                        role=msg["role"],
                        content=msg["content"],
                        metadata=meta,
                        stop_reason="stop" if msg["role"] == "assistant" else None,
                    )

                total_imported += 1
                total_messages += len(conv["messages"])

                if max_rows > 0 and total_rows >= max_rows:
                    break

            if max_rows > 0 and total_rows >= max_rows:
                break

        if max_rows > 0 and total_rows >= max_rows:
            break

    store.close()

    # Export test set as JSONL
    if test_convs:
        test_path = "data/lmsys_test.jsonl"
        with open(test_path, "w", encoding="utf-8") as f:
            for c in test_convs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"  Test set: {len(test_convs)} conversations → {test_path}")

    return {
        "total_rows_seen": total_rows,
        "imported_sessions": total_imported,
        "imported_messages": total_messages,
        "feedback_records": total_feedback,
        "test_conversations": len(test_convs),
    }


# ---------------------------------------------------------------------------
# Step 3 — Statistics
# ---------------------------------------------------------------------------


def print_statistics(db_path: str = "data/conversations.db") -> None:
    """Print dataset summary from the database."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from trainable_openclaw.logging.conversation_store import ConversationStore

    store = ConversationStore(db_path)
    stats = store.get_statistics()

    print(f"\n{'='*60}")
    print(f"  LMSYS-Chat-1M (Clean) Dataset Statistics")
    print(f"{'='*60}")
    print(f"  Database:           {db_path}")
    print(f"  Total sessions:     {stats['total_sessions']}")
    print(f"  Total messages:     {stats['total_messages']}")
    print(f"  Unique users:       {stats['total_users']}")
    print(f"  Role distribution:  {stats['role_distribution']}")

    if stats.get("first_message_at") and stats.get("last_message_at"):
        first = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats["first_message_at"]))
        last = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats["last_message_at"]))
        print(f"  Time range:         {first} → {last}")

    if stats.get("user_breakdown"):
        print(f"\n  Top users:")
        for u in stats["user_breakdown"][:10]:
            print(f"    {u['user_id']:30s} {u['sessions']:4d} sessions, {u.get('msgs', 0):4d} msgs")

    # Category distribution from metadata
    cat_rows = store.conn.execute(
        "SELECT json_extract(metadata, '$.category') as cat, COUNT(*) as cnt "
        "FROM sessions "
        "WHERE json_extract(metadata, '$.source') = 'lmsys_chat_1m_clean' "
        "GROUP BY cat ORDER BY cnt DESC LIMIT 15"
    ).fetchall()
    if cat_rows:
        print(f"\n  Category distribution:")
        for r in cat_rows:
            print(f"    {r['cat'] or 'unknown':30s} {r['cnt']:5d}")

    # Quality score distribution
    score_rows = store.conn.execute(
        "SELECT AVG(json_extract(metadata, '$.quality_score')) as avg, "
        "MIN(json_extract(metadata, '$.quality_score')) as min, "
        "MAX(json_extract(metadata, '$.quality_score')) as max "
        "FROM sessions "
        "WHERE json_extract(metadata, '$.source') = 'lmsys_chat_1m_clean'"
    ).fetchone()
    if score_rows and score_rows["avg"] is not None:
        print(f"\n  Quality score (0-1):")
        print(f"    avg={score_rows['avg']:.3f}  min={score_rows['min']:.3f}  max={score_rows['max']:.3f}")

    # Agreement rate (from metadata JSON, excluding NULLs)
    agree_rows = store.conn.execute(
        "SELECT json_extract(metadata, '$.agreement') as agreement, COUNT(*) as cnt "
        "FROM sessions "
        "WHERE json_extract(metadata, '$.source') = 'lmsys_chat_1m_clean' "
        "  AND json_extract(metadata, '$.agreement') IS NOT NULL "
        "GROUP BY agreement"
    ).fetchall()
    agree_map = {r["agreement"]: r["cnt"] for r in agree_rows}
    total_rated = sum(agree_map.values())
    agreed = agree_map.get(1, 0)
    disagreed = agree_map.get(0, 0)
    if total_rated > 0:
        print(f"\n  Judge agreement (DeepSeek vs Phi-3-mini):")
        print(f"    Agree: {agreed}/{total_rated} = {agreed/total_rated:.1%}")
        print(f"    Disagree: {disagreed}/{total_rated} = {disagreed/total_rated:.1%}")

    store.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="LMSYS-Chat-1M Data Preparation")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Max rows to process (0 = all)")
    parser.add_argument("--test-size", type=float, default=0.1,
                        help="Test set ratio (default: 0.1)")
    parser.add_argument("--db-path", default="data/conversations.db",
                        help="ConversationStore database path")
    parser.add_argument("--data-dir", default="data/lmsys_chat",
                        help="Directory with parquet files")
    parser.add_argument("--no-import", action="store_true",
                        help="Skip store import, only export JSONL")
    parser.add_argument("--stats-only", action="store_true",
                        help="Print statistics from existing database")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing files")
    args = parser.parse_args()

    if args.stats_only:
        print_statistics(args.db_path)
        return

    print(f"\n{'='*60}")
    print(f"  LMSYS-Chat-1M (Clean) Data Preparation")
    print(f"  Max rows: {'ALL' if args.max_rows == 0 else args.max_rows}")
    print(f"  Test size: {args.test_size:.0%}")
    print(f"  DB path:   {args.db_path}")
    print(f"{'='*60}\n")

    # 1. Download
    if not args.skip_download:
        print("Step 1: Download parquet files")
        parquet_files = download_parquet_files(args.data_dir)
        print()
    else:
        parquet_files = sorted(Path(args.data_dir).glob("data/train-*.parquet"))
        if not parquet_files:
            print(f"ERROR: No parquet files found in {args.data_dir}/data/")
            sys.exit(1)
        parquet_files = [str(p) for p in parquet_files]
        print(f"Found {len(parquet_files)} parquet file(s)\n")

    # 2. Import
    if not args.no_import:
        print("Step 2: Import to ConversationStore")
        stats = import_to_store(
            parquet_files,
            db_path=args.db_path,
            max_rows=args.max_rows,
            test_size=args.test_size,
        )
        print(f"\n  Import complete:")
        print(f"    Rows processed:   {stats['total_rows_seen']}")
        print(f"    Sessions:         {stats['imported_sessions']}")
        print(f"    Messages:         {stats['imported_messages']}")
        print(f"    Feedback records: {stats['feedback_records']}")
        print(f"    Test held-out:    {stats['test_conversations']}")

    # 3. Stats
    print()
    print_statistics(args.db_path)

    print(f"\n  Next steps:")
    print(f"    1. View data:  python -m trainable_openclaw.logging.viewer stats")
    print(f"    2. Search:     python -m trainable_openclaw.logging.viewer search 'keyword'")
    print(f"    3. Run tests:  pytest tests/test_conversation_store.py -v")
    print()


if __name__ == "__main__":
    main()
