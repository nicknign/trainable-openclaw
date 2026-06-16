"""
Add agent_name and system prompt to tau-bench jsonl for verl Agent Loop.

Converts:
    {"prompt": "You name is Nancy...", ...}
To:
    {"prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
     "agent_name": "tool_agent", ...}

Usage:
    python scripts/data/add_agent_name.py \
        data/tau_bench/train_split_66.jsonl \
        data/tau_bench/train_agent_66.jsonl

    python scripts/data/add_agent_name.py \
        data/tau_bench/val_split_18.jsonl \
        data/tau_bench/val_agent_18.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a retail customer service agent. "
    "You help customers with orders, returns, exchanges, refunds, address changes, "
    "and other retail inquiries.\n\n"
    "You have access to tools to look up customer information, order details, "
    "inventory, and perform actions like cancelling, returning, exchanging, or "
    "modifying orders.\n\n"
    "Rules:\n"
    "- Before taking any action, verify the customer's identity using available tools.\n"
    "- Never make up information — always use tools to retrieve or modify data.\n"
    "- Be polite, professional, and efficient.\n"
    "- Address each of the customer's needs step by step."
)


def convert(input_path: str, output_path: str) -> None:
    count = 0
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            prompt_text = obj.pop("prompt")
            obj["prompt"] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]
            obj["agent_name"] = "tool_agent"

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} records: {input_path} → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Add agent_name for verl Agent Loop")
    parser.add_argument("input", help="Input jsonl file")
    parser.add_argument("output", help="Output jsonl file")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
