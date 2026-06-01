#!/usr/bin/env python3
"""
Calculate 纠错率 (Correction Rate) from trajectory JSONL files.

Reads S2 simulation trajectories and counts outcomes:
  直接通过 / 纠错后通过 / 部分通过 / 失败

纠错率 = (total - 直接通过) / total

Usage:
    python scripts/calc_correction_rate.py data/test_trajectories.jsonl
    python scripts/calc_correction_rate.py data/train_trajectories.jsonl
    python scripts/calc_correction_rate.py --all  # both train and test
"""

import json
import sys
from pathlib import Path
from collections import Counter


def calc_correction_rate(filepath: str) -> dict:
    trajectories = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))

    outcomes = Counter()
    for t in trajectories:
        verdict = t.get("最终判定", t.get("verdict", "未知"))
        error = t.get("错误信息", t.get("error", ""))
        if error:
            outcomes["失败"] += 1
        else:
            outcomes[verdict] += 1

    total = len(trajectories)
    direct_pass = outcomes.get("直接通过", 0)
    corrected = outcomes.get("纠错后通过", 0)
    partial = outcomes.get("部分通过", 0)
    failed = outcomes.get("失败", 0)

    correction_rate = (total - direct_pass) / total if total > 0 else 0.0

    return {
        "file": str(filepath),
        "total": total,
        "直接通过": direct_pass,
        "纠错后通过": corrected,
        "部分通过": partial,
        "失败": failed,
        "纠错率": round(correction_rate, 4),
        "直接通过率": round(direct_pass / total, 4) if total > 0 else 0.0,
    }


def print_report(result: dict):
    print(f"\n{'='*55}")
    print(f"  文件: {result['file']}")
    print(f"{'='*55}")
    print(f"  总轨迹数:     {result['total']:>5d}")
    print(f"  直接通过:     {result['直接通过']:>5d}  ({result['直接通过']/max(result['total'],1)*100:.1f}%)")
    print(f"  纠错后通过:   {result['纠错后通过']:>5d}  ({result['纠错后通过']/max(result['total'],1)*100:.1f}%)")
    print(f"  部分通过:     {result['部分通过']:>5d}  ({result['部分通过']/max(result['total'],1)*100:.1f}%)")
    print(f"  失败:         {result['失败']:>5d}  ({result['失败']/max(result['total'],1)*100:.1f}%)")
    print(f"  {'─'*45}")
    print(f"  纠错率:       {result['纠错率']:.2%}")
    print(f"  直接通过率:   {result['直接通过率']:.2%}")
    print(f"{'='*55}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="计算纠错率")
    parser.add_argument("file", nargs="?", help="轨迹 JSONL 文件路径")
    parser.add_argument("--all", action="store_true", help="同时计算 train + test")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args()

    if args.all:
        paths = [
            "data/train_trajectories.jsonl",
            "data/test_trajectories.jsonl",
        ]
    elif args.file:
        paths = [args.file]
    else:
        # Default: test set
        paths = ["data/test_trajectories.jsonl"]

    results = []
    for p in paths:
        if Path(p).exists():
            r = calc_correction_rate(p)
            results.append(r)
            if args.json:
                print(json.dumps(r, ensure_ascii=False))
            else:
                print_report(r)
        else:
            print(f"文件不存在: {p}", file=sys.stderr)

    # Summary comparison
    if len(results) == 2 and not args.json:
        train, test = results[0], results[1]
        print(f"  训练集纠错率: {train['纠错率']:.2%}  |  测试集纠错率: {test['纠错率']:.2%}")
        print(f"  差异: {abs(train['纠错率'] - test['纠错率']):.2%}\n")


if __name__ == "__main__":
    main()
