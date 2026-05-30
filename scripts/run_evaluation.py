#!/usr/bin/env python3
"""
Phase 2 评估流水线: S3 → B1 → B2 → B3

处理 S2 产出的纠错轨迹，运行完整的评估和 Rubric 生成流程。

用法:
    # 完整流水线（需要 DeepSeek API）
    python scripts/run_evaluation.py --input data/train_trajectories.jsonl

    # 仅 S3 轨迹评估（不需要 API）
    python scripts/run_evaluation.py --input data/train_trajectories.jsonl --s3-only

    # 简单模式（不调用 LLM，用模板生成 Rubric）
    python scripts/run_evaluation.py --input data/train_trajectories.jsonl --simple
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_env():
    for base in [
        Path(__file__).resolve().parent.parent / ".env",
        Path(".env"),
    ]:
        if base.exists():
            with open(base) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


async def run_full_pipeline(
    input_file: str,
    output_dir: str,
    api_key: str,
    simple: bool = False,
):
    """运行 S3 → B1 → B2 → B3 完整流水线。"""
    from trainable_openclaw.evaluation.trajectory_eval import process_trajectories
    from trainable_openclaw.evaluation.feedback import (
        FeedbackAnalyzer,
        load_rubric_seeds,
        load_training_pairs,
    )
    from trainable_openclaw.evaluation.rubric import RubricGenerator, RubricStore
    from trainable_openclaw.evaluation.judge import JudgeExecutor

    # ── S3: 轨迹评估 ──
    print("\n" + "=" * 60)
    print("  S3: 轨迹评估与数据导出")
    print("=" * 60)
    stats = process_trajectories(input_file, output_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # ── B1: 反馈分析 ──
    print("\n" + "=" * 60)
    print("  B1: 用户反馈模式分析")
    print("=" * 60)
    seeds = load_rubric_seeds(f"{output_dir}/rubric_seeds.json")
    print(f"  Rubric 种子维度: {len(seeds)}")
    for s in seeds[:10]:
        print(f"    {s['维度']}: {s['频次']}次")

    analyzer = FeedbackAnalyzer(api_key=api_key)
    if simple:
        patterns = analyzer.analyze_simple(seeds)
    else:
        pairs = load_training_pairs(f"{output_dir}/training_pairs.jsonl")
        patterns = await analyzer.analyze(seeds, pairs)

    print(f"\n  识别的反馈模式: {len(patterns)}")
    for p in patterns[:10]:
        print(f"    [{p.严重程度}] {p.模式名称} (频次: {p.频次})")

    if not patterns:
        print("  ⚠ 未识别到反馈模式，流水线终止")
        return

    # ── B2: Rubric 生成 ──
    print("\n" + "=" * 60)
    print("  B2: LLM 生成评分 Rubric")
    print("=" * 60)
    store = RubricStore(path=f"{output_dir}/rubrics.json")
    generator = RubricGenerator(api_key=api_key, store=store)

    if simple:
        rubrics = generator.generate_simple(patterns)
    else:
        rubrics = await generator.generate_all(patterns)

    print(f"  生成的 Rubric: {len(rubrics)}")
    for r in rubrics[:5]:
        prompt_preview = r.评分提示词[:120].replace("\n", " ")
        print(f"    {r.名称} (v{r.版本}) — {prompt_preview}...")

    # ── B3: 验证 Judge 可执行性 ──
    print("\n" + "=" * 60)
    print("  B3: Judge 可执行性验证")
    print("=" * 60)
    judge = JudgeExecutor(api_key=api_key)

    # 用一条样例测试
    test_answer = "这是一个测试回答，用于验证 Judge 是否正常工作。"
    try:
        active_rubrics = store.list_active()
        if active_rubrics and not simple:
            # 只测试第一条 rubric（避免消耗太多 API 调用）
            result = await judge.score_one(active_rubrics[0], test_answer)
            print(f"  测试 Rubric: {result.rubric_name}")
            print(f"  分数: {result.分数}")
            print(f"  扣分项: {result.扣分项}")
            print(f"  总结: {result.总结}")
            if result.解析错误:
                print(f"  ⚠ 解析错误: {result.解析错误[:200]}")
        else:
            print(f"  (简单模式，跳过 Judge API 验证)")
            print(f"  活跃 Rubric: {len(active_rubrics)} 条")
            for r in active_rubrics[:3]:
                print(f"    {r.名称}")
    except Exception as e:
        print(f"  ⚠ Judge 验证失败: {e}")

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("  流水线完成")
    print("=" * 60)
    print(f"  输入:      {input_file}")
    print(f"  输出目录:  {output_dir}")
    print(f"  训练对:    {stats['训练对数']} 条")
    print(f"  Rubric:    {len(rubrics)} 条")
    print(f"  输出文件:")
    for k, v in stats.get("输出文件", {}).items():
        print(f"    {k}: {v}")
    if store.path.exists():
        print(f"    Rubric存储: {store.path}")


def main():
    load_env()
    parser = argparse.ArgumentParser(description="Phase 2 评估流水线")
    parser.add_argument("--input", default="data/train_trajectories.jsonl")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--simple", action="store_true",
                        help="简单模式（不调用 LLM，使用模板）")
    parser.add_argument("--s3-only", action="store_true",
                        help="仅运行 S3 轨迹评估")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)

    if not args.api_key and not args.s3_only and not args.simple:
        print("错误: 未设置 API 密钥，使用 --simple 模式或设置 DEEPSEEK_API_KEY")
        sys.exit(1)

    if args.s3_only:
        from trainable_openclaw.evaluation.trajectory_eval import process_trajectories
        stats = process_trajectories(args.input, args.output_dir)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        asyncio.run(run_full_pipeline(
            input_file=args.input,
            output_dir=args.output_dir,
            api_key=args.api_key,
            simple=args.simple,
        ))


if __name__ == "__main__":
    main()
