"""
Phase 2: 自进化评判系统 — evaluation 包

数据流:
  S2 trajectories.jsonl
    → S3 trajectory_eval: 分级 + 提取训练对 + 聚合 Rubric 种子
    → B1 feedback: LLM 分析反馈模式
    → B2 rubric: LLM 生成评分 Rubric
    → B3 judge: 执行 Rubric 打分 → GRPO reward

Usage:
    from trainable_openclaw.evaluation import (
        trajectory_eval,  # S3
        feedback,          # B1
        rubric,            # B2
        judge,             # B3
    )
"""
