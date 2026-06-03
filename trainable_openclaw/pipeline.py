from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full training pipeline."""

    train_data_path: str = "data/phase3_datasets/train_prompts.jsonl"
    test_data_path: str = "data/phase3_datasets/test_prompts.jsonl"
    rubrics_path: str = "data/rubrics_category.json"
    db_path: str = "data/conversations.db"
    model_server_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    base_url: str = ""
    judge_model: str = "deepseek-v4-flash"
    sim_model: str = "deepseek-v4-flash"
    max_test_prompts: int = 0
    reward_mode: str = "mean"
    max_rubrics: int = 8
    enable_thinking: bool = False
    run_pre_eval: bool = True
    run_post_eval: bool = True
    run_rubric_evolution: bool = False

    def __post_init__(self):
        self.api_key = self.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = self.base_url or os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )


@dataclass
class RoundResult:
    """Result of one pipeline round."""

    round_number: int = 1
    pre_纠错率: Optional[float] = None
    post_纠错率: Optional[float] = None
    Δ纠错率: Optional[float] = None
    training_completed: bool = False
    rubrics_updated: int = 0
    errors: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "轮次": self.round_number,
            "pre_纠错率": self.pre_纠错率,
            "post_纠错率": self.post_纠错率,
            "Δ纠错率": self.Δ纠错率,
            "训练完成": self.training_completed,
            "rubric更新数": self.rubrics_updated,
            "耗时秒": round(self.elapsed_seconds, 1),
            "错误": self.errors,
        }


class Pipeline:
    """Main training pipeline orchestrator.

    Coordinates: pre-eval -> training -> post-eval -> rubric evolution.

    Usage::

        config = PipelineConfig(
            model_server_url="http://localhost:8000/v1",
            api_key="sk-xxx",
        )
        pipeline = Pipeline(config)
        result = await pipeline.run_round(round_number=1)
        print(f"Δ纠错率: {result.Δ纠错率}")
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._pre_result = None

    # --- Evaluation methods ---

    async def run_pre_eval(self) -> dict:
        """Run pre-training correction rate baseline.

        Returns:
            CorrectionRateResult.to_dict().
        """
        from trainable_openclaw.evaluation.correction_rate import (
            CorrectionRateEvaluator,
        )

        evaluator = CorrectionRateEvaluator(
            model_server_url=self.config.model_server_url,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            sim_model=self.config.sim_model,
        )
        test_prompts = load_test_prompts(
            self.config.test_data_path, self.config.max_test_prompts
        )
        if not test_prompts:
            logger.warning("No test prompts found at %s", self.config.test_data_path)
            return {}

        logger.info("Running pre-eval on %d prompts...", len(test_prompts))
        result = await evaluator.evaluate(test_prompts)
        self._pre_result = result
        d = result.to_dict()
        logger.info("Pre-eval: 纠错率=%.3f", d.get("纠错率", 0))
        return d

    async def run_post_eval(self, pre_result: dict | None = None) -> dict:
        """Run post-training correction rate evaluation with delta.

        Args:
            pre_result: Pre-training result dict for delta computation.
                If None, uses the result from run_pre_eval().

        Returns:
            CorrectionRateResult.to_dict() with delta fields.
        """
        from trainable_openclaw.evaluation.correction_rate import (
            CorrectionRateEvaluator,
        )

        evaluator = CorrectionRateEvaluator(
            model_server_url=self.config.model_server_url,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            sim_model=self.config.sim_model,
        )
        test_prompts = load_test_prompts(
            self.config.test_data_path, self.config.max_test_prompts
        )
        if not test_prompts:
            logger.warning("No test prompts found at %s", self.config.test_data_path)
            return {}

        pre = pre_result or {}
        if self._pre_result and not pre:
            pre = self._pre_result.to_dict()

        logger.info("Running post-eval on %d prompts...", len(test_prompts))
        result = await evaluator.evaluate(test_prompts)
        post = result.to_dict()

        delta = 0.0
        if pre:
            pre_rate = pre.get("纠错率", 0)
            post_rate = post.get("纠错率", 0)
            delta = post_rate - pre_rate
        post["Δ纠错率"] = delta
        post["pre_纠错率"] = pre.get("纠错率", 0)

        logger.info(
            "Post-eval: 纠错率=%.3f, pre=%.3f, Δ=%.3f",
            post.get("纠错率", 0),
            pre.get("纠错率", 0),
            delta,
        )
        return post

    # --- Training config generation ---

    def generate_training_config(self, output_path: str = "") -> str:
        """Generate Hydra config overrides for serve_ppo training.

        Returns the config string that can be appended to the serve_ppo command.
        """
        overrides = [
            f"+trainer.trajectory.enabled=true",
            f"+trainer.trajectory.data_path={self.config.train_data_path}",
            f"+trainer.trajectory.rubrics_path={self.config.rubrics_path}",
            f"+trainer.trajectory.api_key={self.config.api_key}",
            f"+trainer.trajectory.max_rubrics={self.config.max_rubrics}",
            f"+trainer.trajectory.reward_mode={self.config.reward_mode}",
        ]
        config_str = " ".join(overrides)

        if output_path:
            with open(output_path, "w") as f:
                f.write(config_str + "\n")
            logger.info("Training config written to %s", output_path)

        return config_str

    # --- Rubric evolution ---

    async def evolve_rubrics(self) -> dict:
        """Run rubric evolution using conversation logs."""
        try:
            from trainable_openclaw.evaluation.rubric_evolver import RubricEvolver

            evolver = RubricEvolver(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                model=self.config.judge_model,
                rubrics_path=self.config.rubrics_path,
                db_path=self.config.db_path,
                max_rubrics=self.config.max_rubrics,
            )
            result = await evolver.check_and_evolve()
            return {
                "triggered": result.triggered,
                "reason": result.trigger_reason,
                "new_rubrics": result.new_rubrics,
                "archived": result.archived_rubrics,
                "details": result.details,
            }
        except ImportError:
            logger.warning("RubricEvolver not available — skipping evolution")
            return {"triggered": False, "reason": "模块不可用", "details": []}
        except Exception as e:
            logger.warning("Rubric evolution failed: %s", e)
            return {"triggered": False, "reason": str(e), "details": []}

    # --- Pipeline orchestration ---

    async def run_round(
        self,
        round_number: int = 1,
        run_training: bool = False,
    ) -> RoundResult:
        """Execute one full pipeline round.

        Args:
            round_number: Round identifier.
            run_training: If True, generates training config (training is
                run separately via serve_ppo).

        Returns:
            RoundResult with pre/post scores and delta.
        """
        start = time.time()
        result = RoundResult(round_number=round_number)

        try:
            # Pre-eval
            if self.config.run_pre_eval:
                pre_dict = await self.run_pre_eval()
                if pre_dict:
                    result.pre_纠错率 = pre_dict.get("纠错率", 0)
                    result.metrics["pre_eval"] = pre_dict

            # Training (config generation only — serve_ppo runs separately)
            if run_training:
                config_str = self.generate_training_config()
                result.metrics["training_config"] = config_str
                result.training_completed = False  # Manual step

            # Post-eval
            if self.config.run_post_eval:
                post_dict = await self.run_post_eval()
                if post_dict:
                    result.post_纠错率 = post_dict.get("纠错率", 0)
                    result.Δ纠错率 = post_dict.get("Δ纠错率", 0)
                    result.metrics["post_eval"] = post_dict

            # Rubric evolution
            if self.config.run_rubric_evolution:
                evo_result = await self.evolve_rubrics()
                result.rubrics_updated = evo_result.get("new_rubrics", 0)
                result.metrics["rubric_evolution"] = evo_result

        except Exception as e:
            result.errors.append(str(e))
            logger.error("Round %d failed: %s", round_number, e)

        result.elapsed_seconds = time.time() - start
        return result

    def export_results(
        self, results: list[RoundResult], output_path: str
    ) -> str:
        """Export all round results as JSON."""
        data = {
            "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "配置": {
                "train_data": self.config.train_data_path,
                "test_data": self.config.test_data_path,
                "rubrics": self.config.rubrics_path,
                "model_server": self.config.model_server_url,
            },
            "轮次结果": [r.to_dict() for r in results],
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Results exported to %s", output_path)
        return str(path)


# --- Data loading helpers ---

def load_training_data(path: str | Path) -> list[dict]:
    """Load training pairs from JSONL.

    Each line has: 种子提示词, 类别, 错误回答, 纠错意见, 修正回答.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Training data not found: %s", p)
        return []

    items = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append({
                "prompt": d.get("种子提示词", d.get("prompt", "")),
                "category": d.get("类别", d.get("category", "")),
            })
    return items


def load_test_prompts(
    path: str | Path, max_prompts: int = 0
) -> list[dict]:
    """Load test prompts, deduplicating by seed prompt text.

    Args:
        path: Path to JSONL file.
        max_prompts: Max prompts to load (0 = all).

    Returns:
        List of {prompt, category} dicts.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Test prompts not found: %s", p)
        return []

    seen = set()
    items = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompt = d.get("种子提示词", d.get("prompt", ""))
            if prompt and prompt not in seen:
                seen.add(prompt)
                items.append({
                    "prompt": prompt,
                    "category": d.get("类别", d.get("category", "")),
                })
            if max_prompts > 0 and len(items) >= max_prompts:
                break

    return items


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(
        description="Trainable OpenClaw Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Eval-only mode (no GPU needed for pre-eval):\n"
            "  python -m trainable_openclaw.pipeline --eval-only --max-test-prompts 5\n\n"
            "  # Full round with rubric evolution:\n"
            "  python -m trainable_openclaw.pipeline --evolve-rubrics --output results.json\n\n"
            "  # Generate training config:\n"
            "  python -m trainable_openclaw.pipeline --gen-config --output train_config.txt\n"
        ),
    )
    parser.add_argument("--train-data", default="data/phase3_datasets/train_prompts.jsonl")
    parser.add_argument("--test-data", default="data/phase3_datasets/test_prompts.jsonl")
    parser.add_argument("--rubrics", default="data/rubrics_category.json")
    parser.add_argument("--db", default="data/conversations.db")
    parser.add_argument("--model-server", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--judge-model", default="deepseek-v4-flash")
    parser.add_argument("--max-test-prompts", type=int, default=0)
    parser.add_argument("--max-rubrics", type=int, default=8)
    parser.add_argument("--eval-only", action="store_true",
                       help="Run only pre-eval (no training, no post-eval)")
    parser.add_argument("--evolve-rubrics", action="store_true",
                       help="Run rubric evolution after evaluation")
    parser.add_argument("--gen-config", action="store_true",
                       help="Generate training config and exit")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--output", default="")

    args = parser.parse_args()

    config = PipelineConfig(
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        rubrics_path=args.rubrics,
        db_path=args.db,
        model_server_url=args.model_server,
        api_key=args.api_key,
        base_url=args.base_url,
        judge_model=args.judge_model,
        max_test_prompts=args.max_test_prompts,
        max_rubrics=args.max_rubrics,
        run_pre_eval=True,
        run_post_eval=not args.eval_only and not args.gen_config,
        run_rubric_evolution=args.evolve_rubrics,
    )

    pipeline = Pipeline(config)

    if args.gen_config:
        print(pipeline.generate_training_config(args.output))
        return

    if args.eval_only:
        config.run_post_eval = False
        config.run_rubric_evolution = False

    result = asyncio.run(pipeline.run_round(args.round))

    # Print summary
    print(f"\n=== Round {result.round_number} Results ===")
    print(f"Pre 纠错率:  {result.pre_纠错率:.3f}" if result.pre_纠错率 is not None else "Pre: N/A")
    print(f"Post 纠错率: {result.post_纠错率:.3f}" if result.post_纠错率 is not None else "Post: N/A")
    print(f"Δ纠错率:    {result.Δ纠错率:+.3f}" if result.Δ纠错率 is not None else "Δ: N/A")
    print(f"耗时:       {result.elapsed_seconds:.0f}s")
    if result.errors:
        print(f"错误:       {result.errors}")

    if args.output:
        pipeline.export_results([result], args.output)
        print(f"\n导出: {args.output}")


if __name__ == "__main__":
    main()
