"""
C1: Reward Bridge — B3 Judge → GRPO reward (Phase 3)

Bridges the evaluation system (B3 JudgeExecutor) to the GRPO training loop.
Designed to work inside serve_ppo's train_step (Ray actor), where async HTTP
calls to DeepSeek API are wrapped via asyncio.run().

Usage inside train_step::

    bridge = RewardBridge(rubrics_path="data/rubrics.json", api_key="...")
    prompt = "写一个排序函数"
    responses = ["def sort(arr): ...", "def quick_sort(arr): ..."]
    rewards = bridge.score_responses(prompt, responses, reward_mode="mean")
    # rewards: [0.72, 0.85]
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RewardResult:
    """Result of scoring one response."""
    response: str
    reward: float
    rubric_scores: list[float]
    mean_score: float
    details: list[dict]


class RewardBridge:
    """Synchronous wrapper around B3 JudgeExecutor for GRPO reward computation.

    Callers (train_step in Ray actor) use the synchronous score_responses().
    Internally it uses asyncio.run() to drive the async JudgeExecutor.
    """

    def __init__(
        self,
        rubrics_path: str = "data/rubrics.json",
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        max_rubrics: int = 0,
        reward_mode: str = "mean",
        enable_thinking: bool = True,
    ):
        self.rubrics_path = Path(rubrics_path)
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        self.model = model or "deepseek-v4-flash"
        self.max_rubrics = max_rubrics
        self.reward_mode = reward_mode
        self.enable_thinking = enable_thinking
        self._rubrics: list | None = None

    # ------------------------------------------------------------------
    # Rubric loading (lazy, cached)
    # ------------------------------------------------------------------

    def _load_rubrics(self) -> list:
        if not self.rubrics_path.exists():
            logger.warning("Rubrics file not found: %s — rewards will be 0", self.rubrics_path)
            return []

        from trainable_openclaw.evaluation.rubric import Rubric
        import json

        rubrics = []
        with open(self.rubrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                r = Rubric.from_dict(item)
                if r.状态 == "活跃":
                    rubrics.append(r)

        if self.max_rubrics and len(rubrics) > self.max_rubrics:
            # Select diverse rubrics by source pattern diversity
            seen_patterns = set()
            selected = []
            for r in rubrics:
                if r.来源模式 not in seen_patterns:
                    seen_patterns.add(r.来源模式)
                    selected.append(r)
            remaining = self.max_rubrics - len(selected)
            if remaining > 0:
                for r in rubrics:
                    if r not in selected:
                        selected.append(r)
                        if len(selected) >= self.max_rubrics:
                            break
            rubrics = selected[:self.max_rubrics]

        logger.info("Loaded %d active rubrics from %s", len(rubrics), self.rubrics_path)
        return rubrics

    @property
    def rubrics(self) -> list:
        if self._rubrics is None:
            self._rubrics = self._load_rubrics()
        return self._rubrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_responses(
        self,
        prompt: str,
        responses: list[str],
        reward_mode: str = "",
    ) -> list[float]:
        """Score N responses against M rubrics, return N reward values.

        Synchronous entry point for train_step (Ray actor context).

        Args:
            prompt: Original user prompt.
            responses: N response texts from GRPO generation.
            reward_mode: "mean" / "total" / "pass_fail" (overrides init).

        Returns:
            List of N reward floats, suitable for GRPO rm_scores placement.
        """
        mode = reward_mode or self.reward_mode
        return asyncio.run(self._score_async(prompt, responses, mode))

    async def _score_async(
        self,
        prompt: str,
        responses: list[str],
        reward_mode: str,
    ) -> list[float]:
        from trainable_openclaw.evaluation.judge import JudgeExecutor

        rubrics = self.rubrics
        if not rubrics:
            logger.warning("No rubrics available — returning zero rewards")
            return [0.0] * len(responses)

        judge = JudgeExecutor(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            enable_thinking=self.enable_thinking,
        )

        results = await judge.score_answers(
            prompt=prompt,
            answers=responses,
            rubrics=rubrics,
        )

        rewards = judge.compute_grpo_rewards(results, reward_mode=reward_mode)
        return rewards

    # ------------------------------------------------------------------
    # Detailed scoring (returns full RewardResult, not just float)
    # ------------------------------------------------------------------

    def score_responses_detailed(
        self,
        prompt: str,
        responses: list[str],
        reward_mode: str = "",
    ) -> list[RewardResult]:
        """Like score_responses but returns full scoring details."""
        mode = reward_mode or self.reward_mode
        return asyncio.run(self._score_detailed_async(prompt, responses, mode))

    async def _score_detailed_async(
        self,
        prompt: str,
        responses: list[str],
        reward_mode: str,
    ) -> list[RewardResult]:
        from trainable_openclaw.evaluation.judge import JudgeExecutor

        rubrics = self.rubrics
        judge = JudgeExecutor(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            enable_thinking=self.enable_thinking,
        )

        results = await judge.score_answers(
            prompt=prompt,
            answers=responses,
            rubrics=rubrics,
        )
        rewards = judge.compute_grpo_rewards(results, reward_mode=reward_mode)

        out = []
        for i, (r, result) in enumerate(zip(rewards, results)):
            out.append(RewardResult(
                response=responses[i],
                reward=r,
                rubric_scores=result.get("分数向量", []),
                mean_score=result.get("平均分", 0),
                details=result.get("评分", []),
            ))
        return out


# ---------------------------------------------------------------------------
# Convenience: create a bridge from env/config
# ---------------------------------------------------------------------------

def create_reward_bridge(
    rubrics_path: str = "data/rubrics.json",
    config: dict | None = None,
) -> RewardBridge:
    """Create a RewardBridge from environment + optional config overrides.

    Reads DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL from environment.
    """
    cfg = config or {}
    return RewardBridge(
        rubrics_path=cfg.get("rubrics_path", rubrics_path),
        api_key=cfg.get("api_key", ""),
        base_url=cfg.get("base_url", ""),
        model=cfg.get("model", ""),
        max_rubrics=cfg.get("max_rubrics", 0),
        reward_mode=cfg.get("reward_mode", "mean"),
    )
