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
        enable_thinking: bool = False,
        rubric_weights: list[float] | None = None,
        use_merged: bool = True,
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
        self.rubric_weights = rubric_weights
        self.use_merged = use_merged
        self._rubrics: list | None = None
        self._category_map: dict | None = None  # category → group mapping

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

            # Load category→group mapping if present
            if isinstance(data, dict) and "_category_to_group" in data:
                self._category_map = data["_category_to_group"]

            items = data.get("rubrics", data) if isinstance(data, dict) else data
            if isinstance(items, dict):
                items = []
            for item in items:
                r = Rubric.from_dict(item)
                if r.状态 == "活跃":
                    rubrics.append(r)

        if self.max_rubrics and len(rubrics) > self.max_rubrics:
            rubrics = self._select_diverse(rubrics, self.max_rubrics)

        logger.info("Loaded %d active rubrics from %s", len(rubrics), self.rubrics_path)
        return rubrics

    def _select_diverse(self, rubrics: list, max_n: int) -> list:
        """Select diverse rubrics by source pattern diversity."""
        seen_patterns = set()
        selected = []
        for r in rubrics:
            if r.来源模式 not in seen_patterns:
                seen_patterns.add(r.来源模式)
                selected.append(r)
        remaining = max_n - len(selected)
        if remaining > 0:
            for r in rubrics:
                if r not in selected:
                    selected.append(r)
                    if len(selected) >= max_n:
                        break
        return selected[:max_n]

    def _filter_by_category(self, rubrics: list, category: str) -> list:
        """Filter rubrics to those matching the given category group."""
        if not category or not self._category_map:
            return rubrics

        group = self._category_map.get(category, "")

        matching = []
        for r in rubrics:
            r_cats = getattr(r, "适用类别", []) or []
            r_group = getattr(r, "类别组", "") or ""
            if category in r_cats or group == r_group:
                matching.append(r)

        if matching:
            logger.info("Category '%s' (group='%s'): %d/%d rubrics matched",
                        category, group, len(matching), len(rubrics))
            return matching

        logger.info("Category '%s': no matching rubrics, using all %d", category, len(rubrics))
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
        category: str = "",
    ) -> list[float]:
        """Score N responses against M rubrics, return N reward values.

        Synchronous entry point for train_step (Ray actor context).
        Uses sync API calls to avoid asyncio event loop conflicts.

        Args:
            prompt: Original user prompt.
            responses: N response texts from GRPO generation.
            reward_mode: "mean" / "total" / "pass_fail" (overrides init).
            category: Prompt category for rubric filtering (e.g. "coding").

        Returns:
            List of N reward floats, suitable for GRPO rm_scores placement.
        """
        from trainable_openclaw.evaluation.judge import JudgeExecutor

        rubrics = self.rubrics
        if not rubrics:
            logger.warning("No rubrics available — returning zero rewards")
            return [0.0] * len(responses)

        # Filter rubrics by category if mapping available
        rubrics = self._filter_by_category(rubrics, category)

        judge = JudgeExecutor(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            enable_thinking=self.enable_thinking,
            use_merged=self.use_merged,
        )

        mode = reward_mode or self.reward_mode
        results = judge.score_answers_sync(
            prompt=prompt,
            answers=responses,
            rubrics=rubrics,
        )

        rewards = judge.compute_grpo_rewards(results, reward_mode=mode, weights=self.rubric_weights)
        return rewards

    # ------------------------------------------------------------------
    # Detailed scoring (returns full RewardResult, not just float)
    # ------------------------------------------------------------------

    def score_responses_detailed(
        self,
        prompt: str,
        responses: list[str],
        reward_mode: str = "",
        category: str = "",
    ) -> list[RewardResult]:
        """Like score_responses but returns full scoring details."""
        from trainable_openclaw.evaluation.judge import JudgeExecutor

        rubrics = self._filter_by_category(self.rubrics, category)
        judge = JudgeExecutor(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            enable_thinking=self.enable_thinking,
            use_merged=self.use_merged,
        )

        mode = reward_mode or self.reward_mode
        results = judge.score_answers_sync(
            prompt=prompt,
            answers=responses,
            rubrics=rubrics,
        )
        rewards = judge.compute_grpo_rewards(results, reward_mode=mode, weights=self.rubric_weights)

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
        rubric_weights=cfg.get("rubric_weights"),
    )
