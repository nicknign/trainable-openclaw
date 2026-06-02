"""
C2: 纠错率 (Correction Rate) — Phase 3 key evaluation metric.

Measures how often a User Sim Agent finds flaws in the model's responses.
The core insight: a better model needs fewer corrections.

Metric:
    纠错率 = (total - direct_pass) / total
    Δ纠错率 = 纠错率_post - 纠错率_pre  (negative = improvement)

Usage:
    evaluator = CorrectionRateEvaluator(
        model_server_url="http://localhost:8000/v1",
        api_key="sk-...",
    )
    result = await evaluator.evaluate(test_prompts)
    # result.纠错率, result.Δ纠错率, result.per_prompt_details
"""

from __future__ import annotations

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
class PromptEvalResult:
    """Result for a single test prompt."""
    prompt: str
    category: str = ""
    outcome: str = ""  # "直接通过" | "纠错后通过" | "失败"
    correction_rounds: int = 0
    corrections: list[str] = field(default_factory=list)
    final_response: str = ""
    error: str = ""


@dataclass
class CorrectionRateResult:
    """Aggregate correction rate evaluation result."""
    total: int = 0
    直接通过: int = 0
    纠错后通过: int = 0
    失败: int = 0
    纠错率: float = 0.0
    avg_rounds: float = 0.0
    per_prompt: list[PromptEvalResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    # For delta computation
    Δ纠错率: float = 0.0
    pre_纠错率: float = 0.0

    def to_dict(self) -> dict:
        return {
            "总数": self.total,
            "直接通过": self.直接通过,
            "纠错后通过": self.纠错后通过,
            "失败": self.失败,
            "纠错率": round(self.纠错率, 4),
            "平均纠正轮次": round(self.avg_rounds, 2),
            "耗时秒": round(self.elapsed_seconds, 1),
        }


class CorrectionRateEvaluator:
    """Evaluate model quality via User Sim Agent correction rate.

    Multi-turn evaluation: User Sim reviews model response → if flawed,
    provides correction → model revises → repeat up to max_rounds.

    The correction rate is the fraction of prompts that need ANY correction.
    A lower rate means the model directly satisfies users more often.
    """

    MAX_CORRECTION_ROUNDS = 3

    def __init__(
        self,
        model_server_url: str = "http://localhost:8000/v1",
        api_key: str = "",
        base_url: str = "",
        sim_model: str = "deepseek-v4-flash",
        max_concurrent: int = 5,
    ):
        self.model_server_url = model_server_url.rstrip("/")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        self.sim_model = sim_model
        self.max_concurrent = max_concurrent
        self._http_client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        test_prompts: list[dict],
    ) -> CorrectionRateResult:
        """Evaluate correction rate on test prompts.

        Args:
            test_prompts: [{"prompt": str, "category": str}, ...]

        Returns:
            CorrectionRateResult with aggregate stats and per-prompt details.
        """
        t_start = time.time()
        sem = asyncio.Semaphore(self.max_concurrent)

        async def _eval_one(prompt_item: dict) -> PromptEvalResult:
            async with sem:
                return await self._evaluate_single(prompt_item)

        tasks = [_eval_one(p) for p in test_prompts]
        results = await asyncio.gather(*tasks)

        # Aggregate
        direct = sum(1 for r in results if r.outcome == "直接通过")
        corrected = sum(1 for r in results if r.outcome == "纠错后通过")
        failed = sum(1 for r in results if r.outcome == "失败")
        total = len(results)
        correction_rate = (total - direct) / total if total > 0 else 0.0
        avg_rounds = (
            sum(r.correction_rounds for r in results) / total if total > 0 else 0.0
        )

        elapsed = time.time() - t_start
        logger.info(
            "Correction rate: %.2f%% (%d/%d direct pass, %d corrected, %d failed) in %.1fs",
            correction_rate * 100, direct, total, corrected, failed, elapsed,
        )

        return CorrectionRateResult(
            total=total,
            直接通过=direct,
            纠错后通过=corrected,
            失败=failed,
            纠错率=correction_rate,
            avg_rounds=avg_rounds,
            per_prompt=results,
            elapsed_seconds=elapsed,
        )

    async def evaluate_delta(
        self,
        test_prompts: list[dict],
        pre_result: CorrectionRateResult | None = None,
    ) -> CorrectionRateResult:
        """Evaluate and compute Δ纠错率 against a pre-training baseline.

        Args:
            test_prompts: Same prompts used in baseline evaluation.
            pre_result: Result from pre-training evaluation.

        Returns:
            Result with Δ纠错率 populated.
        """
        result = await self.evaluate(test_prompts)
        if pre_result is not None:
            result.pre_纠错率 = pre_result.纠错率
            result.Δ纠错率 = result.纠错率 - pre_result.纠错率
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _evaluate_single(self, prompt_item: dict) -> PromptEvalResult:
        prompt = prompt_item.get("prompt", prompt_item.get("种子提示词", ""))
        category = prompt_item.get("category", prompt_item.get("类别", ""))

        if not prompt:
            return PromptEvalResult(
                prompt="", category=category, outcome="失败",
                error="Empty prompt",
            )

        from trainable_openclaw.simulation.user_sim import UserSimAgent, select_persona

        persona = select_persona(category)
        user_sim = UserSimAgent(
            persona=persona,
            model=self.sim_model,
            api_key=self.api_key,
            base_url=self.base_url,
        )

        history: list[dict] = []
        corrections: list[str] = []
        rounds = 0
        current_response = ""

        try:
            # Initial model response
            current_response = await self._call_model(prompt)
            if not current_response:
                return PromptEvalResult(
                    prompt=prompt, category=category, outcome="失败",
                    error="Model returned empty response",
                )

            for round_idx in range(self.MAX_CORRECTION_ROUNDS):
                # User Sim reviews
                review = await user_sim.review(prompt, current_response, history)

                if review.verdict == "pass":
                    outcome = "直接通过" if round_idx == 0 else "纠错后通过"
                    return PromptEvalResult(
                        prompt=prompt, category=category, outcome=outcome,
                        correction_rounds=round_idx, corrections=corrections,
                        final_response=current_response,
                    )

                # Needs correction
                rounds += 1
                corrections.append(review.correction)

                history.append({"speaker": "assistant", "content": current_response})
                history.append({"speaker": "user", "content": review.correction})

                # Model revises
                revised = await self._call_model_with_history(prompt, history)
                if not revised:
                    return PromptEvalResult(
                        prompt=prompt, category=category, outcome="失败",
                        correction_rounds=rounds, corrections=corrections,
                        final_response=current_response,
                        error="Model revision returned empty",
                    )
                current_response = revised

            # Max rounds reached — final check
            satisfied, summary = await user_sim.final_check(
                prompt, current_response, history
            )
            if satisfied:
                return PromptEvalResult(
                    prompt=prompt, category=category, outcome="纠错后通过",
                    correction_rounds=rounds, corrections=corrections,
                    final_response=current_response,
                )
            else:
                return PromptEvalResult(
                    prompt=prompt, category=category, outcome="失败",
                    correction_rounds=rounds, corrections=corrections,
                    final_response=current_response, error=summary,
                )

        except Exception as e:
            logger.error(f"Evaluation failed for prompt [{prompt[:80]}...]: {e}")
            return PromptEvalResult(
                prompt=prompt, category=category, outcome="失败",
                correction_rounds=rounds, corrections=corrections,
                final_response=current_response, error=str(e),
            )

    async def _call_model(self, prompt: str) -> str:
        """Send a prompt to the model server, get response text."""
        import aiohttp

        url = f"{self.model_server_url}/chat/completions"
        payload = {
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.7,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(f"Model server error {resp.status}: {body[:300]}")
                        return ""
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            return ""

    async def _call_model_with_history(
        self, prompt: str, history: list[dict]
    ) -> str:
        """Send prompt with conversation history to model server."""
        import aiohttp

        url = f"{self.model_server_url}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        # Convert history speaker→role for OpenAI chat API compatibility
        for h in history:
            role = "assistant" if h.get("speaker") == "assistant" else "user"
            messages.append({"role": role, "content": h.get("content", "")})

        payload = {
            "model": "qwen3-4b",
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.7,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        return ""
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Model revision call failed: {e}")
            return ""


# ---------------------------------------------------------------------------
# Convenience: load test prompts from file
# ---------------------------------------------------------------------------

def load_test_prompts(
    path: str | Path = "data/test_eval/training_pairs.jsonl",
    max_prompts: int = 0,
) -> list[dict]:
    """Load test prompts from training pairs JSONL.

    Extracts unique seed prompts (the "种子提示词" field).
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Test prompts file not found: %s", path)
        return []

    seen = set()
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            seed = item.get("种子提示词", "").strip()
            if seed and seed not in seen:
                seen.add(seed)
                prompts.append({
                    "prompt": seed,
                    "category": item.get("类别", ""),
                })

    if max_prompts and max_prompts < len(prompts):
        prompts = prompts[:max_prompts]

    logger.info("Loaded %d unique test prompts from %s", len(prompts), path)
    return prompts
