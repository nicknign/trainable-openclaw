"""
Multi-turn Interaction Engine (Phase 1.5 S2).

Orchestrates User Sim Agent ↔ Qwen3-4B correction dialogues.
Handles turn management, termination conditions, and trajectory recording.

Supports two modes:
  - "live": Real Qwen3-4B inference API
  - "mock": DeepSeek simulates Qwen3-4B responses (for testing without GPU)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from trainable_openclaw.simulation.user_sim import (
    ReviewResult,
    TurnRecord,
    UserSimAgent,
    select_persona,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CorrectionTurn:
    """A single correction turn within a trajectory."""

    turn: int
    model_response: str
    review: ReviewResult


@dataclass
class Trajectory:
    """Complete correction dialogue trajectory."""

    seed_prompt: str
    category: str
    persona: str
    turns: list[CorrectionTurn] = field(default_factory=list)
    final_response: str = ""
    final_verdict: str = ""  # "direct_pass" | "corrected" | "partial" | "failed"
    correction_count: int = 0
    total_turns: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "seed_prompt": self.seed_prompt,
            "category": self.category,
            "persona": self.persona,
            "final_verdict": self.final_verdict,
            "correction_count": self.correction_count,
            "total_turns": self.total_turns,
            "turns": [
                {
                    "turn": t.turn,
                    "model_response": t.model_response,
                    "verdict": t.review.verdict,
                    "correction": t.review.correction,
                    "dimension": t.review.dimension,
                    "severity": t.review.severity,
                }
                for t in self.turns
            ],
            "final_response": self.final_response,
            "error": self.error,
        }

    def to_training_pairs(self) -> list[dict]:
        """Extract (bad, correction, good) training pairs from trajectory."""
        pairs = []
        for i, turn in enumerate(self.turns):
            if turn.review.verdict == "correct" and turn.review.correction:
                # The "bad" answer is this turn's response
                # The "good" answer is the next turn's response (or final)
                bad = turn.model_response
                correction = turn.review.correction
                if i + 1 < len(self.turns):
                    good = self.turns[i + 1].model_response
                else:
                    good = self.final_response
                if good:
                    pairs.append({
                        "bad": bad,
                        "correction": correction,
                        "good": good,
                        "dimension": turn.review.dimension,
                    })
        return pairs

    def to_rubric_seeds(self) -> list[dict]:
        """Extract correction dimensions as rubric seeds."""
        dimensions = {}
        for turn in self.turns:
            dim = turn.review.dimension
            if dim and turn.review.verdict == "correct":
                if dim not in dimensions:
                    dimensions[dim] = {"count": 0, "examples": []}
                dimensions[dim]["count"] += 1
                dimensions[dim]["examples"].append({
                    "bad_snippet": turn.model_response[:200],
                    "correction": turn.review.correction[:200],
                })
        return [{"dimension": k, **v} for k, v in dimensions.items()]


# ---------------------------------------------------------------------------
# Inference client abstraction
# ---------------------------------------------------------------------------


class InferenceClient:
    """Client for Qwen3-4B inference (or mock)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "default",
        api_key: str = "",
        mock_model: str = "deepseek-v4-flash",
        mock_api_key: Optional[str] = None,
        mock_base_url: Optional[str] = None,
    ):
        self.base_url = base_url
        self.model_name = model
        self.api_key = api_key
        self.mock_model = mock_model
        self.mock_api_key = mock_api_key
        self.mock_base_url = mock_base_url
        self._client = None

    async def generate(self, messages: list[dict], max_tokens: int = 2048) -> str:
        """Generate a response from Qwen3-4B (or mock using DeepSeek)."""
        if self._client is None:
            from openai import AsyncOpenAI

            if self.mock_api_key:
                # Mock mode: use DeepSeek to simulate Qwen3-4B
                base = self.mock_base_url or "https://api.deepseek.com"
                self._client = AsyncOpenAI(
                    api_key=self.mock_api_key,
                    base_url=base,
                )
                self._mock = True
            else:
                # Live mode: real Qwen3-4B API
                self._client = AsyncOpenAI(
                    api_key=self.api_key or "not-needed",
                    base_url=self.base_url,
                )
                self._mock = False

        if getattr(self, "_mock", False):
            # Use DeepSeek with a system prompt that simulates Qwen3-4B behavior
            mock_system = (
                "你是一个AI助手（Qwen3-4B级别的模型）。请根据用户的请求生成回答。"
                "你的能力有限，可能会犯一些错误。请自然地回答，不需要刻意犯错。"
                "你就是你自己的水平，不需要说明你是哪个模型。"
            )
            mock_messages = [{"role": "system", "content": mock_system}] + messages
            response = await self._client.chat.completions.create(
                model=self.mock_model,
                messages=mock_messages,
                temperature=0.8,  # Higher temp to simulate model variance
                max_tokens=max_tokens,
            )
        else:
            # Real Qwen3-4B API
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.8,
                max_tokens=max_tokens,
            )

        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Interaction Engine
# ---------------------------------------------------------------------------


class InteractionEngine:
    """Orchestrates multi-turn User Sim ↔ Model correction dialogues.

    Usage::

        user_sim = UserSimAgent(persona="coder_zhang")
        inference = InferenceClient(mock_api_key="sk-xxx")  # mock mode
        engine = InteractionEngine(user_sim, inference, max_turns=5)

        traj = await engine.run("Write a sorting function in Python")
        print(traj.final_verdict)  # "direct_pass" | "corrected" | ...
        pairs = traj.to_training_pairs()  # (bad, correction, good) triples
    """

    def __init__(
        self,
        user_sim: UserSimAgent,
        inference_client: InferenceClient,
        max_turns: int = 5,
        max_corrections: int = 3,
    ):
        self.user_sim = user_sim
        self.inference = inference_client
        self.max_turns = max_turns
        self.max_corrections = max_corrections

    async def run(
        self,
        seed_prompt: str,
        category: str = "general",
    ) -> Trajectory:
        """Run a complete correction dialogue.

        Flow:
        1. Model generates initial response
        2. User Sim reviews → pass or correct
        3. If correct: model receives correction → regenerates
        4. Repeat until pass or max turns/corrections reached
        5. Final check by User Sim
        """
        traj = Trajectory(
            seed_prompt=seed_prompt,
            category=category,
            persona=self.user_sim.persona,
        )

        # Conversation history for the model (messages format)
        model_history = [
            {"role": "user", "content": seed_prompt},
        ]

        # Structured history for User Sim review context
        review_history: list[dict] = []

        correction_count = 0
        current_response = ""

        try:
            # Turn 1: initial generation
            logger.info(f"[Turn 1] Generating initial response for: {seed_prompt[:80]}...")
            current_response = await self.inference.generate(model_history)
            model_history.append({"role": "assistant", "content": current_response})

            # Review cycle
            for turn_idx in range(1, self.max_turns + 1):
                logger.info(f"[Review {turn_idx}] User Sim ({self.user_sim.persona}) reviewing...")

                # User Sim reviews
                result = await self.user_sim.review(
                    task=seed_prompt,
                    model_response=current_response,
                    history=review_history,
                )

                traj.turns.append(CorrectionTurn(
                    turn=turn_idx,
                    model_response=current_response,
                    review=result,
                ))

                if result.verdict == "pass":
                    logger.info(f"[Turn {turn_idx}] User Sim: PASS")
                    traj.final_response = current_response
                    if correction_count == 0:
                        traj.final_verdict = "direct_pass"
                    else:
                        traj.final_verdict = "corrected"
                    break

                # Need correction
                correction_count += 1
                logger.info(
                    f"[Turn {turn_idx}] User Sim: CORRECT "
                    f"({result.dimension}, {result.severity})"
                )

                if correction_count > self.max_corrections:
                    logger.info(f"Max corrections ({self.max_corrections}) reached")
                    # Final check to see if we're at least partially there
                    satisfied, summary = await self.user_sim.final_check(
                        task=seed_prompt,
                        final_response=current_response,
                        history=review_history,
                    )
                    if satisfied:
                        traj.final_verdict = "corrected"
                    else:
                        traj.final_verdict = "partial"
                    traj.final_response = current_response
                    break

                # Add correction to history
                correction_msg = (
                    f"关于你之前的回答，我发现一个问题：\n{result.correction}\n"
                    f"请修改你的回答，解决这个问题。"
                )
                review_history.append({
                    "speaker": "user_sim",
                    "content": result.correction,
                })
                model_history.append({"role": "user", "content": correction_msg})

                # Regenerate with correction
                logger.info(f"[Turn {turn_idx + 1}] Regenerating with correction...")
                current_response = await self.inference.generate(model_history)
                model_history.append({"role": "assistant", "content": current_response})

            else:
                # Exhausted max turns
                logger.info(f"Max turns ({self.max_turns}) reached")
                satisfied, summary = await self.user_sim.final_check(
                    task=seed_prompt,
                    final_response=current_response,
                    history=review_history,
                )
                traj.final_response = current_response
                if satisfied:
                    traj.final_verdict = "corrected"
                else:
                    traj.final_verdict = "partial" if correction_count > 0 else "failed"

        except Exception as e:
            logger.error(f"Interaction failed: {e}")
            traj.error = str(e)
            traj.final_verdict = "failed"

        traj.correction_count = correction_count
        traj.total_turns = len(traj.turns)
        return traj


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


async def run_seed_prompts(
    seed_file: str,
    output_file: str,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-v4-flash",
    mock: bool = True,
    max_prompts: int = 0,
    max_turns: int = 5,
    qwen_url: str = "http://localhost:8000",
) -> list[Trajectory]:
    """Run simulation on a batch of seed prompts.

    Args:
        seed_file: Path to seed_prompts.jsonl (from S1).
        output_file: Path to write trajectories.jsonl.
        api_key: DeepSeek API key.
        base_url: DeepSeek API base URL.
        model: Model name for User Sim.
        mock: If True, use DeepSeek to simulate Qwen3-4B responses.
        max_prompts: Max prompts to process (0 = all).
        max_turns: Max correction turns per prompt.
        qwen_url: Qwen3-4B API URL (only used when mock=False).

    Returns:
        List of completed trajectories.
    """
    import os as _os

    # Load seed prompts
    seeds = []
    with open(seed_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seeds.append(json.loads(line))

    if max_prompts > 0:
        seeds = seeds[:max_prompts]

    logger.info(f"Loaded {len(seeds)} seed prompts from {seed_file}")

    trajectories: list[Trajectory] = []
    output_dir = _os.path.dirname(output_file) or "."
    _os.makedirs(output_dir, exist_ok=True)

    # Open output file for streaming writes
    out_f = open(output_file, "w", encoding="utf-8")

    try:
        for i, seed in enumerate(seeds):
            prompt = seed["prompt"]
            category = seed.get("category", "general")
            persona = select_persona(category)

            logger.info(
                f"\n{'='*60}\n"
                f"[{i+1}/{len(seeds)}] category={category} persona={persona}\n"
                f"  prompt: {prompt[:100]}..."
            )

            user_sim = UserSimAgent(
                persona=persona,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )

            inference = InferenceClient(
                base_url=qwen_url,
                mock_api_key=api_key if mock else None,
                mock_base_url=base_url if mock else None,
            )

            engine = InteractionEngine(
                user_sim=user_sim,
                inference_client=inference,
                max_turns=max_turns,
            )

            traj = await engine.run(prompt, category=category)
            trajectories.append(traj)

            # Write to output
            out_f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
            out_f.flush()

            logger.info(
                f"  Result: {traj.final_verdict} "
                f"(corrections={traj.correction_count}, turns={traj.total_turns})"
            )
    finally:
        out_f.close()

    # Summary
    verdicts = {}
    for t in trajectories:
        verdicts[t.final_verdict] = verdicts.get(t.final_verdict, 0) + 1
    logger.info(f"\n{'='*60}")
    logger.info(f"Batch complete: {len(trajectories)} trajectories")
    logger.info(f"  Results: {verdicts}")

    return trajectories
