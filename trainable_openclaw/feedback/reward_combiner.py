"""
Three-layer reward combiner.

Combines Layer 1 (deterministic verification), Layer 2 (user signals),
and Layer 3 (LLM judge) into a single final reward.

Default weights: w1=0.5, w2=0.3, w3=0.2
When layer3 is None, its weight is redistributed proportionally to layer1 and layer2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from trainable_openclaw.feedback.deterministic_verifier import compute_layer1_reward
from trainable_openclaw.feedback.signal_extractor import (
    FeedbackSignal,
    extract_signals_from_messages,
    compute_layer2_reward,
)


@dataclass
class CombinedReward:
    """Result of combining all three reward layers."""

    layer1: float
    layer2: float
    layer3: float | None  # None if judge was not called
    final: float
    weights: dict[str, float] = field(default_factory=lambda: {"layer1": 0.5, "layer2": 0.3, "layer3": 0.2})
    breakdown: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)  # Debugging metadata

    def to_dict(self) -> dict:
        return {
            "layer1": self.layer1,
            "layer2": self.layer2,
            "layer3": self.layer3,
            "final": self.final,
            "weights": self.weights,
            "breakdown": self.breakdown,
            "metadata": self.metadata,
        }


def combine(
    layer1: float,
    layer2: float,
    layer3: float | None = None,
    weights: dict[str, float] | None = None,
) -> CombinedReward:
    """Combine three layer rewards with configurable weights.

    If layer3 is None (judge not called), its weight is redistributed
    proportionally to layer1 and layer2.

    Args:
        layer1: Deterministic verification score (0-1)
        layer2: User feedback score (0-1)
        layer3: LLM judge score (0-1), or None if judge was not called
        weights: Override default weights

    Returns:
        CombinedReward with final weighted score.
    """
    w = dict(weights) if weights else {"layer1": 0.5, "layer2": 0.3, "layer3": 0.2}

    if layer3 is None:
        # Redistribute layer3 weight to layer1 and layer2 proportionally
        l3_weight = w.get("layer3", 0.0)
        l1_weight = w.get("layer1", 0.5)
        l2_weight = w.get("layer2", 0.3)
        total = l1_weight + l2_weight
        if total > 0:
            w["layer1"] = l1_weight + l3_weight * (l1_weight / total)
            w["layer2"] = l2_weight + l3_weight * (l2_weight / total)
        w["layer3"] = 0.0
        final = w["layer1"] * layer1 + w["layer2"] * layer2
    else:
        final = w["layer1"] * layer1 + w["layer2"] * layer2 + w["layer3"] * layer3

    return CombinedReward(
        layer1=layer1,
        layer2=layer2,
        layer3=layer3,
        final=round(final, 4),
        weights=w,
        breakdown={
            "layer1_weighted": round(w["layer1"] * layer1, 4),
            "layer2_weighted": round(w["layer2"] * layer2, 4),
            "layer3_weighted": round(w.get("layer3", 0) * (layer3 or 0), 4),
        },
        metadata={
            "layer3_was_none": layer3 is None,
            "original_weights": dict(weights) if weights else {"layer1": 0.5, "layer2": 0.3, "layer3": 0.2},
        },
    )


def compute_full_reward(
    trajectory: list[dict],
    call_judge: bool = False,
    judge_executor=None,
    weights: dict[str, float] | None = None,
    signals: list[FeedbackSignal] | None = None,
) -> CombinedReward:
    """Compute the full 3-layer reward for a trajectory.

    Always computes L1 (deterministic verification) and L2 (user feedback signals).
    Optionally calls L3 judge (LLM-based evaluation).

    Args:
        trajectory: List of message/tool dicts from a conversation.
        call_judge: If True, invoke the LLM judge for Layer 3.
        judge_executor: JudgeExecutor instance (required if call_judge=True).
        weights: Custom layer weights.
        signals: Pre-computed signals (if None, extracted from trajectory).

    Returns:
        CombinedReward with all computed layers.
    """
    # Layer 1: Deterministic verification
    l1 = compute_layer1_reward(trajectory)

    # Layer 2: User feedback signals
    if signals is None:
        messages = [
            m for m in trajectory
            if m.get("role") in ("user", "assistant")
        ]
        signals = extract_signals_from_messages(messages)
    l2 = compute_layer2_reward(signals)

    # Layer 3: LLM Judge (optional)
    l3: float | None = None
    if call_judge:
        if judge_executor is None:
            raise ValueError("judge_executor is required when call_judge=True")
        # Extract the last assistant answer for judging
        assistant_msgs = [m for m in trajectory if m.get("role") == "assistant"]
        last_answer = assistant_msgs[-1].get("content", "") if assistant_msgs else ""

        # Use sync judge API for compatibility
        try:
            rubrics = getattr(judge_executor, 'rubrics', []) or []
            if rubrics:
                scores = judge_executor.score_answer_sync(last_answer, rubrics)
                l3 = scores.mean_score / 10.0  # Normalize to 0-1
            else:
                l3 = 0.5  # No rubrics, default neutral
        except Exception:
            l3 = 0.5  # Judge failed, default neutral

    return combine(l1, l2, l3, weights)
