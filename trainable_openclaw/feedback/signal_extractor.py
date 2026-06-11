"""
Layer 2: User feedback signals extracted from conversation.

FREE — rule-based pattern matching on user messages to detect
positive, negative, correction, and abandonment signals.
No API calls required.

Usage::

    signals = extract_signals_from_messages(messages)
    reward = compute_layer2_reward(signals)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class FeedbackSignal:
    """User feedback extracted from conversation."""

    signal_type: str  # "positive" | "negative" | "correction" | "neutral" | "abandoned"
    confidence: float  # 0-1, how confident the extraction is
    evidence: str  # The user message(s) that produced this signal
    details: str  # Human-readable explanation


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

POSITIVE_PATTERNS: list[tuple[str, float]] = [
    # Strong satisfaction
    (r"\b(?:that'?s (?:perfect|exactly|great|awesome|amazing|fantastic|wonderful|brilliant))\b", 0.9),
    # Gratitude with emphasis
    (r"\b(?:thank you so much|thanks a lot|much appreciated)\b", 0.85),
    # Gratitude
    (r"\b(?:thanks|thank you|thx)\b", 0.7),
    # Strong satisfaction (standalone)
    (r"\b(?:perfect|exactly|great|awesome|wonderful)\b", 0.8),
    # Confirmation of correctness
    (r"\b(?:that works|it works|working now|all good|looks good)\b", 0.8),
    # Chinese positive
    (r"\b(?:解决了|搞定了|可以了|好的谢谢|完美|很好|非常好|不错|太棒了|很棒)\b", 0.8),
    # Chinese gratitude
    (r"\b(?:谢谢|多谢|感谢)\b", 0.75),
]

NEGATIVE_PATTERNS: list[tuple[str, float]] = [
    # Explicit correction needed
    (r"\b(?:try again|redo|rephrase|start over|do over)\b", 0.8),
    # Direct negation
    (r"\b(?:no|wrong|not (?:right|correct|what))\b", 0.6),
    # Chinese negative
    (r"\b(?:不对|错了|不是|不行|重新)\b", 0.7),
    # Chinese dissatisfaction
    (r"\b(?:还是不对|还没对|根本没|完全错|一塌糊涂)\b", 0.85),
    # Complaint
    (r"\b(?:don'?t work|doesn'?t work|not working|broken|useless)\b", 0.75),
    # This is wrong
    (r"\b(?:that'?s wrong|that'?s not right|that'?s incorrect|this is wrong)\b", 0.8),
]

CORRECTION_PATTERNS: list[tuple[str, float]] = [
    # Specific substitution
    (r"\b(?:it should be|change .*? to |instead of|replace .*? with)\b", 0.8),
    # Fix directive
    (r"\b(?:fix .*? by|correct .*? to|the correct .*? is)\b", 0.85),
    # Chinese correction patterns
    (r"\b(?:应该是|改成|换成|用.*?代替)\b", 0.8),
    # Explicit instruction to change
    (r"\b(?:you should (?:use|write|do|say))\b", 0.75),
    # Typo / code fix
    (r"\b(?:typo|syntax error|bug|missing .*? should be)\b", 0.8),
]

# User message ending patterns that suggest abandonment
ABANDONMENT_INDICATORS: list[str] = [
    r"\b(?:never mind|forget it|leave it|stop)\b",
    r"\b(?:算了|不管了|不用了|取消了)\b",
]

# Sessions where the last message is from the assistant (user stopped responding)
# are considered potentially abandoned if no user message follows within the session.


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def extract_signals_from_messages(messages: list[dict]) -> list[FeedbackSignal]:
    """Extract feedback signals from a conversation.

    Analyzes user messages (role="user") for positive/negative/correction/abandonment
    indicators. Returns list of detected signals with confidence scores.

    Multiple signals can be extracted from a single message.
    """
    signals: list[FeedbackSignal] = []

    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return signals

    for i, msg in enumerate(user_messages):
        content = msg.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue

        # Check positive patterns
        for pattern, confidence in POSITIVE_PATTERNS:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                signals.append(FeedbackSignal(
                    signal_type="positive",
                    confidence=confidence,
                    evidence=content[:200],
                    details=f"Matched positive pattern: '{match.group()}' in user message {i + 1}",
                ))
                break  # One positive signal per message is enough

        # Check correction patterns (check before negative, since corrections are more specific)
        for pattern, confidence in CORRECTION_PATTERNS:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                signals.append(FeedbackSignal(
                    signal_type="correction",
                    confidence=confidence,
                    evidence=content[:200],
                    details=f"Matched correction pattern: '{match.group()}' in user message {i + 1}",
                ))
                break

        # Check negative patterns (only if no correction was found for this message)
        has_correction = any(
            s.signal_type == "correction"
            and content[:200] == s.evidence[:200]
            for s in signals
        )
        if not has_correction:
            for pattern, confidence in NEGATIVE_PATTERNS:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    signals.append(FeedbackSignal(
                        signal_type="negative",
                        confidence=confidence,
                        evidence=content[:200],
                        details=f"Matched negative pattern: '{match.group()}' in user message {i + 1}",
                    ))
                    break

        # Check abandonment patterns
        for pattern in ABANDONMENT_INDICATORS:
            if re.search(pattern, content, re.IGNORECASE):
                signals.append(FeedbackSignal(
                    signal_type="abandoned",
                    confidence=0.85,
                    evidence=content[:200],
                    details=f"Matched abandonment pattern in user message {i + 1}",
                ))
                break

    # Check for session abandonment: last message is from assistant
    if messages and messages[-1].get("role") == "assistant":
        signals.append(FeedbackSignal(
            signal_type="abandoned",
            confidence=0.4,
            evidence="Session ended with assistant message, no user acknowledgment",
            details="Session ended without user response — low confidence abandonment",
        ))

    # If no signals at all, mark as neutral
    if not signals:
        signals.append(FeedbackSignal(
            signal_type="neutral",
            confidence=0.5,
            evidence="",
            details="No strong positive/negative/correction signals detected",
        ))

    return signals


# ---------------------------------------------------------------------------
# Layer 2 reward computation
# ---------------------------------------------------------------------------


def compute_layer2_reward(signals: list[FeedbackSignal]) -> float:
    """Compute Layer 2 reward from extracted signals.

    Rules:
    - positive signal: +1.0 per signal (capped at 1.0)
    - correction signal: depends — if the agent then fixed it, +0.5; else 0.0
      (correction is treated as potential improvement signal; detection of fix
       would require trajectory analysis, so default to +0.3 as partial credit)
    - negative signal: -0.5 per signal (floor at 0.0)
    - abandoned: 0.0

    Weighted by confidence. Multiple signals averaged.
    """
    if not signals:
        return 0.5  # Default neutral reward

    rewards: list[float] = []

    for sig in signals:
        if sig.signal_type == "positive":
            rewards.append(1.0 * sig.confidence)
        elif sig.signal_type == "correction":
            # Correction: partial credit — user engaged enough to provide feedback
            rewards.append(0.3 * sig.confidence)
        elif sig.signal_type == "negative":
            rewards.append(-0.5 * sig.confidence)
        elif sig.signal_type == "abandoned":
            rewards.append(0.0)
        elif sig.signal_type == "neutral":
            rewards.append(0.5)
        else:
            rewards.append(0.5)

    avg = sum(rewards) / len(rewards)
    return max(0.0, min(1.0, avg))


def _text_similarity(a: str, b: str) -> float:
    """Simple similarity between two strings based on word overlap."""
    words_a = set(re.findall(r"\w+", a.lower()))
    words_b = set(re.findall(r"\w+", b.lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def detect_retry_pattern(messages: list[dict]) -> list[dict]:
    """Detect when user re-asks the same or similar question.

    Indicates the agent's previous answer was unsatisfactory.
    Returns retry events as list of dicts with:
      {"index": int, "original_index": int, "similarity": float, "content": str}

    A retry is detected when a user message has high (>0.3) similarity to
    a previous user message in the same session.
    """
    retries: list[dict] = []
    user_messages = [(i, m) for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_messages) < 2:
        return retries

    for j in range(1, len(user_messages)):
        idx_j, msg_j = user_messages[j]
        content_j = msg_j.get("content", "")
        if not isinstance(content_j, str) or len(content_j.strip()) < 10:
            continue

        for k in range(j):
            idx_k, msg_k = user_messages[k]
            content_k = msg_k.get("content", "")
            if not isinstance(content_k, str) or len(content_k.strip()) < 10:
                continue

            sim = _text_similarity(content_j, content_k)
            if sim > 0.3:
                retries.append({
                    "index": idx_j,
                    "original_index": idx_k,
                    "similarity": round(sim, 3),
                    "content": content_j[:200],
                })
                break  # Only report the best match

    return retries
