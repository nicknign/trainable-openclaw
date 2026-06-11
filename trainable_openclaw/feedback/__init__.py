"""
Feedback collection module for trainable-openclaw.

Three-layer reward architecture:
  Layer 1: Deterministic verification (tool calls, exit codes, file ops) — FREE
  Layer 2: User feedback signals extracted from conversation — FREE
  Layer 3: LLM Judge for communication quality — LOW-FREQUENCY

Combined: final = w1*L1 + w2*L2 + w3*L3 (defaults: 0.5, 0.3, 0.2)
"""

from trainable_openclaw.feedback.deterministic_verifier import (
    VerificationResult,
    verify_tool_call_format,
    verify_execution_result,
    verify_dangerous_operation,
    verify_task_completion,
    compute_layer1_reward,
)

from trainable_openclaw.feedback.signal_extractor import (
    FeedbackSignal,
    extract_signals_from_messages,
    compute_layer2_reward,
    detect_retry_pattern,
)

from trainable_openclaw.feedback.reward_combiner import (
    CombinedReward,
    combine,
    compute_full_reward,
)
