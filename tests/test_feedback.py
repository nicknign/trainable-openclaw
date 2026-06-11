"""Tests for feedback module: deterministic verifier, signal extractor, reward combiner."""

import pytest

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


# ============================================================================
# Layer 1: Deterministic Verifier
# ============================================================================


class TestVerifyToolCallFormat:
    def test_valid_tool_call(self):
        tc = {
            "id": "call_abc123",
            "function": {
                "name": "read_file",
                "arguments": '{"file_path": "/tmp/test.txt"}',
            },
        }
        result = verify_tool_call_format(tc)
        assert result.passed is True
        assert result.score == 1.0
        assert all(c["passed"] for c in result.checks)

    def test_missing_id(self):
        tc = {
            "function": {
                "name": "read_file",
                "arguments": '{"file_path": "/tmp/test.txt"}',
            },
        }
        result = verify_tool_call_format(tc)
        assert result.passed is False
        assert result.score == 0.0

    def test_missing_function(self):
        tc = {"id": "call_abc123"}
        result = verify_tool_call_format(tc)
        assert result.passed is False
        assert result.score == 0.0
        assert any("function" in str(c).lower() and not c["passed"] for c in result.checks)

    def test_missing_name(self):
        tc = {
            "id": "call_abc123",
            "function": {
                "name": "",
                "arguments": '{"file_path": "/tmp/test.txt"}',
            },
        }
        result = verify_tool_call_format(tc)
        assert result.passed is False
        assert any(c["check"] == "name_valid" and not c["passed"] for c in result.checks)

    def test_invalid_json_arguments(self):
        tc = {
            "id": "call_abc123",
            "function": {
                "name": "read_file",
                "arguments": "not valid json {",
            },
        }
        result = verify_tool_call_format(tc)
        assert result.passed is False
        assert any(c["check"] == "arguments_parseable" and not c["passed"] for c in result.checks)

    def test_missing_arguments(self):
        tc = {
            "id": "call_abc123",
            "function": {
                "name": "read_file",
            },
        }
        result = verify_tool_call_format(tc)
        assert result.passed is False
        assert any(c["check"] == "arguments_present" and not c["passed"] for c in result.checks)

    def test_empty_string_id(self):
        tc = {
            "id": "",
            "function": {
                "name": "read_file",
                "arguments": '{"file_path": "/tmp/test.txt"}',
            },
        }
        result = verify_tool_call_format(tc)
        assert result.passed is False


class TestVerifyDangerousOperation:
    def test_blocks_rm_rf(self):
        tc = {
            "function": {
                "name": "exec",
                "arguments": '{"command": "rm -rf /important/data"}',
            },
        }
        result = verify_dangerous_operation(tc)
        assert result.passed is False
        assert result.score == 0.0

    def test_blocks_dd(self):
        tc = {
            "function": {
                "name": "shell",
                "arguments": '{"cmd": "dd if=/dev/zero of=/dev/sda"}',
            },
        }
        result = verify_dangerous_operation(tc)
        assert result.passed is False

    def test_blocks_system_path_write(self):
        tc = {
            "function": {
                "name": "write_file",
                "arguments": '{"file_path": "/etc/passwd", "content": "hacked"}',
            },
        }
        result = verify_dangerous_operation(tc)
        assert result.passed is False

    def test_allows_safe_exec(self):
        tc = {
            "function": {
                "name": "exec",
                "arguments": '{"command": "ls -la /home/user"}',
            },
        }
        result = verify_dangerous_operation(tc)
        assert result.passed is True
        assert result.score == 1.0

    def test_allows_safe_write(self):
        tc = {
            "function": {
                "name": "write_file",
                "arguments": '{"file_path": "/home/user/output.txt", "content": "safe content"}',
            },
        }
        result = verify_dangerous_operation(tc)
        assert result.passed is True

    def test_benign_tool_no_check(self):
        tc = {
            "function": {
                "name": "read_file",
                "arguments": '{"file_path": "/etc/hosts"}',
            },
        }
        result = verify_dangerous_operation(tc)
        # read_file is not checked for dangerous ops
        assert result.passed is True


class TestVerifyTaskCompletion:
    def test_completion_tool_detected(self):
        trajectory = [
            {"role": "user", "content": "do a task"},
            {"role": "assistant", "content": "Let me finish.",
             "tool_calls": [
                 {"id": "1", "function": {"name": "complete_goal", "arguments": '{"result": "done"}'}},
             ]},
        ]
        result = verify_task_completion(trajectory)
        assert result.passed is True

    def test_completion_language_detected(self):
        trajectory = [
            {"role": "user", "content": "what is 2+2"},
            {"role": "assistant", "content": "Here is the answer: 4. I have completed the calculation."},
        ]
        result = verify_task_completion(trajectory)
        assert result.passed is True

    def test_no_completion(self):
        trajectory = [
            {"role": "user", "content": "what is 2+2"},
            {"role": "assistant", "content": "Hmm"},
        ]
        result = verify_task_completion(trajectory)
        assert result.passed is False

    def test_no_assistant_messages(self):
        trajectory = [
            {"role": "user", "content": "hello"},
        ]
        result = verify_task_completion(trajectory)
        assert result.passed is False


class TestComputeLayer1Reward:
    def test_all_pass(self):
        trajectory = [
            {"role": "user", "content": "read /tmp/test.txt"},
            {"role": "assistant", "content": "Let me read that.",
             "tool_calls": [
                 {"id": "1", "function": {"name": "read_file", "arguments": '{"file_path": "/tmp/test.txt"}'}},
             ]},
            {"role": "tool", "content": '{"status": "success", "result": "file contents"}'},
            {"role": "assistant", "content": "Here is the file content. Task completed."},
        ]
        reward = compute_layer1_reward(trajectory)
        assert reward == 1.0

    def test_partial_fail(self):
        trajectory = [
            {"role": "user", "content": "delete system files"},
            {"role": "assistant", "content": "OK",
             "tool_calls": [
                 {"id": "1", "function": {"name": "exec", "arguments": '{"command": "rm -rf /"}'}},
             ]},
            {"role": "assistant", "content": "Hmm"},
        ]
        reward = compute_layer1_reward(trajectory)
        # format check passes, dangerous op fails, completion fails → 1/3
        assert 0.0 < reward < 1.0

    def test_no_tool_calls(self):
        trajectory = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hi! Here is a summary of what I found."},
        ]
        reward = compute_layer1_reward(trajectory)
        # Only task_completion is checked; the completion language should pass
        assert reward == 1.0

    def test_empty_trajectory(self):
        reward = compute_layer1_reward([])
        assert reward == 0.0


# ============================================================================
# Layer 2: Signal Extractor
# ============================================================================


class TestExtractPositiveSignal:
    def test_english_thank_you(self):
        msgs = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "thank you, that's perfect!"},
        ]
        signals = extract_signals_from_messages(msgs)
        positive = [s for s in signals if s.signal_type == "positive"]
        assert len(positive) == 1
        assert positive[0].confidence >= 0.7

    def test_chinese_positive(self):
        msgs = [
            {"role": "user", "content": "帮我排序"},
            {"role": "assistant", "content": "排好了"},
            {"role": "user", "content": "完美！搞定了谢谢"},
        ]
        signals = extract_signals_from_messages(msgs)
        positive = [s for s in signals if s.signal_type == "positive"]
        assert len(positive) == 1
        assert positive[0].confidence >= 0.7

    def test_strong_satisfaction(self):
        msgs = [
            {"role": "user", "content": "that's exactly what I needed, awesome work!"},
        ]
        signals = extract_signals_from_messages(msgs)
        positive = [s for s in signals if s.signal_type == "positive"]
        assert len(positive) == 1
        # "that's exactly" is a strong pattern → confidence 0.9
        assert positive[0].confidence >= 0.8


class TestExtractNegativeSignal:
    def test_english_negative(self):
        msgs = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "here is result"},
            {"role": "user", "content": "no, that's wrong, try again"},
        ]
        signals = extract_signals_from_messages(msgs)
        negative = [s for s in signals if s.signal_type == "negative"]
        assert len(negative) == 1
        assert negative[0].confidence >= 0.6

    def test_chinese_negative(self):
        msgs = [
            {"role": "user", "content": "不对，重新做"},
        ]
        signals = extract_signals_from_messages(msgs)
        negative = [s for s in signals if s.signal_type == "negative"]
        assert len(negative) == 1
        assert negative[0].confidence >= 0.7


class TestExtractCorrectionSignal:
    def test_english_correction(self):
        msgs = [
            {"role": "user", "content": "it should be sorted(list) instead of list.sort()"},
        ]
        signals = extract_signals_from_messages(msgs)
        correction = [s for s in signals if s.signal_type == "correction"]
        assert len(correction) == 1
        assert correction[0].confidence >= 0.7

    def test_chinese_correction(self):
        msgs = [
            {"role": "user", "content": "应该是用快速排序，改成 quicksort 吧"},
        ]
        signals = extract_signals_from_messages(msgs)
        correction = [s for s in signals if s.signal_type == "correction"]
        assert len(correction) == 1

    def test_correction_overrides_negative(self):
        """When both correction and negative patterns match, correction wins."""
        msgs = [
            {"role": "user", "content": "不对，应该是 sorted(arr, reverse=True)"},
        ]
        signals = extract_signals_from_messages(msgs)
        # Should have correction, not negative
        types = [s.signal_type for s in signals]
        assert "correction" in types
        # "不对" would match negative, but correction patterns take priority
        correction = [s for s in signals if s.signal_type == "correction"]
        assert len(correction) == 1


class TestExtractMixedSignals:
    def test_multiple_signals(self):
        msgs = [
            {"role": "user", "content": "write a sorting function"},
            {"role": "assistant", "content": "here: def sort(arr): return arr"},
            {"role": "user", "content": "no that's wrong"},
            {"role": "assistant", "content": "ok here: def sort(arr): return sorted(arr)"},
            {"role": "user", "content": "perfect, thank you!"},
        ]
        signals = extract_signals_from_messages(msgs)
        types = [s.signal_type for s in signals]
        # Should have negative + positive
        assert "negative" in types
        assert "positive" in types

    def test_neutral_when_no_signals(self):
        msgs = [
            {"role": "user", "content": "what is 2+2"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "ok"},
        ]
        signals = extract_signals_from_messages(msgs)
        assert len(signals) > 0
        # "ok" is too short to match any pattern (except abandonment check for assistant last)
        # Actually "ok" might not match anything. But abandonment check fires if last is assistant.
        # Let's verify: last message is user "ok", so no abandonment
        types = [s.signal_type for s in signals]
        assert "neutral" in types or len(signals) == 0

    def test_abandoned_session(self):
        msgs = [
            {"role": "user", "content": "do a complex task"},
            {"role": "assistant", "content": "Here is the detailed result..."},
        ]
        signals = extract_signals_from_messages(msgs)
        types = [s.signal_type for s in signals]
        assert "abandoned" in types


class TestComputeLayer2Reward:
    def test_positive_signal_gives_high_reward(self):
        signals = [
            FeedbackSignal(signal_type="positive", confidence=0.9, evidence="that's perfect!", details="test"),
        ]
        reward = compute_layer2_reward(signals)
        assert reward >= 0.8

    def test_negative_signal_floor_at_zero(self):
        signals = [
            FeedbackSignal(signal_type="negative", confidence=0.9, evidence="no wrong", details="test"),
            FeedbackSignal(signal_type="negative", confidence=0.9, evidence="still wrong", details="test"),
            FeedbackSignal(signal_type="negative", confidence=0.9, evidence="wrong again", details="test"),
        ]
        reward = compute_layer2_reward(signals)
        # Each negative is -0.5 * confidence, averaged. Floor at 0.
        assert reward == 0.0

    def test_correction_signal_partial_credit(self):
        signals = [
            FeedbackSignal(signal_type="correction", confidence=0.8, evidence="应该是X", details="test"),
        ]
        reward = compute_layer2_reward(signals)
        # 0.3 * 0.8 = 0.24
        assert 0.2 < reward < 0.3

    def test_neutral_default(self):
        signals = [
            FeedbackSignal(signal_type="neutral", confidence=0.5, evidence="", details="test"),
        ]
        reward = compute_layer2_reward(signals)
        assert reward == 0.5

    def test_empty_signals(self):
        reward = compute_layer2_reward([])
        assert reward == 0.5

    def test_mixed_signals_averaged(self):
        signals = [
            FeedbackSignal(signal_type="positive", confidence=0.9, evidence="great!", details=""),
            FeedbackSignal(signal_type="negative", confidence=0.7, evidence="wrong", details=""),
        ]
        reward = compute_layer2_reward(signals)
        # positive: 1.0 * 0.9 = 0.9, negative: -0.5 * 0.7 = -0.35, avg = 0.275
        assert 0.2 < reward < 0.35


class TestDetectRetryPattern:
    def test_detect_retry(self):
        msgs = [
            {"role": "user", "content": "write a function to sort a list of integers"},
            {"role": "assistant", "content": "here is bubble sort"},
            {"role": "user", "content": "can you write a function to sort a list of integers in Python?"},
        ]
        retries = detect_retry_pattern(msgs)
        assert len(retries) == 1
        assert retries[0]["similarity"] > 0.3

    def test_no_retry(self):
        msgs = [
            {"role": "user", "content": "what is 2+2"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "what is the capital of France"},
        ]
        retries = detect_retry_pattern(msgs)
        assert len(retries) == 0

    def test_single_user_message(self):
        msgs = [
            {"role": "user", "content": "hello"},
        ]
        retries = detect_retry_pattern(msgs)
        assert len(retries) == 0

    def test_short_messages_skipped(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "hi"},
        ]
        retries = detect_retry_pattern(msgs)
        # Too short (< 10 chars) to compare
        assert len(retries) == 0


# ============================================================================
# Layer 3: Reward Combiner
# ============================================================================


class TestCombineWithoutJudge:
    def test_weights_redistributed(self):
        result = combine(layer1=1.0, layer2=0.5, layer3=None)
        # Default weights: {0.5, 0.3, 0.2}
        # Without L3, 0.2 redistributes: L1 gets 0.5+0.2*(0.5/0.8)=0.625, L2 gets 0.3+0.2*(0.3/0.8)=0.375
        # final = 0.625*1.0 + 0.375*0.5 = 0.625 + 0.1875 = 0.8125
        assert result.layer3 is None
        assert result.final == pytest.approx(0.8125, rel=1e-4)
        assert result.metadata["layer3_was_none"] is True
        assert result.weights["layer3"] == 0.0

    def test_weights_sum_to_one(self):
        result = combine(layer1=1.0, layer2=1.0, layer3=None)
        # L1=0.625, L2=0.375, final = 0.625+0.375 = 1.0
        w_sum = result.weights["layer1"] + result.weights["layer2"] + result.weights.get("layer3", 0)
        assert w_sum == pytest.approx(1.0, rel=1e-4)

    def test_both_zero(self):
        result = combine(layer1=0.0, layer2=0.0, layer3=None)
        assert result.final == 0.0

    def test_custom_weights_without_l3(self):
        result = combine(layer1=0.8, layer2=0.4, layer3=None, weights={"layer1": 0.4, "layer2": 0.4, "layer3": 0.2})
        # L3=0.2 redistributes: L1=0.4+0.2*(0.4/0.8)=0.5, L2=0.4+0.2*(0.4/0.8)=0.5
        # final = 0.5*0.8 + 0.5*0.4 = 0.4 + 0.2 = 0.6
        assert result.final == pytest.approx(0.6, rel=1e-4)


class TestCombineWithJudge:
    def test_all_three_layers(self):
        result = combine(layer1=1.0, layer2=0.5, layer3=0.8)
        # Default: 0.5*1.0 + 0.3*0.5 + 0.2*0.8 = 0.5 + 0.15 + 0.16 = 0.81
        assert result.layer3 == 0.8
        assert result.final == pytest.approx(0.81, rel=1e-4)
        assert result.metadata["layer3_was_none"] is False

    def test_custom_weights(self):
        result = combine(
            layer1=1.0,
            layer2=0.5,
            layer3=0.8,
            weights={"layer1": 0.3, "layer2": 0.3, "layer3": 0.4},
        )
        # 0.3*1.0 + 0.3*0.5 + 0.4*0.8 = 0.3 + 0.15 + 0.32 = 0.77
        assert result.final == pytest.approx(0.77, rel=1e-4)

    def test_layer3_dominates_with_high_weight(self):
        result = combine(
            layer1=0.2,
            layer2=0.2,
            layer3=0.9,
            weights={"layer1": 0.2, "layer2": 0.2, "layer3": 0.6},
        )
        # 0.2*0.2 + 0.2*0.2 + 0.6*0.9 = 0.04 + 0.04 + 0.54 = 0.62
        assert result.final == pytest.approx(0.62, rel=1e-4)


class TestComputeFullReward:
    def test_without_judge(self):
        trajectory = [
            {"role": "user", "content": "read /tmp/test.txt"},
            {"role": "assistant", "content": "Let me read that.",
             "tool_calls": [
                 {"id": "1", "function": {"name": "read_file", "arguments": '{"file_path": "/tmp/test.txt"}'}},
             ]},
            {"role": "tool", "content": '{"status": "success"}'},
            {"role": "assistant", "content": "Here is the content. Task completed."},
            {"role": "user", "content": "thanks, that's perfect!"},
        ]
        result = compute_full_reward(trajectory, call_judge=False)
        assert result.layer3 is None
        assert 0.0 <= result.final <= 1.0
        # L1 should be 1.0 (all checks pass), L2 should be > 0.5 (positive signal)
        assert result.layer1 == 1.0
        assert result.layer2 > 0.5

    def test_empty_trajectory(self):
        result = compute_full_reward([], call_judge=False)
        assert result.layer1 == 0.0
        assert result.layer3 is None
        assert 0.0 <= result.final <= 1.0

    def test_error_handling(self):
        """compute_full_reward should handle trajectories with mixed content types gracefully."""
        trajectory = [
            {"role": "user", "content": "what is 2+2"},
            {"role": "assistant", "content": None},  # edge case: None content
        ]
        result = compute_full_reward(trajectory, call_judge=False)
        # Should not crash; task completion may fail but should return valid result
        assert 0.0 <= result.final <= 1.0

    def test_with_judge_requires_executor(self):
        """Should raise if call_judge=True but no executor provided."""
        trajectory = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
        with pytest.raises(ValueError, match="judge_executor"):
            compute_full_reward(trajectory, call_judge=True)

    def test_with_judge_no_rubrics(self):
        """Should not crash when judge has no rubrics."""
        from trainable_openclaw.evaluation.judge import JudgeExecutor
        # Create a judge with no API key (won't actually call API)
        judge = JudgeExecutor(api_key="fake", use_merged=False)

        trajectory = [
            {"role": "user", "content": "test prompt"},
            {"role": "assistant", "content": "test response with enough text for completion"},
        ]
        # This will try to call API but fail because fake key → falls back to 0.5
        result = compute_full_reward(trajectory, call_judge=True, judge_executor=judge)
        assert result.layer3 is not None
        assert 0.0 <= result.final <= 1.0


# ============================================================================
# VerificationResult dataclass
# ============================================================================


class TestVerificationResult:
    def test_defaults(self):
        vr = VerificationResult(passed=True, score=1.0)
        assert vr.passed is True
        assert vr.score == 1.0
        assert vr.checks == []
        assert vr.details == ""

    def test_with_checks(self):
        vr = VerificationResult(
            passed=False,
            score=0.0,
            checks=[{"check": "format", "passed": False}],
            details="Bad format",
        )
        assert len(vr.checks) == 1
        assert "Bad format" in vr.details


# ============================================================================
# FeedbackSignal dataclass
# ============================================================================


class TestFeedbackSignal:
    def test_construction(self):
        sig = FeedbackSignal(
            signal_type="positive",
            confidence=0.9,
            evidence="thank you!",
            details="User was happy",
        )
        assert sig.signal_type == "positive"
        assert sig.confidence == 0.9
        assert "thank you" in sig.evidence


# ============================================================================
# CombinedReward dataclass
# ============================================================================


class TestCombinedReward:
    def test_construction(self):
        cr = CombinedReward(
            layer1=1.0,
            layer2=0.5,
            layer3=0.8,
            final=0.81,
            weights={"layer1": 0.5, "layer2": 0.3, "layer3": 0.2},
        )
        assert cr.layer1 == 1.0
        assert cr.final == 0.81


# ============================================================================
# Integration: compute_layer2_reward edge cases
# ============================================================================


class TestLayer2RewardEdgeCases:
    def test_abandoned_zero_reward(self):
        signals = [
            FeedbackSignal(signal_type="abandoned", confidence=0.8, evidence="stop", details=""),
        ]
        reward = compute_layer2_reward(signals)
        assert reward == 0.0

    def test_all_positive_capped(self):
        signals = [
            FeedbackSignal(signal_type="positive", confidence=1.0, evidence="perfect!", details=""),
            FeedbackSignal(signal_type="positive", confidence=1.0, evidence="great!", details=""),
            FeedbackSignal(signal_type="positive", confidence=1.0, evidence="awesome!", details=""),
        ]
        reward = compute_layer2_reward(signals)
        # Three 1.0s average = 1.0, capped at 1.0
        assert reward == 1.0

    def test_unknown_signal_type(self):
        # Graceful handling of unknown types
        sig = FeedbackSignal(signal_type="unknown", confidence=0.5, evidence="?", details="")
        reward = compute_layer2_reward([sig])
        assert 0.0 <= reward <= 1.0
