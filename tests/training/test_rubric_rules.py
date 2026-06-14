"""Unit tests for rubric_rules module."""
from __future__ import annotations

import json

from trainable_openclaw.training.rubric_rules import RubricRuleEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call(name, args, status="success"):
    return {"name": name, "arguments": args, "result": {"status": status}}


def _build_tool_result_list(calls):
    return [{"name": c["name"], "result": c.get("result", {"status": "success"})} for c in calls]


def _fc(name, arguments):
    """Build a function_call XML string."""
    return f'<function_call>{{"name": "{name}", "arguments": {json.dumps(arguments)}}}</function_call>'


def _make_conversation(agent_msgs, tool_calls=None, user_msgs=None):
    conversation = []
    if user_msgs:
        conversation.append({"role": "user", "content": user_msgs[0]})

    if tool_calls:
        tool_text = json.dumps(tool_calls)
        conversation.append({"role": "assistant", "content": agent_msgs[0] if agent_msgs else ""})
        conversation.append({"role": "tool", "content": tool_text})
    else:
        conversation.append({"role": "assistant", "content": agent_msgs[0] if agent_msgs else ""})

    return conversation


# ---------------------------------------------------------------------------
# Test basic scoring
# ---------------------------------------------------------------------------


class TestRubricRuleEngine:
    def test_score_empty_conversation(self):
        engine = RubricRuleEngine()
        result = engine.score({"conversations": []})
        assert "group_scores" in result
        assert "weighted_reward" in result
        assert 0.0 <= result["weighted_reward"] <= 1.0

    def test_score_no_tool_calls(self):
        engine = RubricRuleEngine()
        conversation = [
            {"role": "user", "content": "I need help with my order."},
            {"role": "assistant", "content": "Sure, what's your order ID?"},
        ]
        result = engine.score({"conversations": conversation, "status": "in_progress"})
        assert result["weighted_reward"] < 0.8

    def test_compute_reward_returns_float(self):
        engine = RubricRuleEngine()
        reward = engine.compute_reward({"conversations": []})
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0


# ---------------------------------------------------------------------------
# Test good vs bad trajectory scoring
# ---------------------------------------------------------------------------


class TestGoodVsBadScoring:
    """Verify that good trajectories score higher than bad ones."""

    def test_good_trajectory_scores_higher(self):
        engine = RubricRuleEngine()

        # Good trajectory: lookup user -> get order -> cancel -> confirm
        good_conv = [
            {"role": "user", "content": "Hi, I'm Alice Chen, zip 94102. I want to cancel my pending order."},
            {"role": "assistant", "content": _fc("find_user_id_by_name_zip", {"first_name": "Alice", "last_name": "Chen", "zip": "94102"})
                                      + "\nLet me look up your account first."},
            {"role": "tool", "content": json.dumps(_build_tool_result_list([
                _make_tool_call("find_user_id_by_name_zip", {"first_name": "Alice", "last_name": "Chen", "zip": "94102"}),
            ]))},
            {"role": "user", "content": "Okay, and what about my order?"},
            {"role": "assistant", "content": _fc("get_order_details", {"order_id": "O002"})
                                      + "\nI found your account (U001). Let me check your orders."},
            {"role": "tool", "content": json.dumps(_build_tool_result_list([
                _make_tool_call("get_order_details", {"order_id": "O002"}),
            ]))},
            {"role": "user", "content": "Yes, please cancel it."},
            {"role": "assistant", "content": _fc("cancel_pending_order", {"order_id": "O002"})
                                      + "\nI'll cancel order O002 for the running shoes. Let me confirm — would you like me to proceed?"},
            {"role": "tool", "content": json.dumps(_build_tool_result_list([
                _make_tool_call("cancel_pending_order", {"order_id": "O002"}),
            ]))},
            {"role": "user", "content": "Thank you! That's perfect."},
            {"role": "assistant", "content": "Your order O002 has been cancelled. The refund of $131.99 will go back to your PayPal. Is there anything else I can help with?"},
        ]
        good_score = engine.compute_reward({"conversations": good_conv, "status": "complete"})

        # Bad trajectory: skips lookup, cancels wrong order, no confirmation
        bad_conv = [
            {"role": "user", "content": "Hi, I'm Alice Chen. I want to cancel my pending order."},
            {"role": "assistant", "content": _fc("cancel_pending_order", {"order_id": "O999"})
                                      + "\nI'll cancel your order."},
            {"role": "tool", "content": json.dumps(_build_tool_result_list([
                _make_tool_call("cancel_pending_order", {"order_id": "O999"}, status="error"),
            ]))},
            {"role": "user", "content": "That's not my order! I didn't give you an order number."},
            {"role": "assistant", "content": "Sorry, I can't help you."},
        ]
        bad_score = engine.compute_reward({"conversations": bad_conv, "status": "give_up"})

        assert good_score > bad_score, f"Good {good_score:.3f} should > bad {bad_score:.3f}"

    def test_perfect_trajectory_near_max(self):
        engine = RubricRuleEngine()

        perfect_conv = [
            {"role": "user", "content": "Hi, I'm Alice Chen, zip 94102. I want to cancel my pending order O002."},
            {"role": "assistant", "content": _fc("think", {"reasoning": "Need to verify user identity first"})},
            {"role": "tool", "content": '{"status":"success"}'},
            {"role": "assistant", "content": _fc("find_user_id_by_name_zip", {"first_name": "Alice", "last_name": "Chen", "zip": "94102"})
                                      + "\nLet me verify your identity first."},
            {"role": "tool", "content": json.dumps(_build_tool_result_list([
                _make_tool_call("find_user_id_by_name_zip", {"first_name": "Alice", "last_name": "Chen", "zip": "94102"}),
            ]))},
            {"role": "assistant", "content": _fc("think", {"reasoning": "User verified, now check the order"})},
            {"role": "tool", "content": '{"status":"success"}'},
            {"role": "assistant", "content": _fc("get_order_details", {"order_id": "O002"})
                                      + "\nConfirmed — you're Alice Chen, user U001. Let me check order O002."},
            {"role": "tool", "content": json.dumps(_build_tool_result_list([
                _make_tool_call("get_order_details", {"order_id": "O002"}),
            ]))},
            {"role": "user", "content": "Yes, that's the one."},
            {"role": "assistant", "content": _fc("think", {"reasoning": "Should confirm before cancelling"})},
            {"role": "tool", "content": '{"status":"success"}'},
            {"role": "assistant", "content": "Order O002 for Running Shoes ($119.99) is pending. Would you like me to cancel it?"},
            {"role": "user", "content": "Yes, please."},
            {"role": "assistant", "content": _fc("cancel_pending_order", {"order_id": "O002"})
                                      + "\nCancelling order O002 now."},
            {"role": "tool", "content": json.dumps(_build_tool_result_list([
                _make_tool_call("cancel_pending_order", {"order_id": "O002"}),
            ]))},
            {"role": "assistant", "content": _fc("get_order_details", {"order_id": "O002"})
                                      + "\nLet me verify the cancellation."},
            {"role": "tool", "content": json.dumps(_build_tool_result_list([
                _make_tool_call("get_order_details", {"order_id": "O002"}),
            ]))},
            {"role": "assistant", "content": "I've cancelled order O002. The refund of $131.99 will go back to your PayPal account. Is there anything else I can help with?"},
            {"role": "user", "content": "Thank you so much! That was perfect."},
        ]
        score = engine.compute_reward({"conversations": perfect_conv, "status": "complete"})
        assert score >= 0.7, f"Perfect trajectory score {score:.3f} should be >= 0.7"

    def test_empty_trajectory_low_score(self):
        engine = RubricRuleEngine()
        score = engine.compute_reward({"conversations": [], "status": "in_progress"})
        assert score < 0.8, f"Empty trajectory score {score:.3f} should be low"


# ---------------------------------------------------------------------------
# Test group scoring
# ---------------------------------------------------------------------------


class TestGroupScoring:
    def test_group_scores_sum_correctly(self):
        engine = RubricRuleEngine()
        conv = [
            {"role": "user", "content": "I need help."},
            {"role": "assistant", "content": "I'm here to help."},
        ]
        result = engine.score({"conversations": conv, "status": "in_progress"})
        groups = result["group_scores"]
        assert "tool_selection" in groups
        assert "info_sufficiency" in groups
        assert "step_efficiency" in groups
        assert "error_recovery" in groups
        assert "task_completion" in groups
        assert "communication" in groups
        for g in groups:
            assert 0.0 <= groups[g] <= 10.0, f"{g} score {groups[g]} out of range"

    def test_deductions_recorded(self):
        engine = RubricRuleEngine()
        conv = [
            {"role": "user", "content": "Cancel my order."},
            {"role": "assistant", "content": _fc("cancel_pending_order", {"order_id": "O999"})
                                      + "\nI've cancelled your order."},
            {"role": "tool", "content": json.dumps(_build_tool_result_list([
                _make_tool_call("cancel_pending_order", {"order_id": "O999"}, status="error"),
            ]))},
        ]
        result = engine.score({"conversations": conv, "status": "give_up"})
        assert len(result["deductions"]) > 0, "Should have deductions for bad trajectory"
