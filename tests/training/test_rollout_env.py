"""Unit tests for rollout_env module."""
from __future__ import annotations

import json

from trainable_openclaw.training.rollout_env import (
    ToolExecutor,
    RuleSimulatedUser,
    TauBenchRolloutEnv,
    parse_tool_calls_from_text,
    UserResponse,
)


# ---------------------------------------------------------------------------
# Sample task for testing
# ---------------------------------------------------------------------------

SAMPLE_TASK = {
    "id": "retail_test_1",
    "source": "taubench_retail",
    "domain": "retail",
    "prompt": (
        "You are Alice Chen, and you live in California in zipcode 94102.\n\n"
        "You want to check your recent orders and cancel any that are still pending."
    ),
    "evaluation": {
        "nl_assertions": [
            "Agent should look up user by name and zip before taking action.",
            "Agent should cancel pending orders if user requests.",
        ],
    },
}

SAMPLE_TASK_WITH_ID = {
    "id": "retail_test_2",
    "source": "taubench_retail",
    "domain": "retail",
    "prompt": (
        "Your name is Bob Williams, your user id is U002, "
        "you live in New York, NY 10001.\n\n"
        "You want to return the Smart Fitness Watch from your last order."
    ),
    "evaluation": {
        "nl_assertions": [
            "Agent should look up order details before processing return.",
            "Agent should process the return for the correct order.",
        ],
    },
}


# ---------------------------------------------------------------------------
# ToolExecutor tests
# ---------------------------------------------------------------------------


class TestToolExecutor:
    def test_known_tool_success(self):
        executor = ToolExecutor("retail")
        result = executor.execute(
            "find_user_id_by_name_zip",
            {"first_name": "Alice", "last_name": "Chen", "zip": "94102"},
        )
        assert result["status"] == "success"
        users = result.get("result", result.get("users", []))
        assert len(users) == 1
        assert users[0]["user_id"] == "U001"

    def test_known_tool_error(self):
        executor = ToolExecutor("retail")
        result = executor.execute(
            "find_user_id_by_name_zip",
            {"first_name": "Alice"},
        )
        assert result["status"] == "error"

    def test_unknown_tool(self):
        executor = ToolExecutor("retail")
        result = executor.execute("nonexistent_tool", {})
        assert result["status"] == "error"

    def test_get_order_details(self):
        executor = ToolExecutor("retail")
        result = executor.execute("get_order_details", {"order_id": "O001"})
        assert result["status"] == "success"
        order = result.get("result", result)
        assert order["order_id"] == "O001"
        assert order["status"] == "delivered"

    def test_cancel_pending_order(self):
        executor = ToolExecutor("retail")
        result = executor.execute("cancel_pending_order", {"order_id": "O002"})
        assert result["status"] == "success"

    def test_cancel_delivered_order_fails(self):
        executor = ToolExecutor("retail")
        result = executor.execute("cancel_pending_order", {"order_id": "O001"})
        assert result["status"] == "error"

    def test_return_delivered_order(self):
        executor = ToolExecutor("retail")
        result = executor.execute(
            "return_delivered_order_items",
            {"order_id": "O001", "item_ids": ["I001"], "payment_method": "credit_card"},
        )
        assert result["status"] == "success"

    def test_reset(self):
        executor = ToolExecutor("retail")
        executor.execute("cancel_pending_order", {"order_id": "O002"})
        executor.reset()
        result = executor.execute("get_order_details", {"order_id": "O002"})
        assert result["status"] == "success"
        order = result.get("result", result)
        assert order["status"] == "pending"


# ---------------------------------------------------------------------------
# parse_tool_calls_from_text tests
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    def test_xml_wrapped(self):
        text = '<function_call>{"name": "test_tool", "arguments": {"key": "val"}}</function_call>'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "test_tool"
        assert calls[0]["arguments"] == {"key": "val"}

    def test_raw_json(self):
        text = '{"name": "get_order", "arguments": {"order_id": "O001"}}'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) >= 1
        assert calls[0]["name"] == "get_order"

    def test_openai_style(self):
        text = '{"function": {"name": "get_user", "arguments": "{\\"user_id\\": \\"U001\\"}"}}'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) >= 1
        assert calls[0]["name"] == "get_user"

    def test_no_tool_calls(self):
        text = "Hello, how can I help you today?"
        calls = parse_tool_calls_from_text(text)
        assert calls == []

    def test_empty(self):
        assert parse_tool_calls_from_text("") == []

    def test_multiple_calls(self):
        text = (
            '<function_call>{"name": "find_user", "arguments": {"name": "Alice"}}</function_call>'
            '<function_call>{"name": "get_order", "arguments": {"order_id": "O001"}}</function_call>'
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# RuleSimulatedUser tests
# ---------------------------------------------------------------------------


class TestRuleSimulatedUser:
    def test_initial_message_extracts_identity(self):
        user = RuleSimulatedUser(SAMPLE_TASK)
        msg = user.initial_message
        assert "Alice Chen" in msg
        assert "94102" in msg

    def test_initial_message_with_user_id(self):
        user = RuleSimulatedUser(SAMPLE_TASK_WITH_ID)
        msg = user.initial_message
        assert "Bob Williams" in msg
        assert "U002" in msg

    def test_respond_continue_on_first_round(self):
        user = RuleSimulatedUser(SAMPLE_TASK)
        response = user.respond(
            "Let me look up your account.",
            round_num=1,
            tool_results=[
                {"name": "find_user_id_by_name_zip", "result": {"status": "success"}}
            ],
        )
        assert response.status == "continue"
        assert 0.0 <= response.satisfaction <= 1.0

    def test_respond_complete_when_all_assertions_met(self):
        user = RuleSimulatedUser(SAMPLE_TASK)
        response = user.respond(
            "I've cancelled your pending orders. Is there anything else?",
            round_num=3,
            tool_results=[
                {"name": "find_user_id_by_name_zip", "result": {"status": "success"}},
                {"name": "cancel_pending_order", "result": {"status": "success"}},
            ],
        )
        assert response.status in ("continue", "complete")
        assert response.satisfaction > 0.0

    def test_respond_partial_progress(self):
        user = RuleSimulatedUser(SAMPLE_TASK)
        response = user.respond(
            "I found your account but need to check your orders.",
            round_num=1,
            tool_results=[
                {"name": "find_user_id_by_name_zip", "result": {"status": "success"}},
            ],
        )
        assert response.status == "continue"
        assert "still need" in response.message.lower() or "rest" in response.message.lower() or "okay" in response.message.lower()


# ---------------------------------------------------------------------------
# TauBenchRolloutEnv tests
# ---------------------------------------------------------------------------


class TestTauBenchRolloutEnv:
    def _mock_model_fn_simple(self, conversation):
        last_msg = conversation[-1]["content"] if conversation else ""
        if "find" in last_msg.lower() or "check" in last_msg.lower() or "order" in last_msg.lower():
            return (
                "I'll look up your information. "
                '<function_call>{"name": "find_user_id_by_name_zip", "arguments": {"first_name": "Alice", "last_name": "Chen", "zip": "94102"}}</function_call>'
            )
        return "Let me help you with that. Can you provide your user ID?"

    def _mock_model_fn_complete(self, conversation):
        calls_tool = any("find_user" in msg.get("content", "") for msg in conversation)
        if not calls_tool:
            return (
                '<function_call>{"name": "find_user_id_by_name_zip", "arguments": {"first_name": "Alice", "last_name": "Chen", "zip": "94102"}}</function_call>'
            )
        return "I've found your account Alice. Your order O002 is pending. Let me cancel that for you."

    def test_rollout_one_basic(self):
        env = TauBenchRolloutEnv([SAMPLE_TASK], max_turns=5)
        traj = env.rollout_one(SAMPLE_TASK, self._mock_model_fn_simple)
        assert traj.rounds >= 1
        assert len(traj.conversations) >= 2
        assert traj.status in ("complete", "give_up", "timeout", "in_progress")

    def test_rollout_batch(self):
        tasks = [SAMPLE_TASK, SAMPLE_TASK_WITH_ID]
        env = TauBenchRolloutEnv(tasks, max_turns=5)
        results = env.rollout_batch(tasks, self._mock_model_fn_simple)
        assert len(results) == 2
        for traj in results:
            assert traj.rounds >= 1
