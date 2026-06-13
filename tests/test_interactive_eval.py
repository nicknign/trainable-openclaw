"""Tests for interactive evaluation system (SimulatedUser + InteractiveEvaluator)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from trainable_openclaw.evaluation.simulated_user import SimulatedUser, UserResponse
from trainable_openclaw.evaluation.interactive_eval import (
    AgentRunner,
    InteractiveEvaluator,
    TaskResult,
    EvalReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_task():
    return {
        "id": "retail_task_1",
        "domain": "retail",
        "prompt": (
            "You are Alice Chen in zip code 94102.\n"
            "Your user id is U001.\n"
            "You want to exchange the keyboard in order #W1 for a clicky switch version."
        ),
        "evaluation": {
            "purpose": "Test exchange with valid alternative product.",
            "nl_assertions": ["Agent should process the exchange correctly."],
            "reward_basis": ["DB", "NL_ASSERTION"],
        },
        "tools": [],
    }


@pytest.fixture
def mock_openai_response():
    """Factory for creating mock OpenAI chat completion responses."""

    def _make(content: str):
        mock = MagicMock()
        mock.choices = [MagicMock()]
        mock.choices[0].message = MagicMock()
        mock.choices[0].message.content = content
        mock.choices[0].message.tool_calls = None
        return mock

    return _make


# ---------------------------------------------------------------------------
# SimulatedUser
# ---------------------------------------------------------------------------

class TestSimulatedUser:
    def test_initial_message_parses_want_line(self, sample_task):
        su = SimulatedUser(sample_task)
        msg = su.initial_message
        assert "exchange" in msg.lower()
        assert "keyboard" in msg.lower()

    def test_initial_message_fallback_no_want(self):
        task = {"id": "t1", "prompt": "You are Bob.\nYour user id is B1.\nHelp me please."}
        su = SimulatedUser(task)
        assert "Help me please" in su.initial_message

    def test_respond_parses_complete(self, sample_task, mock_openai_response):
        su = SimulatedUser(sample_task)
        with patch.object(su, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                mock_openai_response(
                    '{"message": "Thanks, that\'s perfect!", "status": "complete", "satisfaction": 1.0}'
                )
            )
            result = su.respond("I've exchanged your keyboard.", [{"name": "exchange", "content": "success"}])

        assert result.status == "complete"
        assert result.satisfaction == 1.0
        assert su.round_count == 1
        assert len(su.history) == 2  # agent + user

    def test_respond_parses_continue(self, sample_task, mock_openai_response):
        su = SimulatedUser(sample_task)
        with patch.object(su, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                mock_openai_response(
                    '{"message": "That\'s not the right keyboard. I want clicky switches.", "status": "continue", "satisfaction": 0.3}'
                )
            )
            result = su.respond("I've processed your order.", [{"name": "get_order", "content": "order #W1"}])

        assert result.status == "continue"
        assert result.satisfaction == 0.3

    def test_respond_parses_give_up(self, sample_task, mock_openai_response):
        su = SimulatedUser(sample_task)
        with patch.object(su, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                mock_openai_response(
                    '{"message": "This is going nowhere. I\'ll call instead.", "status": "give_up", "satisfaction": 0.0}'
                )
            )
            result = su.respond("I don't understand what you want.", [])

        assert result.status == "give_up"

    def test_parse_json_strips_markdown_fences(self, sample_task, mock_openai_response):
        su = SimulatedUser(sample_task)
        with patch.object(su, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                mock_openai_response(
                    '```json\n{"message": "ok!", "status": "complete", "satisfaction": 1.0}\n```'
                )
            )
            result = su.respond("Done!", [])
        assert result.status == "complete"
        assert result.message == "ok!"

    def test_parse_json_fallback_on_invalid(self, sample_task, mock_openai_response):
        su = SimulatedUser(sample_task)
        with patch.object(su, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                mock_openai_response("Just a plain text response without JSON")
            )
            result = su.respond("Done!", [])
        # Should fallback gracefully
        assert result.status == "continue"
        assert len(result.message) > 0

    def test_history_accumulates(self, sample_task, mock_openai_response):
        su = SimulatedUser(sample_task)
        with patch.object(su, "_get_client") as mock_client:
            mock = mock_openai_response(
                '{"message": "Not yet, keep trying.", "status": "continue", "satisfaction": 0.4}'
            )
            mock_client.return_value.chat.completions.create.return_value = mock

            su.respond("First try", [])
            su.respond("Second try", [])

        assert su.round_count == 2
        assert len(su.history) == 4  # 2 agent + 2 user

    def test_format_success_criteria_dict(self, sample_task):
        su = SimulatedUser(sample_task)
        criteria = su._format_success_criteria()
        assert "Test exchange" in criteria
        assert "Agent should process" in criteria

    def test_format_success_criteria_string(self):
        task = {"id": "t1", "evaluation": "The user wants a refund."}
        su = SimulatedUser(task)
        criteria = su._format_success_criteria()
        assert "refund" in criteria


# ---------------------------------------------------------------------------
# InteractiveEvaluator + AgentRunner
# ---------------------------------------------------------------------------

class TestAgentRunner:
    def _make_mock_executor(self, responses=None):
        """Return a tool executor that returns canned responses."""
        if responses is None:
            responses = {}
        return lambda name, args: responses.get(name, {"status": "ok"})

    def test_run_without_tool_calls(self):
        """Agent responds with text only, no tool calls."""
        executor = self._make_mock_executor()
        runner = AgentRunner(tools=[], tool_executor=executor, model="test")

        with patch.object(runner, "_get_client") as mock_client:
            mock_msg = MagicMock()
            mock_msg.content = "How can I help you today?"
            mock_msg.tool_calls = None
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message = mock_msg
            mock_client.return_value.chat.completions.create.return_value = mock_resp

            text, tool_results = runner.run("I need help", [])

        assert "help" in text.lower()
        assert tool_results == []

    def test_run_with_tool_calls(self):
        """Agent makes tool calls then returns text."""
        executor = self._make_mock_executor({"get_order": {"order_id": "W1", "status": "delivered"}})
        tools = [{"type": "function", "function": {"name": "get_order", "description": "Get order details", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}]
        runner = AgentRunner(tools=tools, tool_executor=executor, model="test")

        with patch.object(runner, "_get_client") as mock_client:
            # First response: tool call
            tool_msg = MagicMock()
            tool_msg.content = None
            tool_msg.tool_calls = [MagicMock()]
            tool_msg.tool_calls[0].id = "call_1"
            tool_msg.tool_calls[0].function.name = "get_order"
            tool_msg.tool_calls[0].function.arguments = '{"order_id": "W1"}'
            tool_msg.tool_calls[0].model_dump.return_value = {"id": "call_1", "function": {"name": "get_order", "arguments": '{"order_id": "W1"}'}, "type": "function"}

            # Second response: text
            text_msg = MagicMock()
            text_msg.content = "Your order #W1 was delivered on May 10."
            text_msg.tool_calls = None

            resp1 = MagicMock()
            resp1.choices = [MagicMock()]
            resp1.choices[0].message = tool_msg
            resp2 = MagicMock()
            resp2.choices = [MagicMock()]
            resp2.choices[0].message = text_msg
            mock_client.return_value.chat.completions.create.side_effect = [resp1, resp2]

            text, tool_results = runner.run("Where is my order #W1?", [])

        assert "W1" in text
        assert len(tool_results) == 1
        assert tool_results[0]["name"] == "get_order"


class TestInteractiveEvaluator:
    def _make_mock_agent(self, responses=None):
        """Return an AgentRunner that returns canned (text, tool_results) pairs."""
        if responses is None:
            responses = [("Here you go!", [])]

        class MockAgent:
            def __init__(self):
                self.call_count = 0
                self.responses = responses

            def run(self, user_message, history):
                idx = min(self.call_count, len(self.responses) - 1)
                self.call_count += 1
                return self.responses[idx]

        return MockAgent()

    def _make_task(self, tid="t1", domain="retail"):
        return {
            "id": tid,
            "domain": domain,
            "prompt": f"Task {tid}",
            "evaluation": {"purpose": f"Test {tid}"},
            "tools": [],
        }

    def test_single_task_first_try_success(self):
        """Simulated user gives complete on first response."""
        agent = self._make_mock_agent([("I've processed your exchange. Done!", [])])

        evaluator = InteractiveEvaluator(agent=agent, max_rounds=5)
        task = self._make_task()

        with patch("trainable_openclaw.evaluation.interactive_eval.SimulatedUser") as MockSU:
            mock_su = MagicMock()
            mock_su.initial_message = "Hi, I want to exchange an item."
            mock_su.history = []
            mock_su.respond.return_value = UserResponse(
                message="Thanks, that's exactly right!",
                status="complete",
                satisfaction=1.0,
            )
            MockSU.return_value = mock_su

            report = evaluator.evaluate([task])

        assert report.total_tasks == 1
        assert report.completed_tasks == 1
        assert report.avg_rounds == 1.0
        assert report.first_try_rate == 1.0
        assert report.abandonment_rate == 0.0

    def test_task_needs_correction_then_completes(self):
        """Simulated user corrects once, then completes."""
        agent = self._make_mock_agent([
            ("I've cancelled your order.", []),
            ("OK, I've cancelled order #P123 specifically.", []),
        ])

        evaluator = InteractiveEvaluator(agent=agent, max_rounds=5)

        with patch("trainable_openclaw.evaluation.interactive_eval.SimulatedUser") as MockSU:
            mock_su = MagicMock()
            mock_su.initial_message = "Hi, I want to cancel order #P123."
            mock_su.history = []
            mock_su.respond.side_effect = [
                UserResponse(message="No, I said #P123, not any order.", status="continue", satisfaction=0.3),
                UserResponse(message="Yes, thank you!", status="complete", satisfaction=1.0),
            ]
            MockSU.return_value = mock_su

            report = evaluator.evaluate([self._make_task()])

        assert report.completed_tasks == 1
        assert report.avg_rounds == 2.0
        assert report.first_try_rate == 0.0
        assert report.recovery_rate == 1.0  # recovered after correction

    def test_task_user_gives_up(self):
        """Simulated user gives up after repeated failures."""
        agent = self._make_mock_agent([("I don't know.", [])])

        evaluator = InteractiveEvaluator(agent=agent, max_rounds=5)

        with patch("trainable_openclaw.evaluation.interactive_eval.SimulatedUser") as MockSU:
            mock_su = MagicMock()
            mock_su.initial_message = "Hi!"
            mock_su.history = []
            mock_su.respond.return_value = UserResponse(
                message="Never mind, I'll go elsewhere.",
                status="give_up",
                satisfaction=0.0,
            )
            MockSU.return_value = mock_su

            report = evaluator.evaluate([self._make_task()])

        assert report.completed_tasks == 0
        assert report.abandonment_rate == 1.0

    def test_task_timeout(self):
        """Agent never satisfies user within max rounds."""
        agent = self._make_mock_agent([("Trying...", [])])

        evaluator = InteractiveEvaluator(agent=agent, max_rounds=3)

        with patch("trainable_openclaw.evaluation.interactive_eval.SimulatedUser") as MockSU:
            mock_su = MagicMock()
            mock_su.initial_message = "Hi!"
            mock_su.history = []
            mock_su.respond.return_value = UserResponse(
                message="Still not right...",
                status="continue",
                satisfaction=0.2,
            )
            MockSU.return_value = mock_su

            report = evaluator.evaluate([self._make_task()])

        assert report.completed_tasks == 0
        # 3 rounds → timeout
        assert len(report.results) == 1
        assert report.results[0].status == "timeout"
        assert report.results[0].rounds == 3

    def test_multiple_tasks_mixed_results(self):
        """Two tasks: one pass, one fail."""
        agent = self._make_mock_agent([("Done!", [])])

        evaluator = InteractiveEvaluator(agent=agent, max_rounds=5)

        with patch("trainable_openclaw.evaluation.interactive_eval.SimulatedUser") as MockSU:
            mock_su1 = MagicMock()
            mock_su1.initial_message = "Hi 1"
            mock_su1.history = []
            mock_su1.respond.return_value = UserResponse(message="Great!", status="complete", satisfaction=1.0)

            mock_su2 = MagicMock()
            mock_su2.initial_message = "Hi 2"
            mock_su2.history = []
            mock_su2.respond.return_value = UserResponse(message="Bad!", status="give_up", satisfaction=0.0)

            MockSU.side_effect = [mock_su1, mock_su2]

            report = evaluator.evaluate([self._make_task("t1"), self._make_task("t2")])

        assert report.total_tasks == 2
        assert report.completed_tasks == 1
        assert report.to_dict()["completion_rate"] == 0.5
        assert report.abandonment_rate == 0.5

    def test_empty_tasks(self):
        agent = self._make_mock_agent()
        evaluator = InteractiveEvaluator(agent=agent)
        report = evaluator.evaluate([])
        assert report.total_tasks == 0
        assert report.completed_tasks == 0


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------

class TestEvalReport:
    def test_to_dict(self):
        results = [
            TaskResult("t1", "retail", True, 1, 1.0, "complete"),
            TaskResult("t2", "retail", True, 3, 0.8, "complete"),
            TaskResult("t3", "airline", False, 5, 0.0, "give_up"),
        ]
        report = EvalReport(
            results=results, total_tasks=3, completed_tasks=2, avg_rounds=3.0,
            first_try_rate=1 / 3, recovery_rate=0.5, abandonment_rate=1 / 3,
        )
        d = report.to_dict()
        assert d["total_tasks"] == 3
        assert d["completed_tasks"] == 2
        assert d["completion_rate"] == round(2 / 3, 3)
        assert d["avg_rounds"] == 3.0
        assert d["first_try_rate"] == round(1 / 3, 3)
        assert d["recovery_rate"] == 0.5

    def test_print(self):
        report = EvalReport(
            results=[], total_tasks=1, completed_tasks=1, avg_rounds=1.0,
            first_try_rate=1.0, recovery_rate=0.0, abandonment_rate=0.0,
        )
        text = report.print()
        assert "INTERACTIVE EVALUATION REPORT" in text
        assert "1 (100.0%)" in text


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------

class TestTaskResult:
    def test_first_try_success(self):
        assert TaskResult("t1", "retail", True, 1, 1.0, "complete").first_try_success is True
        assert TaskResult("t2", "retail", True, 2, 0.9, "complete").first_try_success is False
        assert TaskResult("t3", "retail", False, 1, 0.0, "give_up").first_try_success is False

    def test_to_dict(self):
        tr = TaskResult("t1", "retail", True, 1, 1.0, "complete")
        d = tr.to_dict()
        assert d["task_id"] == "t1"
        assert d["first_try"] is True
