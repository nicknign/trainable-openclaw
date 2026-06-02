"""Smoke tests for inference API server — no GPU required.

These tests verify:
1. Pydantic models and FastAPI app from trainable_openclaw.server.api
2. Endpoints return correct responses with a mock LLM client
3. OpenAI-compatible request/response formats are valid

Run with:
    python -m pytest tests/test_serve_ppo.py -v
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Test Pydantic models
# ---------------------------------------------------------------------------


class TestChatModels:
    """Verify OpenAI-compatible request/response models."""

    def test_chat_message(self):
        from trainable_openclaw.server.api import ChatMessage

        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

        d = msg.model_dump()
        assert d == {"role": "user", "content": "Hello"}

    def test_chat_completion_request(self):
        from trainable_openclaw.server.api import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            temperature=0.5,
            max_tokens=100,
        )
        assert req.model == "test-model"
        assert len(req.messages) == 1
        assert req.temperature == 0.5
        assert req.max_tokens == 100
        # Defaults
        assert req.stream is False
        assert req.top_p is None

    def test_chat_completion_request_defaults(self):
        from trainable_openclaw.server.api import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")])
        assert req.model == "default"
        assert req.temperature is None

    def test_chat_completion_response(self):
        from trainable_openclaw.server.api import (
            ChatCompletionResponse,
            ChatCompletionResponseChoice,
            ChatMessage,
            UsageInfo,
        )

        resp = ChatCompletionResponse(
            id="test-id",
            created=1234567890,
            model="test",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        d = resp.model_dump()
        assert d["object"] == "chat.completion"
        assert d["choices"][0]["message"]["content"] == "Hello!"
        assert d["usage"]["total_tokens"] == 8

    def test_health_response(self):
        from trainable_openclaw.server.api import HealthResponse

        resp = HealthResponse(status="ok", mode="serving", uptime_seconds=10.5, active_requests=2, gpu_count=4)
        d = resp.model_dump()
        assert d["status"] == "ok"
        assert d["mode"] == "serving"


# ---------------------------------------------------------------------------
# Test FastAPI application
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer that converts text to dummy token IDs."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<|user|>Hello<|assistant|>"
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "Hello! This is a test response."
    return tokenizer


@pytest.fixture
def mock_llm_client():
    """Create a mock LLMServerClient that returns a fixed TokenOutput-like object."""
    client = AsyncMock()
    output = MagicMock()
    output.token_ids = [10, 20, 30]
    output.log_probs = None
    output.stop_reason = "stop"
    client.generate.return_value = output
    return client


@pytest.fixture
def test_app(mock_tokenizer, mock_llm_client):
    """Create a FastAPI test client with mock dependencies."""
    from fastapi.testclient import TestClient

    from trainable_openclaw.server.api import _app_state, create_app

    app = create_app()

    # Inject mock dependencies
    _app_state["llm_client"] = mock_llm_client
    _app_state["tokenizer"] = mock_tokenizer
    _app_state["rollout_config"] = {"temperature": 0.7, "top_p": 1.0, "top_k": -1, "response_length": 2048}
    _app_state["gpu_count"] = 4
    _app_state["active_requests"] = 0
    _app_state["start_time"] = 1000.0

    return TestClient(app)


class TestFastAPIEndpoints:
    """Integration tests for FastAPI endpoints."""

    def test_health(self, test_app):
        response = test_app.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["mode"] == "serving"
        assert "uptime_seconds" in data
        assert data["gpu_count"] == 4

    def test_chat_completions(self, test_app):
        response = test_app.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.5,
                "max_tokens": 100,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello! This is a test response."
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] == 5
        assert data["usage"]["completion_tokens"] == 3
        assert data["usage"]["total_tokens"] == 8

    def test_chat_completions_uses_default_sampling(self, test_app, mock_tokenizer, mock_llm_client):
        """Sampling params fall back to config defaults when not in request."""
        test_app.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

        call_kwargs = mock_llm_client.generate.call_args.kwargs
        assert call_kwargs["sampling_params"]["temperature"] == 0.7
        assert call_kwargs["sampling_params"]["max_tokens"] == 2048

    def test_chat_completions_uninitialized(self):
        """Returns 503 when server not initialized."""
        from fastapi.testclient import TestClient

        from trainable_openclaw.server.api import _app_state, create_app

        app = create_app()
        _app_state.clear()  # Simulate uninitialized state

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert response.status_code == 503


class TestFeedbackModels:
    """Verify Feedback request/response models."""

    def test_feedback_request(self):
        from trainable_openclaw.server.api import FeedbackRequest

        req = FeedbackRequest(session_id="abc", rating=5, correction="请修改")
        assert req.session_id == "abc"
        assert req.rating == 5
        assert req.correction == "请修改"

    def test_feedback_request_with_trace_id(self):
        from trainable_openclaw.server.api import FeedbackRequest

        req = FeedbackRequest(trace_id="t-xyz", rating=3)
        assert req.trace_id == "t-xyz"
        assert req.session_id is None

    def test_feedback_response(self):
        from trainable_openclaw.server.api import FeedbackResponse

        resp = FeedbackResponse(status="ok", event_id=42, event_type="user.accepted")
        d = resp.model_dump()
        assert d["status"] == "ok"
        assert d["event_id"] == 42
        assert d["event_type"] == "user.accepted"


class TestFeedbackEndpoint:
    """Integration tests for /v1/feedback endpoint."""

    @pytest.fixture
    def feedback_app(self, mock_tokenizer, mock_llm_client):
        """Test app with a ConversationStore mock for feedback."""
        from unittest.mock import MagicMock
        from fastapi.testclient import TestClient

        from trainable_openclaw.server.api import _app_state, create_app

        app = create_app()
        _app_state.clear()
        mock_store = MagicMock()
        mock_store.record_telemetry.return_value = 42
        mock_store.conn.execute.return_value.fetchall.return_value = (
            [{"session_id": "abc123"}]
        )

        _app_state["llm_client"] = mock_llm_client
        _app_state["tokenizer"] = mock_tokenizer
        _app_state["rollout_config"] = {
            "temperature": 0.7, "top_p": 1.0, "top_k": -1, "response_length": 2048,
        }
        _app_state["gpu_count"] = 4
        _app_state["active_requests"] = 0
        _app_state["start_time"] = 1000.0
        _app_state["conversation_store"] = mock_store

        return TestClient(app), mock_store

    def test_feedback_accepted(self, feedback_app):
        client, mock_store = feedback_app
        response = client.post(
            "/v1/feedback",
            json={"session_id": "abc123", "rating": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["event_type"] == "user.accepted"

    def test_feedback_corrected(self, feedback_app):
        client, mock_store = feedback_app
        response = client.post(
            "/v1/feedback",
            json={"session_id": "abc123", "rating": 2, "correction": "请用中文回答"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["event_type"] == "user.corrected"

    def test_feedback_abandoned(self, feedback_app):
        client, mock_store = feedback_app
        response = client.post(
            "/v1/feedback",
            json={"session_id": "abc123", "rating": 1},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["event_type"] == "user.abandoned"

    def test_feedback_missing_session_and_trace(self, feedback_app):
        client, _ = feedback_app
        response = client.post(
            "/v1/feedback",
            json={"rating": 3},
        )
        assert response.status_code == 400

    def test_feedback_no_store(self, mock_tokenizer, mock_llm_client):
        from fastapi.testclient import TestClient
        from trainable_openclaw.server.api import _app_state, create_app

        app = create_app()
        _app_state.clear()
        _app_state["llm_client"] = mock_llm_client
        _app_state["tokenizer"] = mock_tokenizer
        _app_state["rollout_config"] = {
            "temperature": 0.7, "top_p": 1.0, "top_k": -1, "response_length": 2048,
        }
        _app_state["gpu_count"] = 4
        _app_state["active_requests"] = 0
        _app_state["start_time"] = 1000.0
        # No conversation_store injected

        client = TestClient(app)
        response = client.post(
            "/v1/feedback",
            json={"session_id": "abc", "rating": 3},
        )
        assert response.status_code == 503
