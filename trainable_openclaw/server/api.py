# Copyright 2026 Trainable OpenClaw Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Standalone FastAPI layer for the veRL inference server.

This module contains ONLY the Pydantic models and FastAPI app factory.
It has NO imports from veRL internals (no ray, no torch, no verl.*)
so it can be imported and tested on any machine without GPU dependencies.

Used by:
- ``verl.trainer.serve_ppo`` — production entry point (wires in Ray infra)
- ``tests.test_serve_ppo`` — smoke tests (mocks llm_client and tokenizer)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models for OpenAI-compatible API
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    enable_thinking: bool = True
    user: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo
    session_id: Optional[str] = None  # Phase 3: link feedback via /v1/feedback


class HealthResponse(BaseModel):
    status: str
    mode: str
    uptime_seconds: float
    active_requests: int
    gpu_count: int


class FeedbackRequest(BaseModel):
    """Telemetry feedback from end-user (Phase 3 — Trajectory-inspired).

    ``session_id`` or ``trace_id`` is required to link the feedback
    back to the original conversation.  Rating follows a 1-5 scale.

    Event type is derived automatically:
    - rating >= 4  → ``user.accepted``
    - rating 2-3 + correction text → ``user.corrected``
    - rating == 1  → ``user.abandoned``
    """
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    rating: int  # 1-5 scale
    correction: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class FeedbackResponse(BaseModel):
    status: str
    event_id: int
    event_type: str


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------

_app_state: dict[str, Any] = {}


def create_app() -> FastAPI:
    """Build the FastAPI application with all endpoints.

    Callers must inject these keys into ``_app_state`` before serving:
    - ``llm_client`` — an async callable with a ``generate()`` method
    - ``tokenizer`` — a HuggingFace-compatible tokenizer object
    - ``rollout_config`` — dict with sampling defaults
    - ``gpu_count`` — int
    - ``start_time`` — set automatically by lifespan
    - ``active_requests`` — int, set to 0 initially
    - ``orch_state`` (A2) — async orchestrator state dict
    - ``record_request`` (A2) — async sample recording callback
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import asyncio
        _app_state["start_time"] = time.time()
        # Schedule the async training monitor on uvicorn's event loop
        monitor_coro_fn = _app_state.get("_monitor_coro")
        monitor_task = asyncio.create_task(monitor_coro_fn()) if monitor_coro_fn else None
        try:
            yield
        finally:
            if monitor_task:
                monitor_task.cancel()
            logger.info("Server shutting down")

    app = FastAPI(title="Trainable OpenClaw - veRL Inference Server", version="0.1.0", lifespan=lifespan)

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        orch_state = _app_state.get("orch_state", {})
        return HealthResponse(
            status="ok",
            mode=orch_state.get("mode", "serving"),
            uptime_seconds=time.time() - _app_state.get("start_time", time.time()),
            active_requests=_app_state.get("active_requests", 0),
            gpu_count=_app_state.get("gpu_count", 0),
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(req: ChatCompletionRequest):
        llm_client = _app_state.get("llm_client")
        tokenizer = _app_state.get("tokenizer")
        rollout_config = _app_state.get("rollout_config")

        if llm_client is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Server not initialized yet")

        orch_state = _app_state.get("orch_state")
        if orch_state is not None and orch_state.get("training_in_progress"):
            raise HTTPException(status_code=503, detail="Training in progress, try later")

        # Build prompt from messages (OpenAI chat format)
        prompt_text = tokenizer.apply_chat_template(
            [m.model_dump() for m in req.messages],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=req.enable_thinking,
        )

        # Tokenize
        prompt_ids = tokenizer.encode(prompt_text)

        # Sampling params -- use request values or fall back to config defaults
        sampling_params = {
            "temperature": req.temperature if req.temperature is not None else rollout_config.get("temperature", 0.7),
            "top_p": req.top_p if req.top_p is not None else rollout_config.get("top_p", 1.0),
            "top_k": req.top_k if req.top_k is not None else rollout_config.get("top_k", -1),
            "max_tokens": req.max_tokens or rollout_config.get("response_length", 2048),
        }

        request_id = f"chatcmpl-{os.urandom(6).hex()}"

        try:
            _app_state["active_requests"] = _app_state.get("active_requests", 0) + 1
            t_req_start = time.time()

            # Call veRL's async generate
            logger.info(
                f"[DIAG] chat_completions {request_id}: "
                f"prompt_len={len(prompt_ids)}, first_10_ids={prompt_ids[:10]!r}, "
                f"last_10_ids={prompt_ids[-10:]!r}, prompt_text={prompt_text[:100]!r}"
            )
            output = await llm_client.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
            latency_ms = (time.time() - t_req_start) * 1000

            # Detokenize response
            logger.info(
                f"[DIAG] chat_completions {request_id} output: "
                f"output_token_len={len(output.token_ids)}, "
                f"first_10_ids={output.token_ids[:10]!r}, "
                f"decoded_text={tokenizer.decode(output.token_ids[:20], skip_special_tokens=False)!r}"
            )
            response_text = tokenizer.decode(output.token_ids, skip_special_tokens=True)
            prompt_tokens = len(prompt_ids)
            completion_tokens = len(output.token_ids)

            # Record as training sample (A2)
            record_fn = _app_state.get("record_request")
            if record_fn is not None:
                await record_fn(prompt_ids, list(output.token_ids))

            # Log to conversation store (Phase 2 — B1 analysis)
            store = _app_state.get("conversation_store")
            chat_session_id: Optional[str] = None
            if store is not None:
                user_id = req.user or "anonymous"
                chat_session_id = store.create_session(user_id, model=req.model)
                trace_meta = {"trace_id": request_id}
                store.add_message(chat_session_id, "user", prompt_text,
                                  token_count=prompt_tokens,
                                  metadata=trace_meta)
                store.add_message(chat_session_id, "assistant", response_text,
                                  token_count=completion_tokens,
                                  latency_ms=latency_ms,
                                  temperature=sampling_params["temperature"],
                                  max_tokens=sampling_params["max_tokens"],
                                  stop_reason=output.stop_reason,
                                  metadata=trace_meta)

            return ChatCompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=req.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=response_text),
                        finish_reason=output.stop_reason or "stop",
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
                session_id=chat_session_id,
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            _app_state["active_requests"] = max(0, _app_state.get("active_requests", 1) - 1)

    @app.post("/v1/feedback", response_model=FeedbackResponse)
    async def submit_feedback(req: FeedbackRequest):
        """Record user feedback on a previous chat completion.

        Accepts either ``session_id`` or ``trace_id`` to link feedback
        to the original conversation.  Rating is 1-5 scale.
        """
        store = _app_state.get("conversation_store")
        if store is None:
            raise HTTPException(status_code=503, detail="Conversation store not available")

        # Resolve session_id from trace_id if needed
        session_id = req.session_id
        if not session_id and req.trace_id:
            # Look up session by trace_id in message metadata
            rows = store.conn.execute(
                "SELECT session_id FROM messages WHERE metadata LIKE ? LIMIT 1",
                (f"%{req.trace_id}%",),
            ).fetchall()
            if rows:
                session_id = rows[0]["session_id"]

        if not session_id:
            raise HTTPException(
                status_code=400,
                detail="session_id or trace_id required to link feedback",
            )

        # Derive event type from rating + correction
        if req.rating >= 4:
            event_type = "user.accepted"
        elif req.correction and req.correction.strip():
            event_type = "user.corrected"
        else:
            event_type = "user.abandoned"

        event_id = store.record_telemetry(
            session_id=session_id,
            event_type=event_type,
            trace_id=req.trace_id,
            rating=req.rating,
            correction=req.correction,
            metadata=req.metadata,
        )

        logger.info(
            "Feedback recorded: session=%s event=%s rating=%s id=%d",
            session_id, event_type, req.rating, event_id,
        )
        return FeedbackResponse(status="ok", event_id=event_id, event_type=event_type)

    return app
