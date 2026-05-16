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


class HealthResponse(BaseModel):
    status: str
    mode: str
    uptime_seconds: float
    active_requests: int
    gpu_count: int


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
    - ``orchestrator`` (A2, optional) — TrainingOrchestrator instance
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _app_state["start_time"] = time.time()
        yield
        logger.info("Server shutting down")

    app = FastAPI(title="Trainable OpenClaw - veRL Inference Server", version="0.1.0", lifespan=lifespan)

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            mode="serving",
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

        orchestrator = _app_state.get("orchestrator")
        if orchestrator is not None and orchestrator.training_in_progress:
            raise HTTPException(status_code=503, detail="Training in progress, try later")

        # Build prompt from messages (OpenAI chat format)
        prompt_text = tokenizer.apply_chat_template(
            [m.model_dump() for m in req.messages],
            tokenize=False,
            add_generation_prompt=True,
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

            # Call veRL's async generate
            output = await llm_client.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )

            # Record as training sample (A2)
            if orchestrator is not None:
                orchestrator.record_request(prompt_ids, list(output.token_ids))

            # Detokenize response
            response_text = tokenizer.decode(output.token_ids, skip_special_tokens=True)
            prompt_tokens = len(prompt_ids)
            completion_tokens = len(output.token_ids)

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
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            _app_state["active_requests"] = max(0, _app_state.get("active_requests", 1) - 1)

    return app
