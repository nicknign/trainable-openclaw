# Test Record — 2026/05/25

## Summary

| Suite | Tests | Type | Result |
|-------|-------|------|--------|
| `test_serve_ppo.py` | 9 | Mock (no GPU) | ✅ 9/9 |
| `test_orchestrator.py` | 24 | Mock (no GPU) | ✅ 24/24 |
| `test_a1_integration.py` | 12 | GPU integration | ✅ 12/12 |
| `test_a2_integration.py` | 5 | GPU integration | ✅ 5/5 |
| `test_a3_integration.py` | 9 | GPU integration | ✅ 9/9 |
| **Total** | **59** | | **✅ 59/59** |

## Test Details

### Mock Tests (Local/Any Machine)

**test_serve_ppo.py** — Pydantic models + FastAPI endpoints (mock LLM)
- ChatMessage / ChatCompletionRequest / ChatCompletionResponse models
- Health response model
- Health endpoint
- Chat completions endpoint (with mock generate)
- Default sampling params
- Uninitialized server (503)

**test_orchestrator.py** — Training orchestrator logic
- TrainingSample construction (defaults, metadata)
- Orchestrator construction (defaults, custom, validation)
- Record request (adds samples, accumulates, max buffer)
- ShouldTrain logic (busy, not enough samples, idle+enough, during training)
- Trigger training (callback, buffer clear, mode recovery, error recovery)
- Monitor loop (trigger conditions, busy skip, idempotent start)
- Thread safety (concurrent records)
- Idle tracking (time increases, resets on request)

### GPU Integration Tests (Remote Server with Model)

**test_a1_integration.py** — A1: Inference API
- Health check with GPU count
- Health response field completeness
- Basic chat completions (coherent response, no think block)
- enable_thinking=True
- max_tokens enforcement
- Temperature=0 determinism
- Sampling params override
- Multi-turn conversation context
- 404 error handling
- OpenAI-compatible response format
- Empty/short user messages
- Chinese input

**test_a2_integration.py** — A2: Idle Detection + Training Trigger
- Server starts in serving mode
- Request accumulation (min_samples=2)
- Idle detection triggers training (idle_timeout=5s)
- 503 response during training
- Server recovery to serving mode
- Post-training inference works

**test_a3_integration.py** — A3: Weight Sync + Multi-cycle Stability
- Server starts in serving mode
- Cycle 1: accumulate → idle → train → 503 → recover → inference (no garbage output)
- Cycle 2: accumulate → idle → train → recover → inference
- Final health check: serving mode

## Environment

- **Server**: RTX 3090 48GB, Qwen3-4B + LoRA rank=16
- **Config**: idle_timeout=5, min_samples=2, train_steps_per_cycle=1, gsm8k.enabled=false
- **Key fix**: `asyncio.to_thread(ray.get, ...)` prevents uvicorn event loop blocking during training
