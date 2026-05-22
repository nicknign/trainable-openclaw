#!/usr/bin/env python3
"""A1 GPU integration tests — run on remote server with real model.

Usage:
    python tests/test_a1_integration.py                          # default http://localhost:8000
    python tests/test_a1_integration.py http://remote:8000       # custom server
"""

import json
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"

passed = 0
failed = 0
errors = []


def test(name: str):
    """Decorator-like wrapper for test functions."""
    def decorator(fn):
        global passed, failed
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            msg = f"  FAIL  {name}: {e}"
            print(msg)
            errors.append(msg)
        except Exception as e:
            failed += 1
            msg = f"  ERROR {name}: {type(e).__name__}: {e}"
            print(msg)
            errors.append(msg)

    return decorator


def post(path: str, payload: dict, timeout: int = 120) -> dict:
    """Send POST request and return JSON response."""
    url = f"{BASE_URL}{path}"
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urlopen(req, timeout=timeout)
    return json.loads(resp.read().decode("utf-8"))


def get(path: str) -> dict:
    """Send GET request and return JSON response."""
    url = f"{BASE_URL}{path}"
    resp = urlopen(url, timeout=10)
    return json.loads(resp.read().decode("utf-8"))


print(f"\n{'='*60}")
print(f"  A1 GPU Integration Tests")
print(f"  Server: {BASE_URL}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------


@test("GET /v1/health returns ok with gpu_count > 0")
def _():
    data = get("/v1/health")
    assert data["status"] == "ok", f"status={data['status']}"
    assert data["mode"] == "serving", f"mode={data['mode']}"
    assert data["gpu_count"] > 0, f"gpu_count={data['gpu_count']}"
    assert data["uptime_seconds"] > 0, f"uptime_seconds={data['uptime_seconds']}"
    assert data["active_requests"] == 0, f"active_requests={data['active_requests']}"


@test("Health response has all required fields")
def _():
    data = get("/v1/health")
    required = {"status", "mode", "uptime_seconds", "active_requests", "gpu_count"}
    assert required.issubset(set(data.keys())), f"Missing fields: {required - set(data.keys())}"


# ---------------------------------------------------------------------------
# 2. Basic chat completions
# ---------------------------------------------------------------------------


@test("POST /v1/chat/completions returns coherent response")
def _():
    data = post(
        "/v1/chat/completions",
        {
            "model": "test",
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "enable_thinking": False,
            "max_tokens": 50,
        },
    )
    assert data["object"] == "chat.completion", f"object={data['object']}"
    assert data["model"] == "test"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["index"] == 0
    assert data["choices"][0]["finish_reason"] is not None
    content = data["choices"][0]["message"]["content"]
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(content) > 0, "Empty response content"
    assert not content.startswith("<think>"), "Thinking block leaked with enable_thinking=False"
    # Content should be readable text, not random bytes/binary
    for ch in content:
        assert ord(ch) >= 32 or ch in "\n\t", f"Control character found: ord={ord(ch)}"
    # Usage should be valid
    usage = data["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@test("POST /v1/chat/completions with enable_thinking=True")
def _():
    data = post(
        "/v1/chat/completions",
        {
            "model": "test",
            "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
            "enable_thinking": True,
            "max_tokens": 150,
        },
    )
    content = data["choices"][0]["message"]["content"]
    # Qwen3 thinking mode: content includes <think>...</think> followed by answer
    assert len(content) > 20, f"Response too short: {content!r}"
    # With thinking enabled, the model might or might not output a think block
    # But the response should be coherent


@test("POST /v1/chat/completions respects max_tokens")
def _():
    data_short = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "Count from 1 to 100."}],
            "enable_thinking": False,
            "max_tokens": 10,
        },
    )
    short_tokens = data_short["usage"]["completion_tokens"]
    assert short_tokens <= 12, f"Expected <=12 tokens, got {short_tokens}"  # Allow small overshoot

    data_long = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "Write a long essay about AI."}],
            "enable_thinking": False,
            "max_tokens": 80,
        },
    )
    long_tokens = data_long["usage"]["completion_tokens"]
    assert long_tokens > 5, f"Expected >5 tokens with max_tokens=80, got {long_tokens}"


# ---------------------------------------------------------------------------
# 3. Sampling parameters
# ---------------------------------------------------------------------------


@test("Temperature=0 produces low variance output")
def _():
    results = []
    for _ in range(3):
        data = post(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Say the word 'banana'."}],
                "enable_thinking": False,
                "temperature": 0.0,
                "max_tokens": 30,
            },
        )
        results.append(data["choices"][0]["message"]["content"])
    # With temperature=0, results should be identical or very similar
    # (greedy decoding may still have slight variance due to hardware nondeterminism)
    unique = len(set(results))
    assert unique <= 2, f"Expected deterministic output (<=2 unique), got {unique} unique: {results}"


@test("Sampling params from request override defaults")
def _():
    response = urlopen(Request(f"{BASE_URL}/v1/chat/completions", data=json.dumps({
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.3,
        "top_p": 0.5,
        "max_tokens": 60,
    }).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"), timeout=120)
    data = json.loads(response.read().decode("utf-8"))
    assert data["choices"][0]["message"]["content"], "Should return content"


# ---------------------------------------------------------------------------
# 4. Multi-turn conversation
# ---------------------------------------------------------------------------


@test("Multi-turn conversation preserves context")
def _():
    messages = [
        {"role": "user", "content": "My name is Alice. Remember it."},
    ]
    data1 = post("/v1/chat/completions", {
        "model": "test",
        "messages": messages,
        "enable_thinking": False,
        "max_tokens": 50,
    })
    reply1 = data1["choices"][0]["message"]["content"]
    assert len(reply1) > 0

    # Add assistant reply + follow-up
    messages.append({"role": "assistant", "content": reply1})
    messages.append({"role": "user", "content": "What is my name?"})
    data2 = post("/v1/chat/completions", {
        "model": "test",
        "messages": messages,
        "enable_thinking": False,
        "max_tokens": 50,
    })
    reply2 = data2["choices"][0]["message"]["content"].lower()
    # Should mention Alice somewhere
    assert "alice" in reply2, f"Expected 'Alice' in response, got: {reply2!r}"


# ---------------------------------------------------------------------------
# 5. Error handling
# ---------------------------------------------------------------------------


@test("GET /v1/nonexistent returns 404")
def _():
    try:
        get("/v1/nonexistent")
        raise AssertionError("Expected 404")
    except URLError as e:
        assert "404" in str(e) or "Not Found" in str(e), f"Expected 404, got: {e}"
    except OSError as e:  # urlopen raises OSError for HTTP errors on some Python versions
        assert hasattr(e, 'code') and e.code == 404, f"Expected 404, got: {e}"
        # This is fine, HTTPError with code 404
        pass


# No "training in progress" test here — that requires orchestrator control
# which is part of A2 testing.


# ---------------------------------------------------------------------------
# 6. Response format validation
# ---------------------------------------------------------------------------


@test("Response has valid OpenAI-compatible format")
def _():
    data = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "Ping"}],
            "enable_thinking": False,
            "max_tokens": 10,
        },
    )
    # Top-level fields
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert isinstance(data["created"], int)
    assert data["created"] > 0
    # Choices
    choices = data["choices"]
    assert isinstance(choices, list)
    assert len(choices) >= 1
    choice = choices[0]
    assert isinstance(choice["index"], int)
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] in ("stop", "completed", "length")
    # Usage
    usage = data["usage"]
    assert isinstance(usage["prompt_tokens"], int)
    assert isinstance(usage["completion_tokens"], int)
    assert isinstance(usage["total_tokens"], int)
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


# ---------------------------------------------------------------------------
# 7. Edge cases: varied inputs
# ---------------------------------------------------------------------------


@test("Handles empty/short user messages")
def _():
    data = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "?"}],
            "enable_thinking": False,
            "max_tokens": 30,
        },
    )
    assert len(data["choices"][0]["message"]["content"]) > 0


@test("Handles Chinese input")
def _():
    data = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己"}],
            "enable_thinking": False,
            "max_tokens": 80,
        },
    )
    content = data["choices"][0]["message"]["content"]
    assert len(content) > 0
    # Should contain Chinese characters or coherent response
    assert any("\u4e00" <= c <= "\u9fff" for c in content), f"No Chinese in response: {content!r}"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"  Results: {passed}/{passed+failed} passed")
if errors:
    print(f"\n  Failures:")
    for e in errors:
        print(f"    {e}")
print(f"{'='*60}\n")

sys.exit(0 if failed == 0 else 1)
