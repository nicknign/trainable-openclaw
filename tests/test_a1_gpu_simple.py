#!/usr/bin/env python3
"""A1 GPU integration tests — run on remote server with real GPU.

Usage:
    python tests/test_a1_gpu_simple.py [server_url]
"""

import json
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"

passed = 0
failed = 0


def post(payload, timeout=120):
    url = f"{BASE_URL}/v1/chat/completions"
    req = Request(url, data=json.dumps(payload).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urlopen(req, timeout=timeout).read().decode("utf-8"))


def get(path):
    return json.loads(urlopen(f"{BASE_URL}{path}", timeout=10).read().decode("utf-8"))


def T(name):
    """Minimal test runner: assert inside the body, return normally."""
    def wrap(fn):
        global passed, failed
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    return wrap


print(f"\n{'='*60}")
print(f"  A1 GPU Integration Tests")
print(f"  Server: {BASE_URL}")
print(f"{'='*60}\n")

# ---- Core API Tests ----

MSG = {"role": "user", "content": "Greeting, one word only."}


@T("GET /v1/health returns ok and gpu_count=1")
def _():
    d = get("/v1/health")
    assert d["status"] == "ok"
    assert d["gpu_count"] == 1
    assert d["uptime_seconds"] > 0


@T("Health has all required fields")
def _():
    d = get("/v1/health")
    required = {"status", "mode", "uptime_seconds", "active_requests", "gpu_count"}
    assert required.issubset(set(d.keys()))


@T("Basic chat completion (thinking=False)")
def _():
    d = post({"messages": [MSG], "enable_thinking": False, "max_tokens": 30})
    assert d["object"] == "chat.completion"
    content = d["choices"][0]["message"]["content"]
    assert len(content) > 0
    # No thinking block when disabled
    assert not content.startswith("<think>")


@T("Basic chat completion (thinking=True)")
def _():
    d = post({"messages": [MSG], "enable_thinking": True, "max_tokens": 80})
    content = d["choices"][0]["message"]["content"]
    assert len(content) > 10


@T("OpenAI-compatible response format")
def _():
    d = post({"messages": [MSG], "enable_thinking": False, "max_tokens": 15})
    assert d["id"].startswith("chatcmpl-")
    assert isinstance(d["created"], int) and d["created"] > 0
    assert d["choices"][0]["index"] == 0
    assert d["choices"][0]["message"]["role"] == "assistant"
    assert d["choices"][0]["finish_reason"] in ("stop", "completed", "length")


@T("Usage token counts are valid")
def _():
    d = post({"messages": [MSG], "enable_thinking": False, "max_tokens": 20})
    assert d["usage"]["prompt_tokens"] > 0
    assert d["usage"]["completion_tokens"] > 0
    assert d["usage"]["total_tokens"] == d["usage"]["prompt_tokens"] + d["usage"]["completion_tokens"]


@T("max_tokens=8 respected (short output)")
def _():
    d = post({"messages": [MSG], "enable_thinking": False, "max_tokens": 8})
    assert d["usage"]["completion_tokens"] <= 12


@T("Custom sampling params override defaults")
def _():
    d = post({"messages": [MSG], "temperature": 0.3, "top_p": 0.5, "max_tokens": 25})
    assert len(d["choices"][0]["message"]["content"]) > 0


@T("Chinese input returns Chinese response")
def _():
    d = post({"messages": [{"role": "user", "content": "说一个中文词"}],
              "enable_thinking": False, "max_tokens": 30})
    content = d["choices"][0]["message"]["content"]
    assert any("\u4e00" <= c <= "\u9fff" for c in content), f"No Chinese: {content!r}"


@T("404 for nonexistent endpoint")
def _():
    try:
        urlopen(f"{BASE_URL}/v1/nonexistent", timeout=10)
        raise AssertionError("Expected 404")
    except HTTPError as e:
        assert e.code == 404


# ---- Summary ----

print(f"\n{'='*60}")
print(f"  Results: {passed}/{passed+failed} passed")
if failed:
    print(f"  ({failed} failure(s))")
print(f"{'='*60}\n")

sys.exit(0 if failed == 0 else 1)
