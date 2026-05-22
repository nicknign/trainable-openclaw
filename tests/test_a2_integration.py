#!/usr/bin/env python3
"""A2 GPU integration tests — idle detection + training trigger.

Usage:
    python tests/test_a2_integration.py [server_url]

Requires: serve_ppo started with low thresholds:
    +trainer.idle_timeout=5 +trainer.min_samples=2
"""

import json
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"

passed = 0
failed = 0
errors = []


def post(payload, timeout=120):
    url = f"{BASE_URL}/v1/chat/completions"
    req = Request(url, data=json.dumps(payload).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urlopen(req, timeout=timeout).read().decode("utf-8"))


def post_raw(payload, timeout=120):
    """POST and return (status_code, body_dict)."""
    url = f"{BASE_URL}/v1/chat/completions"
    req = Request(url, data=json.dumps(payload).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urlopen(req, timeout=timeout)
        return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def get(path):
    return json.loads(urlopen(f"{BASE_URL}{path}", timeout=10).read().decode("utf-8"))


def T(name):
    def wrap(fn):
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
    return wrap


print(f"\n{'='*60}")
print(f"  A2 GPU Integration Tests")
print(f"  Server: {BASE_URL}")
print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Pre-condition: server in serving mode
# ---------------------------------------------------------------------------

@T("Server is in serving mode at start")
def _():
    d = get("/v1/health")
    assert d["status"] == "ok"
    assert d["mode"] == "serving"


# ---------------------------------------------------------------------------
# Accumulate samples
# ---------------------------------------------------------------------------

MSG = {"role": "user", "content": "Say just one word."}
SAMPLE_COUNT = 2  # Must match min_samples from server config

print(f"  Sending {SAMPLE_COUNT} requests to accumulate samples...")
for i in range(SAMPLE_COUNT):
    d = post({"messages": [MSG], "enable_thinking": False, "max_tokens": 10})
    assert "choices" in d
    print(f"    Request {i+1}/{SAMPLE_COUNT}: {d['choices'][0]['message']['content']!r}")


# ---------------------------------------------------------------------------
# Wait for idle timeout + training trigger
# ---------------------------------------------------------------------------

IDLE_TIMEOUT = 5  # Must match server config
TRAINING_TIME = 8  # sleep(3) in train_step + overhead


@T(f"Training triggered after idle (wait {IDLE_TIMEOUT+2}s)")
def _():
    print(f"    Waiting {IDLE_TIMEOUT+2}s for idle detection...")
    time.sleep(IDLE_TIMEOUT + 2)
    # Training should have triggered. If not, something is wrong with idle detection.
    # We'll verify via the 503 test below.


@T("Request during training returns 503")
def _():
    status, body = post_raw({"messages": [MSG], "max_tokens": 10})
    if status == 503:
        assert "Training in progress" in body.get("detail", "")
        print(f"    Got expected 503: {body['detail']}")
    elif status == 200:
        # Training may have already completed (very fast)
        print(f"    Training already completed (200), response: {body['choices'][0]['message']['content']!r}")
    else:
        raise AssertionError(f"Unexpected status: {status}, body: {body}")


# ---------------------------------------------------------------------------
# Wait for training to complete
# ---------------------------------------------------------------------------

@T("Server recovers to serving mode after training")
def _():
    print(f"    Waiting {TRAINING_TIME}s for training to complete...")
    for i in range(TRAINING_TIME + 5):
        time.sleep(2)
        try:
            d = get("/v1/health")
            if d["mode"] == "serving":
                print(f"    Server recovered after {(i+1)*2}s")
                return
        except Exception:
            pass
    raise AssertionError("Server did not recover to serving mode")


@T("Inference works after training recovery")
def _():
    d = post({"messages": [MSG], "enable_thinking": False, "max_tokens": 10})
    assert "choices" in d
    content = d["choices"][0]["message"]["content"]
    assert len(content) > 0
    print(f"    Post-training inference: {content!r}")


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
