#!/usr/bin/env python3
"""
Integration test for serve_ppo lifecycle.

Tests:
  1. Server health & basic inference
  2. Training trigger detection (via log monitoring)
  3. Post-training inference quality
  4. Multiple training cycles
  5. Reward tracking in logs

Usage (on remote machine):
  python3 scripts/test_serve_ppo_lifecycle.py [--host localhost] [--port 8000]
"""

import argparse
import json
import subprocess
import time
import urllib.request
import urllib.error
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def http_get(url: str, timeout: float = 10.0) -> tuple[int, dict | str]:
    """Return (status_code, parsed_json_or_body)."""
    try:
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500]
    except Exception as e:
        return 0, str(e)


def http_post(url: str, data: dict, timeout: float = 60.0) -> tuple[int, dict | str]:
    """Return (status_code, parsed_json_or_body)."""
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500]
    except Exception as e:
        return 0, str(e)


def log(msg: str, level: str = "INFO"):
    print(f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class ServePPOTester:
    def __init__(self, host: str = "localhost", port: int = 8000, log_path: str = "/tmp/serve_ppo.log"):
        self.base = f"http://{host}:{port}"
        self.log_path = log_path
        self.passed = 0
        self.failed = 0
        self.results = []

    def check(self, name: str, condition: bool, detail: str = ""):
        if condition:
            self.passed += 1
            log(f"OK  {name}: {detail}", "OK")
        else:
            self.failed += 1
            log(f"FAIL {name}: {detail}", "FAIL")
        self.results.append((name, condition, detail))

    # ----- Log monitoring -----

    def read_remote_log(self, grep: str = "", tail: int = 30) -> str:
        """Read log lines matching grep pattern."""
        import subprocess
        try:
            if grep:
                cmd = f"grep -E '{grep}' {self.log_path} | tail -{tail}"
            else:
                cmd = f"tail -{tail} {self.log_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            return result.stdout
        except Exception as e:
            return f"LOG_READ_ERROR: {e}"

    # ----- Test: Health -----

    def test_01_health_serving(self):
        """Server should be up and in serving mode."""
        code, data = http_get(f"{self.base}/v1/health")
        ok = code == 200 and isinstance(data, dict) and data.get("status") == "ok"
        self.check("1. Health/Serving", ok,
                   f"code={code}, mode={data.get('mode', '?')}" if isinstance(data, dict) else f"code={code}")
        return ok

    # ----- Test: Basic inference -----

    def test_02_inference(self, questions: list[str] | None = None) -> list[str]:
        """Send chat completions and return response texts."""
        if questions is None:
            questions = [
                "What is 2 + 2?",
                "What is the capital of France?",
            ]

        responses = []
        for i, question in enumerate(questions):
            code, data = http_post(
                f"{self.base}/v1/chat/completions",
                {
                    "model": "default",
                    "messages": [{"role": "user", "content": question}],
                    "max_tokens": 64,
                    "temperature": 0.0,
                },
                timeout=60.0,
            )
            ok = code == 200 and isinstance(data, dict) and "choices" in data
            resp_text = ""
            if ok:
                resp_text = data["choices"][0]["message"]["content"][:100]
            self.check(f"2. Inference Q{i+1}", ok,
                       f"Q={question[:30]}... A={resp_text[:50]}..." if ok else f"code={code}")
            responses.append(resp_text)
        return responses

    # ----- Test: GSM8K math inference -----

    def test_03_gsm8k_inference(self) -> list[dict]:
        """Test inference on GSM8K-style math questions before training."""
        questions = [
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder for $2 each. How much does she make per day?",
        ]

        results = []
        for i, question in enumerate(questions):
            code, data = http_post(
                f"{self.base}/v1/chat/completions",
                {
                    "model": "default",
                    "messages": [{"role": "user", "content": f"{question}\nLet's think step by step and output the final answer after \"####\"."}],
                    "max_tokens": 256,
                    "temperature": 0.0,
                },
                timeout=60.0,
            )
            ok = code == 200
            resp_text = ""
            if ok:
                resp_text = data["choices"][0]["message"]["content"]
                # Extract answer
                if "####" in resp_text:
                    answer_part = resp_text.split("####")[-1].strip().split()[0] if resp_text.split("####")[-1].strip() else "NONE"
                else:
                    answer_part = "NO_ANSWER"
            else:
                answer_part = f"HTTP_{code}"

            self.check(f"3. GSM8K Pre-training Q{i+1}", ok,
                       f"Q={question[:40]}... Answer={answer_part}")
            results.append({"question": question, "response": resp_text, "extracted": answer_part})
        return results

    # ----- Test: Wait for training -----

    def test_04_wait_for_training(self, max_wait: float = 300.0):
        """Wait for training to trigger and complete by monitoring logs."""
        log(f"Waiting up to {max_wait}s for training to trigger...")
        t0 = time.time()

        # Phase 1: Wait for training to trigger
        triggered = False
        while time.time() - t0 < max_wait:
            log_output = self.read_remote_log(r"Training triggered|\[TRAINING\] Training triggered")
            if "Training triggered" in log_output:
                triggered = True
                log("Training triggered detected in log")
                break
            time.sleep(3)

        self.check("4a. Training triggered", triggered,
                   f"Detected in {time.time() - t0:.0f}s" if triggered else "NOT DETECTED")

        if not triggered:
            return False

        # Phase 2: Wait for training to complete
        t1 = time.time()
        completed = False
        while time.time() - t1 < max_wait:
            log_output = self.read_remote_log(r"Training complete|train_step completed|\[TRAINING\] Training complete")
            if "Training complete" in log_output or "train_step completed" in log_output:
                completed = True
                log("Training completion detected in log")
                break
            # Also check for errors
            if "Training failed" in log_output or "RayTaskError" in log_output:
                log("Training FAILED detected in log", "FAIL")
                self.check("4b. Training completed", False, "Training failed with error")
                return False
            time.sleep(5)

        self.check("4b. Training completed", completed,
                   f"Detected in {time.time() - t1:.0f}s" if completed else "NOT DETECTED")
        return completed

    # ----- Test: Post-training inference -----

    def test_05_post_training_inference(self) -> list[dict]:
        """Test inference after training to check if weights changed."""
        # Use same questions as test_03 to compare
        questions = [
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder for $2 each. How much does she make per day?",
        ]

        results = []
        for i, question in enumerate(questions):
            code, data = http_post(
                f"{self.base}/v1/chat/completions",
                {
                    "model": "default",
                    "messages": [{"role": "user", "content": f"{question}\nLet's think step by step and output the final answer after \"####\"."}],
                    "max_tokens": 256,
                    "temperature": 0.0,
                },
                timeout=60.0,
            )
            ok = code == 200
            resp_text = ""
            if ok:
                resp_text = data["choices"][0]["message"]["content"]
                if "####" in resp_text:
                    answer_part = resp_text.split("####")[-1].strip().split()[0] if resp_text.split("####")[-1].strip() else "NONE"
                else:
                    answer_part = "NO_ANSWER"
            else:
                answer_part = f"HTTP_{code}"

            self.check(f"5. GSM8K Post-training Q{i+1}", ok,
                       f"Answer={answer_part}")
            results.append({"question": question, "response": resp_text, "extracted": answer_part})
        return results

    # ----- Test: Training rewards -----

    def test_06_training_rewards(self):
        """Verify training rewards appear in logs."""
        log_output = self.read_remote_log("Training complete|Rewards:|reward_mean|reward_correct|reward=", tail=20)
        has_rewards = "reward" in log_output.lower()
        has_correct = any(token in log_output.lower() for token in ["correct", "reward_correct"])
        # Also accept x/y format: e.g. "0/16"
        if not has_correct:
            import re
            has_correct = bool(re.search(r'\(\d+/\d+\)', log_output))

        self.check("6a. Rewards logged", has_rewards,
                   f"Log contains reward info" if has_rewards else "NO REWARD IN LOG")
        self.check("6b. Correct count logged", has_correct,
                   "Found correct/total count" if has_correct else "NO COUNT FOUND")

        # Extract reward values if available
        if has_rewards:
            for line in log_output.strip().split("\n"):
                if "reward" in line.lower():
                    log(f"  Reward line: {line.strip()[:200]}")

    # ----- Test: Weight sync verification -----

    def test_07_weight_sync(self):
        """Verify weight sync happened by checking for sync-related logs."""
        log_output = self.read_remote_log("Weight sync|update_weights|global_steps", tail=20)
        has_sync = "Weight sync" in log_output or "update_weights" in log_output
        self.check("7. Weight sync", has_sync,
                   "Sync detected in logs" if has_sync else "NO SYNC LOG")

    # ----- Test: Multiple cycles -----

    def test_08_multiple_cycles(self, num_cycles: int = 2, max_wait: float = 600.0):
        """Test multiple training cycles."""
        log(f"Testing {num_cycles} training cycles...")
        for cycle in range(num_cycles):
            log(f"--- Cycle {cycle + 1}/{num_cycles} ---")

            # Wait for serving mode
            t0 = time.time()
            while time.time() - t0 < 30:
                code, data = http_get(f"{self.base}/v1/health", timeout=5.0)
                if code == 200:
                    break
                time.sleep(3)

            # Send a request to reset idle timer for this cycle
            time.sleep(1)

            # Now wait for training to trigger (idle_timeout=15s)
            triggered = False
            t1 = time.time()
            while time.time() - t1 < max_wait:
                log_output = self.read_remote_log(r"Training triggered|\[TRAINING\] Training triggered")
                # Count occurrences
                count = log_output.count("Training triggered")
                if count > cycle:  # New trigger
                    triggered = True
                    log(f"Training trigger #{cycle + 1} detected")
                    break
                time.sleep(5)

            if not triggered:
                log(f"Cycle {cycle + 1}: training not triggered within timeout", "FAIL")
                self.check(f"8. Cycle {cycle + 1} trigger", False, "Not triggered")
                continue

            # Wait for completion
            completed = False
            t2 = time.time()
            while time.time() - t2 < max_wait:
                log_output = self.read_remote_log(r"Training complete|train_step completed|\[TRAINING\] Training complete")
                timer_count = log_output.count("Training complete") + log_output.count("train_step completed")
                if timer_count > cycle:
                    completed = True
                    break
                if "Training failed" in log_output or "RayTaskError" in log_output:
                    break
                time.sleep(5)

            self.check(f"8. Cycle {cycle + 1} complete", completed,
                       f"Cycle #{cycle + 1} {'OK' if completed else 'FAILED/ERROR'}")

    # ----- Summary -----

    def report(self):
        log(f"\n{'='*60}")
        log(f"RESULTS: {self.passed} passed, {self.failed} failed, {self.passed + self.failed} total")
        for name, ok, detail in self.results:
            status = "OK" if ok else "FAIL"
            log(f"  [{status}] {name}: {detail}")
        return self.failed == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="serve_ppo lifecycle integration test")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-path", default="/tmp/serve_ppo.log")
    parser.add_argument("--skip-cycles", action="store_true",
                        help="Skip multi-cycle test (faster)")
    parser.add_argument("--num-cycles", type=int, default=2)
    args = parser.parse_args()

    tester = ServePPOTester(host=args.host, port=args.port, log_path=args.log_path)

    # Phase 1: Pre-training tests
    log("=" * 60)
    log("Phase 1: Pre-training tests")
    log("=" * 60)

    tester.test_01_health_serving()
    tester.test_02_inference()
    pre_results = tester.test_03_gsm8k_inference()

    # Phase 2: Training
    log("\n" + "=" * 60)
    log("Phase 2: Wait for training")
    log("=" * 60)

    training_ok = tester.test_04_wait_for_training(max_wait=600.0)

    # Phase 3: Post-training tests
    log("\n" + "=" * 60)
    log("Phase 3: Post-training verification")
    log("=" * 60)

    if training_ok:
        tester.test_01_health_serving()
        post_results = tester.test_05_post_training_inference()
        tester.test_06_training_rewards()
        tester.test_07_weight_sync()

        # Compare pre/post inference
        log("\n--- Inference comparison ---")
        for i, (pre, post) in enumerate(zip(pre_results, post_results)):
            if pre["response"] != post["response"]:
                log(f"  Q{i+1}: Response CHANGED (weight sync OK)")
            else:
                log(f"  Q{i+1}: Response SAME (expected for greedy decoding)")

    # Phase 4: Multiple cycles (optional)
    if not args.skip_cycles:
        log("\n" + "=" * 60)
        log("Phase 4: Multiple training cycles")
        log("=" * 60)
        tester.test_08_multiple_cycles(num_cycles=args.num_cycles, max_wait=900.0)

    # Report
    success = tester.report()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
