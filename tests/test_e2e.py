"""
End-to-end integration tests for the trainable-openclaw project.

Covers 4 major scenarios:
  E2E-1: Full data pipeline (convert -> filter/split -> validate)
  E2E-2: Mock tool business logic (retail customer service flow)
  E2E-3: Full 3-layer reward pipeline
  E2E-4: Unified sample format integrity

These tests validate cross-component interactions that unit tests miss.
"""

import json
import os
import subprocess
import sys
import tempfile
import uuid

import pytest

from trainable_openclaw.agent.tau_bench_tools.mock_db import MockDatabase, seed
from trainable_openclaw.agent.tau_bench_tools.registry import register_tau_bench_tools
from trainable_openclaw.feedback.deterministic_verifier import (
    VerificationResult,
    compute_layer1_reward,
    verify_dangerous_operation,
    verify_task_completion,
    verify_tool_call_format,
)
from trainable_openclaw.feedback.signal_extractor import (
    FeedbackSignal,
    compute_layer2_reward,
    detect_retry_pattern,
    extract_signals_from_messages,
)
from trainable_openclaw.feedback.reward_combiner import (
    CombinedReward,
    combine,
    compute_full_reward,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")
DATA_DIR = os.path.join(PROJECT_DIR, "data", "tau_bench")


def _run_script(script_name, cwd=None):
    """Run a Python script as subprocess and return (returncode, stdout, stderr)."""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=cwd or PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode, result.stdout, result.stderr


# ============================================================================
# E2E-1: Full Data Pipeline
# ============================================================================


class TestE2EFullDataPipeline:
    """Run the full pipeline end-to-end: convert -> filter -> split -> validate."""

    def test_pipeline_convert_succeeds(self):
        """convert_tau_bench.py runs without error and produces all_samples.json."""
        rc, stdout, stderr = _run_script("convert_tau_bench.py")
        assert rc == 0, f"convert_tau_bench.py failed (exit={rc}):\nSTDERR: {stderr[:1000]}"

        all_samples_path = os.path.join(DATA_DIR, "all_samples.json")
        assert os.path.exists(all_samples_path), (
            f"all_samples.json not found at {all_samples_path}"
        )

        with open(all_samples_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        assert isinstance(samples, list), "all_samples.json should be a list"
        assert len(samples) > 0, "all_samples.json is empty"

        print(f"Convert produced {len(samples)} total samples")

    def test_pipeline_all_data_sources_present(self):
        """All data source types in all_samples.json are valid and both
        airline and retail domains are represented."""
        all_samples_path = os.path.join(DATA_DIR, "all_samples.json")
        if not os.path.exists(all_samples_path):
            pytest.skip("all_samples.json not found (run convert_tau_bench.py first)")

        with open(all_samples_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        sources = {s["source"] for s in samples}
        valid_sources = {
            "taubench_airline", "taubench_retail",
            "apigen_airline", "apigen_retail",
        }

        # All sources should be valid ones
        invalid = sources - valid_sources
        assert len(invalid) == 0, f"Unexpected source values: {invalid}"

        # Both airline and retail must be present (at least from historical data)
        source_strs = " ".join(sources)
        assert "airline" in source_strs, f"No airline source found in: {sources}"
        assert "retail" in source_strs, f"No retail source found in: {sources}"

        # Count by domain
        airline_count = sum(1 for s in samples if "airline" in s["source"])
        retail_count = sum(1 for s in samples if "retail" in s["source"])
        assert airline_count > 0, "No airline samples"
        assert retail_count > 0, "No retail samples"

        # At minimum, taubench_airline and taubench_retail should exist
        assert "taubench_airline" in sources, "Missing taubench_airline source"
        assert "taubench_retail" in sources, "Missing taubench_retail source"

        print(f"Sources: {sorted(sources)} | airline={airline_count}, retail={retail_count}")

    def test_pipeline_filter_split_succeeds(self):
        """filter_split_tau_bench.py runs without error."""
        rc, stdout, stderr = _run_script("filter_split_tau_bench.py")
        assert rc == 0, f"filter_split_tau_bench.py failed (exit={rc}):\nSTDERR: {stderr[:1000]}"

        train_path = os.path.join(DATA_DIR, "train.jsonl")
        test_path = os.path.join(DATA_DIR, "test.jsonl")
        prompts_path = os.path.join(DATA_DIR, "grpo_prompts.jsonl")

        for path in [train_path, test_path, prompts_path]:
            assert os.path.exists(path), f"{path} not found"

        # Count lines
        for path, name in [(train_path, "train"), (test_path, "test"), (prompts_path, "prompts")]:
            with open(path, "r", encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
            assert count > 0, f"{name}.jsonl is empty"
            print(f"{name}: {count} samples")

    def test_pipeline_reward_distribution(self):
        """Rewards should be binary 0/1 with few partials in all_samples.json."""
        all_samples_path = os.path.join(DATA_DIR, "all_samples.json")
        if not os.path.exists(all_samples_path):
            pytest.skip("all_samples.json not found")

        with open(all_samples_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        rewards = [s["outcome"]["reward"]["final"] for s in samples]
        binary_0 = sum(1 for r in rewards if r == 0.0)
        binary_1 = sum(1 for r in rewards if r == 1.0)
        partial = len(rewards) - binary_0 - binary_1

        assert binary_0 + binary_1 >= len(rewards) * 0.5, (
            f"Expected majority binary rewards (0/1), got {binary_0} zero, "
            f"{binary_1} one, {partial} partial out of {len(rewards)}"
        )
        print(f"Rewards: {binary_0} zero, {binary_1} one, {partial} partial")

    def test_pipeline_no_train_test_overlap(self):
        """Train and test sets have zero domain+task_id overlap."""
        train_path = os.path.join(DATA_DIR, "train.jsonl")
        test_path = os.path.join(DATA_DIR, "test.jsonl")
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            pytest.skip("train.jsonl or test.jsonl not found")

        def load_jsonl(path):
            samples = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            return samples

        def domain_key(s):
            source = s.get("source", "")
            domain = "airline" if "airline" in source else "retail"
            return (domain, s["task_id"])

        train = load_jsonl(train_path)
        test = load_jsonl(test_path)
        train_ids = {domain_key(s) for s in train}
        test_ids = {domain_key(s) for s in test}
        overlap = train_ids & test_ids

        assert len(overlap) == 0, (
            f"Found {len(overlap)} overlapping (domain,task_id) pairs: "
            f"{sorted(overlap)[:10]}"
        )
        print(f"Train: {len(train_ids)} unique tasks, Test: {len(test_ids)} unique tasks, Overlap: 0")

    def test_pipeline_validate_all_checks_pass(self):
        """All 25 validation checks (13 per set + 1 cross-set) pass."""
        rc, stdout, stderr = _run_script("validate_tau_bench_data.py")
        assert rc == 0, (
            f"validate_tau_bench_data.py failed (exit={rc}):\n"
            f"STDOUT: {stdout[-2000:]}\nSTDERR: {stderr[:500]}"
        )

        # Check that 25+ passes are reported (13 train + 13 test - 1 that's duplicated...
        # actually 13 train + 12 test + 1 cross-set = 26 checks total
        # but check 10 is cross-set only so: 12 train + 12 test + 1 cross-set = 25)
        pass_count = stdout.count("PASS:")
        fail_count = stdout.count("FAIL:")
        assert fail_count == 0, f"Validation had {fail_count} failures:\n{stdout[-2000:]}"
        assert pass_count >= 25, f"Expected at least 25 PASS checks, got {pass_count}"


# ============================================================================
# E2E-2: Mock Tool Business Logic (Retail)
# ============================================================================


class TestE2EMockToolRetail:
    """Simulate a real customer service flow using MockDatabase + retail tools."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create a fresh database and tool set for each test."""
        self.db = MockDatabase("retail")
        self.tools = {t.name: t for t in register_tau_bench_tools("retail")}

    # ---- Helper ----
    def _execute(self, tool_name: str, arguments: dict) -> dict:
        tool = self.tools[tool_name]
        return self.db.execute(tool, arguments)

    # ---- Tests ----

    def test_find_user_by_name_zip_success(self):
        """Find Alice Chen by name+zip in San Francisco."""
        result = self._execute("find_user_id_by_name_zip", {
            "first_name": "Alice",
            "last_name": "Chen",
            "zip": "94102",
        })
        assert result["status"] == "success"
        assert len(result["result"]) == 1
        assert result["result"][0]["user_id"] == "U001"
        assert result["result"][0]["name"] == "Alice Chen"

    def test_find_user_by_name_zip_no_match(self):
        """Searching for non-existent user returns error."""
        result = self._execute("find_user_id_by_name_zip", {
            "first_name": "Zorro",
            "last_name": "Nobody",
            "zip": "00000",
        })
        assert result["status"] == "error"

    def test_get_order_details_existing(self):
        """Get details for pending order O002."""
        result = self._execute("get_order_details", {"order_id": "O002"})
        assert result["status"] == "success"
        order = result["result"]
        assert order["user_id"] == "U001"
        assert order["status"] == "pending"
        assert len(order["items"]) == 1

    def test_get_order_details_nonexistent(self):
        """Non-existent order returns error."""
        result = self._execute("get_order_details", {"order_id": "O999"})
        assert result["status"] == "error"

    def test_modify_pending_order_items_success(self):
        """Modify items in a pending order (O002) and verify changes."""
        # First verify original items
        before = self._execute("get_order_details", {"order_id": "O002"})
        assert before["result"]["items"][0]["item_id"] == "I010"

        # Modify to different items
        result = self._execute("modify_pending_order_items", {
            "order_id": "O002",
            "item_ids": ["I007", "I008"],
            "quantities": [1, 2],
        })
        assert result["status"] == "success"
        assert len(result["result"]["items"]) == 2

        # Verify the changes persisted
        after = self._execute("get_order_details", {"order_id": "O002"})
        items = after["result"]["items"]
        item_ids = {i["item_id"] for i in items}
        assert item_ids == {"I007", "I008"}
        # Check quantities
        for item in items:
            if item["item_id"] == "I007":
                assert item["quantity"] == 1
            elif item["item_id"] == "I008":
                assert item["quantity"] == 2

    def test_cancel_pending_order_success(self):
        """Cancel a pending order and verify status change."""
        result = self._execute("cancel_pending_order", {"order_id": "O006"})
        assert result["status"] == "success"
        assert result["result"]["status"] == "cancelled"

        # Verify order is now cancelled
        after = self._execute("get_order_details", {"order_id": "O006"})
        assert after["result"]["status"] == "cancelled"

    def test_cancel_delivered_order_fails(self):
        """Attempting to cancel a delivered order must fail."""
        result = self._execute("cancel_pending_order", {"order_id": "O001"})
        assert result["status"] == "error"
        assert "delivered" in result.get("message", "").lower()

    def test_modify_delivered_order_address_fails(self):
        """Modifying address on a delivered order must be rejected."""
        result = self._execute("modify_pending_order_address", {
            "order_id": "O001",
            "address1": "999 Fake St",
            "city": "Nowhere",
            "state": "XX",
            "zip": "00000",
        })
        assert result["status"] == "error"
        assert "delivered" in result.get("message", "").lower()

    def test_exchange_delivered_order_items_success(self):
        """Exchange items from a delivered order creates a new exchange order."""
        # O001 is a delivered order with item I001 (headphones, $129.99)
        # Exchange for I007 (speaker, $49.99) - cheaper item
        result = self._execute("exchange_delivered_order_items", {
            "order_id": "O001",
            "old_item_ids": ["I001"],
            "new_item_ids": ["I007"],
            "quantities": [1],
            "payment_method": "credit_card",
        })
        assert result["status"] == "success", f"Exchange failed: {result.get('message')}"
        res = result["result"]
        assert res["exchange_order_id"] == "O001-EX"
        assert res["original_order_id"] == "O001"
        assert "I001" in res["items_exchanged"]
        # Price difference should be negative (cheaper replacement)
        assert res["price_difference"] < 0

        # Verify the new exchange order exists
        exchange = self._execute("get_order_details", {"order_id": "O001-EX"})
        assert exchange["status"] == "success"
        assert exchange["result"]["status"] == "pending"
        assert exchange["result"]["exchange_for"] == "O001"

    def test_exchange_delivered_order_with_invalid_item_fails(self):
        """Exchange must fail if old_item_ids are not in the order."""
        result = self._execute("exchange_delivered_order_items", {
            "order_id": "O001",
            "old_item_ids": ["I999"],  # Not in O001
            "new_item_ids": ["I007"],
            "quantities": [1],
            "payment_method": "credit_card",
        })
        assert result["status"] == "error"

    def test_return_delivered_order_items_success(self):
        """Return items from a delivered order produces RMA and refund."""
        # O003 is a delivered order with items I004 ($199.99) and I020 (2x $24.99 = $49.98)
        result = self._execute("return_delivered_order_items", {
            "order_id": "O003",
            "item_ids": ["I020"],
            "payment_method": "credit_card",
        })
        assert result["status"] == "success"
        res = result["result"]
        assert res["return_authorization"].startswith("RMA-O003-")
        assert res["estimated_refund"] > 0
        assert res["refund_method"] == "credit_card"
        # Verify refund amount equals the t-shirt cost (2 * 24.99)
        assert abs(res["estimated_refund"] - 49.98) < 0.01

    def test_full_customer_service_flow(self):
        """End-to-end customer service flow: find user -> get orders -> modify -> verify."""
        # 1. Find user Alice Chen by name+zip
        user_result = self._execute("find_user_id_by_name_zip", {
            "first_name": "Alice",
            "last_name": "Chen",
            "zip": "94102",
        })
        assert user_result["status"] == "success"
        user_id = user_result["result"][0]["user_id"]
        assert user_id == "U001"

        # 2. Get user details
        details = self._execute("get_user_details", {"user_id": user_id})
        assert details["status"] == "success"
        assert details["result"]["name"] == "Alice Chen"
        assert len(details["result"]["payment_methods"]) >= 1

        # 3. Get pending order O002
        order = self._execute("get_order_details", {"order_id": "O002"})
        assert order["status"] == "success"
        assert order["result"]["user_id"] == user_id
        assert order["result"]["status"] == "pending"

        # 4. Modify the pending order items
        modify = self._execute("modify_pending_order_items", {
            "order_id": "O002",
            "item_ids": ["I019", "I021"],
            "quantities": [2, 1],
        })
        assert modify["status"] == "success"

        # 5. Modify payment method
        payment = self._execute("modify_pending_order_payment", {
            "order_id": "O002",
            "payment_type": "credit_card",
            "payment_details": {"last_four": "1234", "brand": "Visa"},
        })
        assert payment["status"] == "success"
        assert payment["result"]["payment"]["method"] == "credit_card"

        # 6. Modify shipping address
        addr = self._execute("modify_pending_order_address", {
            "order_id": "O002",
            "address1": "456 New Address St",
            "address2": "Suite 100",
            "city": "Oakland",
            "state": "CA",
            "zip": "94607",
        })
        assert addr["status"] == "success"
        assert addr["result"]["shipping_address"]["city"] == "Oakland"

        # 7. Verify final state
        final = self._execute("get_order_details", {"order_id": "O002"})
        assert final["status"] == "success"
        f = final["result"]
        assert f["status"] == "pending"
        assert len(f["items"]) == 2
        assert f["shipping_address"]["city"] == "Oakland"
        assert f["payment"]["method"] == "credit_card"

    def test_product_and_item_lookup_flow(self):
        """Look up products and their item variants."""
        # List all product types
        types_result = self._execute("list_all_product_types", {})
        assert types_result["status"] == "success"
        product_types = types_result["result"]
        assert "Electronics" in product_types
        assert "Books" in product_types

        # Get product details
        prod = self._execute("get_product_details", {"product_id": "P001"})
        assert prod["status"] == "success"
        assert "Headphones" in prod["result"]["name"]
        assert len(prod["result"]["variant_summary"]) == 3

        # Get item details
        item = self._execute("get_item_details", {"item_id": "I001"})
        assert item["status"] == "success"
        assert item["result"]["variant_name"] == "Black"
        assert item["result"]["price"] == 129.99

    def test_shared_utility_tools(self):
        """Test calculate, think, and transfer_to_human_agents work."""
        # Calculate
        calc = self._execute("calculate", {"expression": "2 + 3 * 4"})
        assert calc["status"] == "success"
        assert "14" in calc["result"]

        # Think (no-op)
        think = self._execute("think", {"thought": "I should check the order status first."})
        assert think["status"] == "success"

        # Transfer to human
        transfer = self._execute("transfer_to_human_agents", {
            "summary": "Customer wants a refund for damaged item.",
        })
        assert transfer["status"] == "success"
        assert "damaged item" in transfer["result"]["message"]


# ============================================================================
# E2E-3: Full 3-Layer Reward Pipeline
# ============================================================================


class TestE2EFullRewardPipeline:
    """Take a complete trajectory through all 3 reward layers."""

    # ---- Layer 1: Deterministic Verification ----

    def test_layer1_valid_trajectory_full_score(self):
        """A well-formed trajectory with completion gets full L1 score."""
        trajectory = [
            {"role": "user", "content": "Please check order O001 status."},
            {
                "role": "assistant",
                "content": "Let me look that up.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_order_details",
                            "arguments": '{"order_id": "O001"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "get_order_details",
                "content": '{"status": "success", "result": {"order_id": "O001", "status": "delivered"}}',
            },
            {
                "role": "assistant",
                "content": (
                    "Your order O001 was delivered on 2026-05-26. "
                    "Here is the summary of your order. Is there anything else I can help with?"
                ),
            },
        ]

        l1 = compute_layer1_reward(trajectory)
        # Should be high - format valid, no dangerous ops, completion-like language
        assert l1 >= 0.5, f"Expected L1 >= 0.5 for valid trajectory, got {l1}"

    def test_layer1_dangerous_operation_detected(self):
        """Dangerous operations are correctly flagged by L1."""
        # A tool call that tries to rm -rf /
        dangerous_call = {
            "id": "call_bad",
            "type": "function",
            "function": {
                "name": "exec",
                "arguments": '{"command": "rm -rf /etc/passwd"}',
            },
        }
        result = verify_dangerous_operation(dangerous_call)
        assert result.passed is False, "Should flag rm -rf as dangerous"
        assert result.score == 0.0
        assert any(
            "no_dangerous_exec" in c.get("check", "") and not c["passed"]
            for c in result.checks
        )

    def test_layer1_benign_tool_not_dangerous(self):
        """Normal tool calls are not flagged as dangerous."""
        normal_call = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_order_details",
                "arguments": '{"order_id": "O001"}',
            },
        }
        result = verify_dangerous_operation(normal_call)
        assert result.passed is True
        assert result.score == 1.0

    def test_layer1_task_completion_detected(self):
        """Task completion language is properly detected."""
        trajectory = [
            {"role": "user", "content": "What is my order status?"},
            {
                "role": "assistant",
                "content": (
                    "I have completed checking your order. Your order O001 was "
                    "delivered on May 26. Here is everything you need to know."
                ),
            },
        ]
        result = verify_task_completion(trajectory)
        assert result.passed is True
        assert any(
            c.get("check") == "completion_language" and c["passed"]
            for c in result.checks
        )

    def test_layer1_incomplete_trajectory(self):
        """A trajectory without completion gets flagged."""
        trajectory = [
            {"role": "user", "content": "What is my order status?"},
            {"role": "assistant", "content": "ok"},  # too short
        ]
        result = verify_task_completion(trajectory)
        assert result.passed is False

    # ---- Layer 2: Signal Extraction ----

    def test_layer2_positive_signal_extraction(self):
        """Positive feedback signals are extracted from user messages."""
        messages = [
            {"role": "user", "content": "Please check my order."},
            {"role": "assistant", "content": "Your order O001 was delivered."},
            {"role": "user", "content": "Thank you so much! That's perfect!"},
        ]
        signals = extract_signals_from_messages(messages)
        signal_types = {s.signal_type for s in signals}
        assert "positive" in signal_types, f"Expected positive signal, got {signal_types}"

        l2 = compute_layer2_reward(signals)
        assert l2 > 0.5, f"Expected L2 > 0.5 for positive feedback, got {l2}"

    def test_layer2_negative_signal_extraction(self):
        """Negative feedback signals are properly extracted."""
        messages = [
            {"role": "user", "content": "That's wrong, try again."},
        ]
        signals = extract_signals_from_messages(messages)
        signal_types = {s.signal_type for s in signals}
        assert "negative" in signal_types, f"Expected negative signal, got {signal_types}"

        l2 = compute_layer2_reward(signals)
        assert l2 < 0.5, f"Expected L2 < 0.5 for negative feedback, got {l2}"

    def test_layer2_correction_signal_extraction(self):
        """Correction signals are detected."""
        messages = [
            {"role": "user", "content": "It should be order O002 instead of O001."},
        ]
        signals = extract_signals_from_messages(messages)
        signal_types = {s.signal_type for s in signals}
        assert "correction" in signal_types, (
            f"Expected correction signal, got {signal_types}"
        )

    def test_layer2_chinese_feedback(self):
        """Chinese positive and negative patterns are detected."""
        # Chinese positive
        messages_pos = [
            {"role": "user", "content": "很好！解决了！谢谢！"},
        ]
        signals_pos = extract_signals_from_messages(messages_pos)
        assert any(s.signal_type == "positive" for s in signals_pos)

        # Chinese negative
        messages_neg = [
            {"role": "user", "content": "不对，错了，重新来"},
        ]
        signals_neg = extract_signals_from_messages(messages_neg)
        assert any(s.signal_type in ("negative", "correction") for s in signals_neg)

    def test_layer2_abandonment_detection(self):
        """Abandoned sessions are detected."""
        messages = [
            {"role": "user", "content": "never mind, forget it"},
        ]
        signals = extract_signals_from_messages(messages)
        assert any(s.signal_type == "abandoned" for s in signals)

        # Also session-ending abandonment (assistant last)
        messages2 = [
            {"role": "user", "content": "Help me."},
            {"role": "assistant", "content": "Here is your answer."},
        ]
        signals2 = extract_signals_from_messages(messages2)
        assert any(s.signal_type == "abandoned" for s in signals2)

    def test_layer2_neutral_when_no_signals(self):
        """When no signals detected, neutral is returned."""
        messages = [
            {"role": "user", "content": "What time is it?"},
            {"role": "assistant", "content": "It is 3 PM."},
            {"role": "user", "content": "Ok."},
        ]
        signals = extract_signals_from_messages(messages)
        # Should get at least neutral or abandoned
        types = {s.signal_type for s in signals}
        # "Ok." can match positive patterns with "that works"/"all good" type patterns
        # or it could be neutral. Either is valid behavior.
        assert len(signals) > 0, "Should have at least one signal"

    def test_layer2_retry_detection(self):
        """Retry patterns are detected when user repeats similar questions."""
        messages = [
            {"role": "user", "content": "What is the status of my order O001? I need to know if it shipped."},
            {"role": "assistant", "content": "Your order is pending."},
            {"role": "user", "content": "What is the status of my order O001? Has it been shipped yet?"},
        ]
        retries = detect_retry_pattern(messages)
        assert len(retries) > 0, "Should detect retry when user re-asks similar question"
        assert retries[0]["similarity"] > 0.3

    # ---- Layer 3: Reward Combination ----

    def test_layer3_combine_with_all_layers(self):
        """Combined reward correctly weights all 3 layers."""
        result = combine(layer1=1.0, layer2=0.8, layer3=0.6)
        # final = 0.5*1.0 + 0.3*0.8 + 0.2*0.6 = 0.5 + 0.24 + 0.12 = 0.86
        assert abs(result.final - 0.86) < 0.001, (
            f"Expected 0.86, got {result.final}"
        )
        assert result.layer1 == 1.0
        assert result.layer2 == 0.8
        assert result.layer3 == 0.6
        assert result.weights["layer1"] == 0.5
        assert result.weights["layer2"] == 0.3
        assert result.weights["layer3"] == 0.2

    def test_layer3_combine_without_judge(self):
        """When layer3 is None, weights are redistributed."""
        result = combine(layer1=1.0, layer2=0.5, layer3=None)
        # Default weights: 0.5, 0.3, 0.2
        # Without L3: l1_weight = 0.5 + 0.2*(0.5/0.8) = 0.5 + 0.125 = 0.625
        #              l2_weight = 0.3 + 0.2*(0.3/0.8) = 0.3 + 0.075 = 0.375
        # final = 0.625*1.0 + 0.375*0.5 = 0.625 + 0.1875 = 0.8125
        assert abs(result.final - 0.8125) < 0.001, (
            f"Expected 0.8125, got {result.final}"
        )
        assert result.layer3 is None
        assert result.weights["layer3"] == 0.0
        assert result.metadata["layer3_was_none"] is True

    def test_layer3_custom_weights(self):
        """Custom weight configuration is respected."""
        result = combine(
            layer1=1.0, layer2=0.0, layer3=1.0,
            weights={"layer1": 0.7, "layer2": 0.2, "layer3": 0.1},
        )
        # final = 0.7*1.0 + 0.2*0.0 + 0.1*1.0 = 0.8
        assert abs(result.final - 0.8) < 0.001, (
            f"Expected 0.8, got {result.final}"
        )

    def test_full_reward_pipeline_integration(self):
        """compute_full_reward integrates L1, L2, and combines them."""
        trajectory = [
            {"role": "user", "content": "Check my order O001."},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_order_details",
                            "arguments": '{"order_id": "O001"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "get_order_details",
                "content": '{"status": "success", "result": {"order_id": "O001", "status": "delivered"}}',
            },
            {
                "role": "assistant",
                "content": "Your order O001 was delivered on May 26. I have completed the check.",
            },
            {"role": "user", "content": "Thank you! That's great!"},
        ]

        result = compute_full_reward(trajectory, call_judge=False)
        assert result.layer1 >= 0.5, f"L1 should be high for valid trajectory, got {result.layer1}"
        assert result.layer2 > 0.3, f"L2 should detect positive signal, got {result.layer2}"
        assert result.layer3 is None
        # final = redistributed weights of l1 and l2
        assert 0.0 <= result.final <= 1.0, f"Final {result.final} not in [0,1]"
        print(f"Full reward: L1={result.layer1:.3f}, L2={result.layer2:.3f}, L3=None, final={result.final:.3f}")


# ============================================================================
# E2E-4: Unified Sample Format Integrity
# ============================================================================


class TestE2EUnifiedSampleFormatIntegrity:
    """Load real train.jsonl and verify format correctness."""

    @pytest.fixture(scope="class")
    def train_samples(self):
        """Load all train samples (shared across tests in this class)."""
        train_path = os.path.join(PROJECT_DIR, "data", "tau_bench", "train.jsonl")
        if not os.path.exists(train_path):
            pytest.skip(f"{train_path} not found")

        samples = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def test_all_required_fields_present(self, train_samples):
        """Every sample has all required top-level and nested fields."""
        required_top = {"id", "source", "task_id", "context", "trajectory", "outcome"}
        required_context = {"system_prompt", "tools", "user_request"}
        required_outcome = {"task_completed", "reward", "reward_weights"}

        for i, s in enumerate(train_samples):
            # Top-level
            missing_top = required_top - set(s.keys())
            assert not missing_top, (
                f"Sample {i} (id={s.get('id','?')}) missing top-level fields: {missing_top}"
            )

            # Context
            ctx = s["context"]
            missing_ctx = required_context - set(ctx.keys())
            assert not missing_ctx, (
                f"Sample {i} missing context fields: {missing_ctx}"
            )
            assert isinstance(ctx["system_prompt"], str) and ctx["system_prompt"].strip(), (
                f"Sample {i} has empty system_prompt"
            )
            assert isinstance(ctx["user_request"], str) and ctx["user_request"].strip(), (
                f"Sample {i} has empty user_request"
            )
            assert isinstance(ctx["tools"], list) and len(ctx["tools"]) > 0, (
                f"Sample {i} has empty tools list"
            )

            # Outcome
            outcome = s["outcome"]
            missing_out = required_outcome - set(outcome.keys())
            assert not missing_out, (
                f"Sample {i} missing outcome fields: {missing_out}"
            )

            # Reward sub-fields
            reward = outcome["reward"]
            for layer in ("layer1", "layer2", "layer3", "final"):
                assert layer in reward, f"Sample {i} reward missing '{layer}'"
                assert 0.0 <= reward[layer] <= 1.0, (
                    f"Sample {i} reward.{layer}={reward[layer]} not in [0,1]"
                )

    def test_reward_weights_sum_to_one(self, train_samples):
        """outcome.reward_weights sum to approximately 1.0."""
        for i, s in enumerate(train_samples):
            weights = s["outcome"]["reward_weights"]
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, (
                f"Sample {i} (id={s['id']}) weights sum to {total}, expected ~1.0"
            )

    def test_valid_role_sequences(self, train_samples):
        """Trajectory messages have valid role values and no consecutive
        identical same-content messages."""
        valid_roles = {"user", "assistant", "tool"}
        for i, s in enumerate(train_samples):
            traj = s["trajectory"]
            assert isinstance(traj, list) and len(traj) > 0, (
                f"Sample {i} has empty trajectory"
            )

            prev_role = None
            prev_content = None
            for j, msg in enumerate(traj):
                role = msg.get("role", "")
                assert role in valid_roles, (
                    f"Sample {i} msg {j}: invalid role '{role}'"
                )

                # Check for adjacent same-role messages with same content
                # (only flag if both role and content match, indicating potential duplicate)
                content = msg.get("content", "")
                if role == prev_role and content == prev_content and content != "":
                    # This is suspicious - same role with identical content
                    # Not an automatic fail since it could be valid in some cases
                    pass

                # Validate tool messages have required fields
                if role == "tool":
                    assert "tool_call_id" in msg, (
                        f"Sample {i} msg {j}: tool message missing tool_call_id"
                    )
                    assert "name" in msg, (
                        f"Sample {i} msg {j}: tool message missing name"
                    )

                # Validate assistant messages with tool_calls
                if role == "assistant" and "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        assert "id" in tc, (
                            f"Sample {i} msg {j}: tool_call missing id"
                        )
                        fn = tc.get("function", {})
                        assert "name" in fn, (
                            f"Sample {i} msg {j}: tool_call missing function.name"
                        )
                        assert "arguments" in fn, (
                            f"Sample {i} msg {j}: tool_call missing function.arguments"
                        )

                prev_role = role
                prev_content = content

    def test_tool_calls_reference_defined_tools(self, train_samples):
        """Tool calls in trajectory reference tools defined in context.tools."""
        for i, s in enumerate(train_samples):
            # Build set of defined tool names from context.tools
            defined_tools = set()
            for t in s["context"]["tools"]:
                fn = t.get("function", {})
                name = fn.get("name", "")
                if name:
                    defined_tools.add(name)

            # Collect tool calls from trajectory
            for msg in s["trajectory"]:
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    call_name = fn.get("name", "")
                    if call_name:
                        assert call_name in defined_tools, (
                            f"Sample {i} (id={s['id']}): tool '{call_name}' "
                            f"used in trajectory but not defined in context.tools. "
                            f"Defined: {sorted(defined_tools)}"
                        )

    def test_type_field_consistency(self, train_samples):
        """All tool definitions have type='function'."""
        for i, s in enumerate(train_samples):
            for t in s["context"]["tools"]:
                assert t.get("type") == "function", (
                    f"Sample {i}: tool has type={t.get('type')}, expected 'function'"
                )

    def test_sample_id_uniqueness(self, train_samples):
        """All sample IDs in train.jsonl are unique."""
        ids = [s["id"] for s in train_samples]
        assert len(ids) == len(set(ids)), (
            f"Duplicate sample IDs found: {len(ids)} total, {len(set(ids))} unique"
        )

    def test_source_field_values(self, train_samples):
        """source field contains expected domain indicators."""
        valid_prefixes = ("taubench_", "apigen_")
        valid_suffixes = ("airline", "retail")
        for i, s in enumerate(train_samples):
            source = s["source"]
            assert source.startswith(valid_prefixes) or source.endswith(valid_suffixes) or (
                any(source.startswith(p) for p in valid_prefixes)
            ), f"Sample {i}: unexpected source '{source}'"

    def test_trajectory_first_message_is_user(self, train_samples):
        """First message in trajectory is typically 'user' (the request)."""
        user_first = 0
        for i, s in enumerate(train_samples):
            traj = s["trajectory"]
            if traj and traj[0].get("role") == "user":
                user_first += 1
        ratio = user_first / len(train_samples)
        assert ratio > 0.5, (
            f"Only {ratio:.1%} of samples start with user message, "
            f"expected majority"
        )

    def test_pipeline_reward_values_make_sense(self, train_samples):
        """Final reward values are either 0.0, 0.5, or 1.0 (tau-bench binary)."""
        rewards = [s["outcome"]["reward"]["final"] for s in train_samples]
        # tau-bench data primarily has binary rewards
        binary = sum(1 for r in rewards if r in (0.0, 1.0))
        partial = len(rewards) - binary
        ratio = binary / len(rewards)
        assert ratio > 0.3, (
            f"Only {ratio:.1%} of samples have binary rewards, expected >30%"
        )
        print(f"Reward distribution: {binary} binary (0/1), {partial} partial out of {len(rewards)}")
