"""Validate rubric rule engine on synthetic test trajectories.

Tests: good vs bad separation, dimension coverage, rule firing coverage.

Note: Full validation on real trajectories requires evaluation output with
actual conversation data (from Linux GPU evaluation runs). This script uses
synthetically constructed conversations to verify engine logic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainable_openclaw.training.rubric_rules import RubricRuleEngine, GROUP_WEIGHTS


def make_good_trajectory() -> list[dict]:
    """A well-executed retail trajectory: lookup → verify → confirm → modify → verify."""
    return [
        {"role": "user", "content": "I need to change the shipping address on my order"},
        {"role": "assistant", "content": '<function_call>{"name": "think", "arguments": {"reasoning": "Need to identify user first"}}</function_call>'},
        {"role": "tool", "content": '{"status":"success"}'},
        {"role": "assistant", "content": '<function_call>{"name": "find_user_id_by_email", "arguments": {"email": "alice@example.com"}}</function_call>'},
        {"role": "tool", "content": '{"status":"success","result":{"user_id":"U001","name":"Alice"}}'},
        {"role": "assistant", "content": '<function_call>{"name": "get_user_details", "arguments": {"user_id": "U001"}}</function_call>'},
        {"role": "tool", "content": '{"status":"success","result":{"orders":[{"order_id":"O123","status":"pending"}]}}'},
        {"role": "assistant", "content": '<function_call>{"name": "get_order_details", "arguments": {"order_id": "O123"}}</function_call>'},
        {"role": "tool", "content": '{"status":"success","result":{"order_id":"O123","status":"pending","address":"123 Main St","items":[{"item_id":"I456","price":29.99}]}}'},
        {"role": "assistant", "content": '<function_call>{"name": "think", "arguments": {"reasoning": "Should confirm before making changes"}}</function_call>'},
        {"role": "tool", "content": '{"status":"success"}'},
        {"role": "assistant", "content": "Your order #O123 (item I456, $29.99) is shipping to 123 Main St. Would you like me to change the address? Please confirm the new address."},
        {"role": "user", "content": "Yes, change it to 456 Oak Ave"},
        {"role": "assistant", "content": '<function_call>{"name": "modify_pending_order_address", "arguments": {"order_id": "O123", "new_address": "456 Oak Ave"}}</function_call>'},
        {"role": "tool", "content": '{"status":"success","result":{"order_id":"O123","address":"456 Oak Ave"}}'},
        {"role": "assistant", "content": '<function_call>{"name": "get_order_details", "arguments": {"order_id": "O123"}}</function_call>'},
        {"role": "tool", "content": '{"status":"success","result":{"order_id":"O123","address":"456 Oak Ave"}}'},
        {"role": "assistant", "content": "Done! Order #O123 now ships to 456 Oak Ave. Is there anything else I can help with?"},
        {"role": "user", "content": "Thank you, that's perfect"},
    ]


def make_bad_trajectory_no_identity() -> list[dict]:
    """Failed: skips user identity, modifies without lookup."""
    return [
        {"role": "user", "content": "Change my shipping address"},
        {"role": "assistant", "content": '<function_call>{"name": "modify_pending_order_address", "arguments": {"order_id": "O999", "new_address": "Nowhere"}}</function_call>'},
        {"role": "tool", "content": '{"status":"error","message":"Order not found"}'},
        {"role": "assistant", "content": "I cannot help with that."},
    ]


def make_bad_trajectory_transfer() -> list[dict]:
    """Failed: gives up and transfers immediately."""
    return [
        {"role": "user", "content": "I want to return my order"},
        {"role": "assistant", "content": '<function_call>{"name": "transfer_to_human_agents", "arguments": {"reason": "Need help with return"}}</function_call>'},
        {"role": "tool", "content": '{"status":"success"}'},
        {"role": "assistant", "content": "Transferring you to a human agent."},
    ]


def make_bad_trajectory_no_verification() -> list[dict]:
    """Failed: modifies without verifying, no order IDs in response."""
    return [
        {"role": "user", "content": "Cancel my recent order"},
        {"role": "assistant", "content": '<function_call>{"name": "cancel_pending_order", "arguments": {"order_id": "O123"}}</function_call>'},
        {"role": "tool", "content": '{"status":"error","message":"Order not found for this user"}'},
        {"role": "assistant", "content": "Could not cancel. Is there anything else?"},
    ]


def make_bad_trajectory_redundant_calls() -> list[dict]:
    """Failed: redundant tool calls, no think, no follow-up."""
    return [
        {"role": "user", "content": "Check my order status"},
        {"role": "assistant", "content": '<function_call>{"name": "find_user_id_by_email", "arguments": {"email": "bob@test.com"}}</function_call>'},
        {"role": "tool", "content": '{"status":"success","result":{"user_id":"U002"}}'},
        {"role": "assistant", "content": '<function_call>{"name": "find_user_id_by_email", "arguments": {"email": "bob@test.com"}}</function_call>'},
        {"role": "tool", "content": '{"status":"success","result":{"user_id":"U002"}}'},
        {"role": "assistant", "content": '<function_call>{"name": "find_user_id_by_email", "arguments": {"email": "bob@test.com"}}</function_call>'},
        {"role": "tool", "content": '{"status":"success","result":{"user_id":"U002"}}'},
        {"role": "assistant", "content": "Found your account."},
    ]


def main():
    rubrics_path = "data/rubrics_retail.json"
    with open(rubrics_path, encoding="utf-8") as f:
        rubric_data = json.load(f)

    rules = rubric_data["rubrics"]
    print(f"Loaded {len(rules)} DeepSeek-generated rubrics")

    engine = RubricRuleEngine()

    # Test trajectories
    tests = {
        "good_address_change": make_good_trajectory(),
        "bad_no_identity": make_bad_trajectory_no_identity(),
        "bad_premature_transfer": make_bad_trajectory_transfer(),
        "bad_no_verification": make_bad_trajectory_no_verification(),
        "bad_redundant_calls": make_bad_trajectory_redundant_calls(),
    }

    print(f"\n{'='*60}")
    print(f"{'Trajectory':<30} {'Score':>8} {'Deductions'}")
    print(f"{'='*60}")

    good_scores = []
    bad_scores = []
    all_results = {}

    for name, conv in tests.items():
        result = engine.score({"conversations": conv, "status": "complete" if name.startswith("good") else "failed"})
        score = result["weighted_reward"]

        if name.startswith("good"):
            good_scores.append(score)
        else:
            bad_scores.append(score)

        deduc_text = ", ".join(d["rule"][:25] for d in result["deductions"][:3])
        if len(result["deductions"]) > 3:
            deduc_text += f" (+{len(result['deductions'])-3} more)"
        print(f"{name:<30} {score:>8.4f}  {deduc_text}")
        all_results[name] = result

    avg_good = sum(good_scores) / len(good_scores) if good_scores else 0
    avg_bad = sum(bad_scores) / len(bad_scores) if bad_scores else 0
    ratio = avg_good / avg_bad if avg_bad > 0 else float("inf")

    print(f"\nResults:")
    print(f"  Good avg: {avg_good:.4f}  Bad avg: {avg_bad:.4f}  Ratio: {ratio:.2f}x")
    separation = min(good_scores) > max(bad_scores) if good_scores and bad_scores else False
    print(f"  Clear separation (all good > all bad): {separation}")

    # Dimension coverage from rubrics file
    print(f"\nDimension distribution:")
    dims = {}
    for r in rules:
        d = r["dimension"]
        dims[d] = dims.get(d, 0) + 1
    for d, c in sorted(dims.items()):
        print(f"  {d}: {c}")

    # Rule firing coverage
    all_rules_fired = set()
    for r in all_results.values():
        for d in r["deductions"]:
            all_rules_fired.add(d["rule"])
    print(f"\nRules fired: {len(all_rules_fired)}/{len(rules)}")

    if avg_good > avg_bad and ratio > 1.3:
        print(f"\nPASS: Rubrics distinguish good from bad (ratio={ratio:.2f}x > 1.3x)")
    else:
        print(f"\nFAIL: Insufficient separation")

    avg_good = sum(good_scores) / len(good_scores) if good_scores else 0
    avg_bad = sum(bad_scores) / len(bad_scores) if bad_scores else 0
    ratio = avg_good / avg_bad if avg_bad > 0 else float("inf")

    print(f"\nResults:")
    print(f"  Good ({len(good_scores)} completed): avg={avg_good:.4f}")
    print(f"  Bad  ({len(bad_scores)} failed):    avg={avg_bad:.4f}")
    print(f"  Ratio: {ratio:.2f}x")
    print(f"  Min good: {min(good_scores):.4f}, Max bad: {max(bad_scores):.4f}")

    separation = min(good_scores) > max(bad_scores) if good_scores and bad_scores else False
    print(f"  Clear separation: {separation}")

    # Group-level analysis
    print(f"\nDimension distribution:")
    dims = {}
    for r in rules:
        d = r["dimension"]
        dims[d] = dims.get(d, 0) + 1
    for d, c in sorted(dims.items()):
        print(f"  {d}: {c}")

    if avg_good > avg_bad and ratio > 1.3:
        print(f"\nPASS: Rubrics distinguish good from bad (ratio={ratio:.2f}x > 1.3x)")
    else:
        print(f"\nFAIL: Insufficient separation")


if __name__ == "__main__":
    main()
