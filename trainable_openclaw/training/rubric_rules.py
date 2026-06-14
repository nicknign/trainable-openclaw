"""
Zero-cost rubric rule engine for trajectory scoring.

23 deterministic rules across 6 groups. Each rule checks one aspect of
agent behavior and returns a 0-10 score. Final reward is the weighted
average of group scores.

Rules are retail-specific, aligned with DeepSeek-generated rubrics in
data/rubrics_retail.json. No API calls — purely rule-based.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Group weights
# ---------------------------------------------------------------------------

GROUP_WEIGHTS = {
    "tool_selection": 0.20,
    "info_sufficiency": 0.20,
    "step_efficiency": 0.15,
    "error_recovery": 0.15,
    "task_completion": 0.20,
    "communication": 0.10,
}

# ---------------------------------------------------------------------------
# Tau-bench retail tool name sets (for precise matching)
# ---------------------------------------------------------------------------

IDENTITY_TOOLS = r"find_user_id_by_name_zip|find_user_id_by_email"
LOOKUP_TOOLS = r"find_user_id_by_name_zip|find_user_id_by_email|get_order_details|get_user_details|get_product_details|get_item_details"
MODIFY_TOOLS = r"modify_pending_order_items|modify_pending_order_address|modify_pending_order_payment"
CANCEL_TOOLS = r"cancel_pending_order"
RETURN_TOOLS = r"return_delivered_order_items"
EXCHANGE_TOOLS = r"exchange_delivered_order_items"
ACTION_TOOLS = r"modify_pending_order|cancel_pending_order|return_delivered_order|exchange_delivered_order"
IRREVERSIBLE_TOOLS = r"cancel_pending_order|return_delivered_order_items"
ORDER_QUERY_TOOLS = r"get_order_details"
PRODUCT_QUERY_TOOLS = r"get_product_details|get_item_details"
THINK_TOOL = r"^think$"
TRANSFER_TOOL = r"transfer_to_human_agents"

# ---------------------------------------------------------------------------
# Utility: tool call extraction from trajectory
# ---------------------------------------------------------------------------


def _extract_tool_calls(conversation: list[dict]) -> list[dict]:
    """Extract tool call dicts from conversation messages."""
    calls: list[dict] = []
    for msg in conversation:
        if msg.get("role") == "assistant":
            from trainable_openclaw.training.rollout_env import parse_tool_calls_from_text
            parsed = parse_tool_calls_from_text(msg.get("content", ""))
            calls.extend(parsed)
    return calls


def _extract_tool_results(conversation: list[dict]) -> list[dict]:
    """Extract tool execution results from conversation."""
    results: list[dict] = []
    for msg in conversation:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str):
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        results.extend(data)
                    elif isinstance(data, dict):
                        results.append(data)
                except json.JSONDecodeError:
                    pass
    return results


def _get_agent_messages(conversation: list[dict]) -> list[str]:
    """Extract agent text messages."""
    return [
        msg.get("content", "") for msg in conversation
        if msg.get("role") == "assistant"
    ]


def _get_user_messages(conversation: list[dict]) -> list[str]:
    """Extract user text messages."""
    return [
        msg.get("content", "") for msg in conversation
        if msg.get("role") == "user"
    ]


def _has_tool_call(calls: list[dict], name_pattern: str) -> bool:
    """Check if any tool call matches a name pattern (case-insensitive)."""
    for c in calls:
        name = c.get("name", c.get("function", {}).get("name", ""))
        if re.search(name_pattern, name, re.IGNORECASE):
            return True
    return False


def _count_tool_call(calls: list[dict], name_pattern: str) -> int:
    """Count tool calls matching a name pattern."""
    count = 0
    for c in calls:
        name = c.get("name", c.get("function", {}).get("name", ""))
        if re.search(name_pattern, name, re.IGNORECASE):
            count += 1
    return count


def _tool_call_names(calls: list[dict]) -> list[str]:
    """Return ordered list of tool call names (excluding think)."""
    names = []
    for c in calls:
        name = c.get("name", c.get("function", {}).get("name", ""))
        if not re.search(THINK_TOOL, name, re.IGNORECASE):
            names.append(name)
    return names


def _tool_call_names_with_think(calls: list[dict]) -> list[str]:
    """Return ordered list of all tool call names including think."""
    return [c.get("name", c.get("function", {}).get("name", "")) for c in calls]


def _tool_errors(results: list[dict]) -> list[dict]:
    """Return tool results that indicate errors."""
    errors: list[dict] = []
    for r in results:
        result = r.get("result", r)
        if isinstance(result, dict) and result.get("status") == "error":
            errors.append(r)
    return errors


# ---------------------------------------------------------------------------
# RubricRuleEngine
# ---------------------------------------------------------------------------


class RubricRuleEngine:
    """Deterministic trajectory scorer with 23 rules across 6 groups.

    Each rule is a method named _rule_<name> that returns a deduction value.
    Score starts at 10 per group; deductions bring it down. Final score
    clamped to [0, 10].
    """

    def __init__(self, rules_spec: list[dict] | None = None):
        self._rules_spec = rules_spec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, trajectory: dict | Any) -> dict[str, Any]:
        """Score a trajectory, returning per-group scores and deductions.

        Args:
            trajectory: Either a dict with 'conversations' key, or a Trajectory object.

        Returns:
            Dict with keys: group_scores, deductions, total, weighted_reward.
        """
        conversation = self._get_conversation(trajectory)

        tool_calls = _extract_tool_calls(conversation)
        tool_results = _extract_tool_results(conversation)
        agent_msgs = _get_agent_messages(conversation)
        user_msgs = _get_user_messages(conversation)
        all_text = " ".join(agent_msgs)
        errors = _tool_errors(tool_results)

        deductions: list[dict] = []

        group_scores = {
            "tool_selection": self._score_tool_selection(tool_calls, agent_msgs, deductions),
            "info_sufficiency": self._score_info_sufficiency(tool_calls, agent_msgs, deductions),
            "step_efficiency": self._score_step_efficiency(tool_calls, agent_msgs, conversation, deductions),
            "error_recovery": self._score_error_recovery(errors, tool_calls, agent_msgs, deductions),
            "task_completion": self._score_task_completion(tool_calls, tool_results, agent_msgs, user_msgs, deductions),
            "communication": self._score_communication(agent_msgs, all_text, deductions),
        }

        total = sum(
            group_scores[g] * GROUP_WEIGHTS[g]
            for g in group_scores
        )

        completion_status = trajectory.get("status", "in_progress") if isinstance(trajectory, dict) else getattr(trajectory, "status", "in_progress")

        return {
            "group_scores": group_scores,
            "deductions": deductions,
            "total": round(total, 3),
            "weighted_reward": round(total / 10.0, 4),
            "status": completion_status,
        }

    def compute_reward(self, trajectory: dict | Any) -> float:
        """Return a single reward value (0-1) for the trajectory."""
        result = self.score(trajectory)
        return result["weighted_reward"]

    # ==================================================================
    # Group: Tool Selection (4 rules)
    # ==================================================================

    def _score_tool_selection(self, calls: list[dict], agent_msgs: list[str], deductions: list) -> float:
        score = 10.0
        score -= self._rule_user_identity_first(calls, deductions)
        score -= self._rule_correct_tool_for_state(calls, deductions)
        score -= self._rule_payment_method_check(calls, deductions)
        score -= self._rule_tool_arguments(calls, deductions)
        return max(0.0, min(10.0, score))

    def _rule_user_identity_first(self, calls: list[dict], deductions: list) -> float:
        """User identity lookup (find_user_id_by_*) must happen before any order operation.

        DeepSeek: "Start at 10. If the agent calls any tool other than
        find_user_id_by_name_zip or find_user_id_by_email before identifying
        the user, deduct 5. If the agent never calls find_user_id_by_name_zip
        or find_user_id_by_email, deduct 10."
        """
        if not calls:
            deductions.append({"rule": "user_identity_first", "deduction": 10,
                               "reason": "No tools called at all"})
            return 10.0

        names = _tool_call_names(calls)
        has_identity = _has_tool_call(calls, IDENTITY_TOOLS)
        has_action = _has_tool_call(calls, ACTION_TOOLS)

        if not has_identity:
            deductions.append({"rule": "user_identity_first", "deduction": 10,
                               "reason": "Never called find_user_id_by_name_zip or find_user_id_by_email"})
            return 10.0

        # Check if the first non-think tool call is an identity lookup
        first_real = names[0] if names else ""
        if not re.search(IDENTITY_TOOLS, first_real, re.IGNORECASE) and has_action:
            deductions.append({"rule": "user_identity_first", "deduction": 5,
                               "reason": f"First tool was {first_real}, not identity lookup"})
            return 5.0

        return 0.0

    def _rule_correct_tool_for_state(self, calls: list[dict], deductions: list) -> float:
        """Correct tool for order state: pending vs delivered.

        DeepSeek: "Start at 10. If the agent calls cancel_pending_order on a
        delivered order, deduct 5. If the agent calls return_delivered_order_items
        on a pending order, deduct 5. If the agent calls modify_pending_order_items
        on a delivered order, deduct 5."

        We approximate order state from tool call sequence: if exchange/return
        tools are used, the order is likely delivered. If modify/cancel tools
        are used, the order is likely pending. Cross-contamination suggests
        wrong tool for state.
        """
        has_pending_ops = _has_tool_call(calls, MODIFY_TOOLS + "|" + CANCEL_TOOLS)
        has_delivered_ops = _has_tool_call(calls, RETURN_TOOLS + "|" + EXCHANGE_TOOLS)

        # If agent uses BOTH pending and delivered tools on what should be
        # a single order, that's a state confusion error
        if has_pending_ops and has_delivered_ops:
            deductions.append({"rule": "correct_tool_for_state", "deduction": 5,
                               "reason": "Mixed pending and delivered order tools — possible state confusion"})
            return 5.0

        return 0.0

    def _rule_payment_method_check(self, calls: list[dict], deductions: list) -> float:
        """Payment method verified before payment modification.

        DeepSeek: "Start at 10. If the agent calls modify_pending_order_payment
        without first calling get_user_details to check payment methods, deduct 5."
        """
        has_payment_mod = _has_tool_call(calls, r"modify_pending_order_payment")
        has_user_details = _has_tool_call(calls, r"get_user_details")

        if has_payment_mod and not has_user_details:
            deductions.append({"rule": "payment_method_check", "deduction": 5,
                               "reason": "Modified payment without checking user payment methods"})
            return 5.0
        return 0.0

    def _rule_tool_arguments(self, calls: list[dict], deductions: list) -> float:
        """Check for severely malformed tool arguments."""
        penalty = 0.0
        for c in calls:
            args = c.get("arguments", c.get("function", {}).get("arguments", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    penalty += 2.0
                    continue
            if args is None or (isinstance(args, dict) and len(args) == 0):
                penalty += 1.0
        if penalty > 0:
            deductions.append({"rule": "tool_arguments", "deduction": min(5.0, penalty),
                               "reason": "Malformed tool arguments"})
        return min(5.0, penalty)

    # ==================================================================
    # Group: Information Sufficiency (4 rules)
    # ==================================================================

    def _score_info_sufficiency(self, calls: list[dict], agent_msgs: list[str], deductions: list) -> float:
        score = 10.0
        score -= self._rule_gather_user_before_address(calls, deductions)
        score -= self._rule_order_details_before_modify(calls, deductions)
        score -= self._rule_product_before_exchange(calls, deductions)
        score -= self._rule_order_lookup_by_user(calls, deductions)
        return max(0.0, min(10.0, score))

    def _rule_gather_user_before_address(self, calls: list[dict], deductions: list) -> float:
        """Gather user details before address changes.

        DeepSeek: "Start at 10. If the agent calls modify_pending_order_address
        or modify_user_address without first calling get_user_details, deduct 5.
        If without first calling find_user_id_by_name_zip or find_user_id_by_email,
        deduct 3."
        """
        has_address_mod = _has_tool_call(calls, r"modify_pending_order_address|modify_user_address")
        if not has_address_mod:
            return 0.0

        has_user_details = _has_tool_call(calls, r"get_user_details")
        has_identity = _has_tool_call(calls, IDENTITY_TOOLS)

        if not has_user_details:
            deductions.append({"rule": "gather_user_before_address", "deduction": 5,
                               "reason": "Address modification without get_user_details"})
            return 5.0
        if not has_identity:
            deductions.append({"rule": "gather_user_before_address", "deduction": 3,
                               "reason": "Address modification without user identity lookup"})
            return 3.0
        return 0.0

    def _rule_order_details_before_modify(self, calls: list[dict], deductions: list) -> float:
        """Query order details before modifying.

        DeepSeek: "Start at 10. If the agent calls modify_pending_order_items,
        modify_pending_order_address, or modify_pending_order_payment without
        first calling get_order_details, deduct 5. If the agent calls
        cancel_pending_order without first calling get_order_details, deduct 3."
        """
        has_modify = _has_tool_call(calls, MODIFY_TOOLS)
        has_cancel = _has_tool_call(calls, CANCEL_TOOLS)
        has_order = _has_tool_call(calls, ORDER_QUERY_TOOLS)

        if has_modify and not has_order:
            deductions.append({"rule": "order_details_before_modify", "deduction": 5,
                               "reason": "Order modification without get_order_details"})
            return 5.0
        if has_cancel and not has_order:
            deductions.append({"rule": "order_details_before_modify", "deduction": 3,
                               "reason": "Order cancellation without get_order_details"})
            return 3.0
        return 0.0

    def _rule_product_before_exchange(self, calls: list[dict], deductions: list) -> float:
        """Query product details before exchange.

        DeepSeek: "Start at 10. If the agent calls exchange_delivered_order_items
        without first calling get_product_details or get_item_details, deduct 5.
        If without first calling get_order_details, deduct 3."
        """
        has_exchange = _has_tool_call(calls, EXCHANGE_TOOLS)
        if not has_exchange:
            return 0.0

        has_product = _has_tool_call(calls, PRODUCT_QUERY_TOOLS)
        has_order = _has_tool_call(calls, ORDER_QUERY_TOOLS)

        if not has_product:
            deductions.append({"rule": "product_before_exchange", "deduction": 5,
                               "reason": "Exchange without get_product_details or get_item_details"})
            return 5.0
        if not has_order:
            deductions.append({"rule": "product_before_exchange", "deduction": 3,
                               "reason": "Exchange without get_order_details"})
            return 3.0
        return 0.0

    def _rule_order_lookup_by_user(self, calls: list[dict], deductions: list) -> float:
        """Order lookup must follow user identity verification.

        DeepSeek: "Start at 10. If the agent calls get_order_details without
        first identifying the user via find_user_id_by_name_zip or
        find_user_id_by_email, deduct 5."
        """
        has_order = _has_tool_call(calls, ORDER_QUERY_TOOLS)
        if not has_order:
            return 0.0

        has_identity = _has_tool_call(calls, IDENTITY_TOOLS)

        if not has_identity:
            deductions.append({"rule": "order_lookup_by_user", "deduction": 5,
                               "reason": "get_order_details called without user identity lookup"})
            return 5.0

        # Verify identity lookup occurs before first order query
        names = _tool_call_names(calls)
        first_identity = -1
        first_order = -1
        for i, n in enumerate(names):
            if first_identity < 0 and re.search(IDENTITY_TOOLS, n, re.IGNORECASE):
                first_identity = i
            if first_order < 0 and re.search(ORDER_QUERY_TOOLS, n, re.IGNORECASE):
                first_order = i

        if first_order >= 0 and (first_identity < 0 or first_order < first_identity):
            deductions.append({"rule": "order_lookup_by_user", "deduction": 5,
                               "reason": "get_order_details called before user identity lookup"})
            return 5.0
        return 0.0

    # ==================================================================
    # Group: Step Efficiency (4 rules)
    # ==================================================================

    def _score_step_efficiency(self, calls: list[dict], agent_msgs: list[str],
                                conversation: list[dict], deductions: list) -> float:
        score = 10.0
        score -= self._rule_no_duplicate_calls(calls, deductions)
        score -= self._rule_minimal_tool_calls(calls, deductions)
        score -= self._rule_think_usage(calls, deductions)
        score -= self._rule_no_redundant_item_calls(calls, deductions)
        return max(0.0, min(10.0, score))

    def _rule_no_duplicate_calls(self, calls: list[dict], deductions: list) -> float:
        """No redundant duplicate tool+args calls.

        DeepSeek: "Start at 10. For each pair of consecutive calls to the same
        tool with identical arguments (excluding think), deduct 3. If the same
        tool+args appears 3 or more times, deduct 5."
        """
        names = _tool_call_names_with_think(calls)
        # Build list of (name, args) for non-think calls
        entries = []
        for c in calls:
            name = c.get("name", c.get("function", {}).get("name", ""))
            if re.search(THINK_TOOL, name, re.IGNORECASE):
                entries.append(None)  # placeholder for think
            else:
                args = c.get("arguments", c.get("function", {}).get("arguments", {}))
                entries.append((name, json.dumps(args, sort_keys=True)))

        # Count consecutive duplicates (ignoring think gaps)
        last_non_think = None
        consecutive_dup_pairs = 0
        for entry in entries:
            if entry is None:
                continue
            if last_non_think is not None and entry == last_non_think:
                consecutive_dup_pairs += 1
            last_non_think = entry

        # Count total occurrences
        from collections import Counter
        non_think_entries = [e for e in entries if e is not None]
        counts = Counter(non_think_entries)

        penalty = 0.0
        for count in counts.values():
            if count >= 3:
                penalty += 5.0
        penalty += consecutive_dup_pairs * 3.0

        if penalty > 0:
            deductions.append({"rule": "no_duplicate_calls", "deduction": min(10.0, penalty),
                               "reason": f"Redundant tool calls detected"})
        return min(10.0, penalty)

    def _rule_minimal_tool_calls(self, calls: list[dict], deductions: list) -> float:
        """Minimal tool calls per task.

        DeepSeek: "Start at 10. If the agent makes more than 10 tool calls
        total (excluding think), deduct 5. If the agent makes more than 15
        tool calls total (excluding think), deduct 8."
        """
        non_think = len(_tool_call_names(calls))
        if non_think > 15:
            deductions.append({"rule": "minimal_tool_calls", "deduction": 8,
                               "reason": f"{non_think} tool calls (>15)"})
            return 8.0
        if non_think > 10:
            deductions.append({"rule": "minimal_tool_calls", "deduction": 5,
                               "reason": f"{non_think} tool calls (>10)"})
            return 5.0
        return 0.0

    def _rule_think_usage(self, calls: list[dict], deductions: list) -> float:
        """Use think tool for reasoning between tool calls.

        DeepSeek: "Start at 10. If the agent makes 3 or more consecutive tool
        calls without any think call in between, deduct 3. If the agent never
        calls think, deduct 2."
        """
        all_names = _tool_call_names_with_think(calls)
        has_think = any(re.search(THINK_TOOL, n, re.IGNORECASE) for n in all_names)

        if not has_think and len(_tool_call_names(calls)) >= 3:
            deductions.append({"rule": "think_usage", "deduction": 2,
                               "reason": "Never used think tool"})
            return 2.0

        # Count max consecutive non-think calls
        max_consecutive = 0
        current = 0
        for n in all_names:
            if re.search(THINK_TOOL, n, re.IGNORECASE):
                max_consecutive = max(max_consecutive, current)
                current = 0
            else:
                current += 1
        max_consecutive = max(max_consecutive, current)

        if max_consecutive >= 3:
            deductions.append({"rule": "think_usage", "deduction": 3,
                               "reason": f"{max_consecutive} consecutive tool calls without think"})
            return 3.0
        return 0.0

    def _rule_no_redundant_item_calls(self, calls: list[dict], deductions: list) -> float:
        """No unnecessary get_item_details calls.

        DeepSeek: "Start at 10. If the agent calls get_item_details for an item
        already retrieved via get_order_details or get_product_details, deduct 2
        per redundant call."

        Approximate: count get_item_details calls beyond the first for each
        unique item_id argument.
        """
        item_calls = []
        for c in calls:
            name = c.get("name", c.get("function", {}).get("name", ""))
            if re.search(r"get_item_details", name, re.IGNORECASE):
                args = c.get("arguments", c.get("function", {}).get("arguments", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                item_id = args.get("item_id", "") if isinstance(args, dict) else str(args)
                item_calls.append(item_id)

        from collections import Counter
        item_counts = Counter(item_calls)
        redundant = sum(max(0, c - 1) for c in item_counts.values())

        if redundant:
            deductions.append({"rule": "no_redundant_item_calls", "deduction": min(6, redundant * 2),
                               "reason": f"{redundant} redundant get_item_details call(s)"})
        return float(min(6, redundant * 2))

    # ==================================================================
    # Group: Error Recovery (3 rules)
    # ==================================================================

    def _score_error_recovery(self, errors: list[dict], calls: list[dict],
                               agent_msgs: list[str], deductions: list) -> float:
        score = 10.0
        if errors:
            score -= self._rule_retry_with_correction(errors, calls, deductions)
        score -= self._rule_no_premature_transfer(calls, deductions)
        score -= self._rule_give_up_handling(agent_msgs, deductions)
        return max(0.0, min(10.0, score))

    def _rule_retry_with_correction(self, errors: list[dict], calls: list[dict], deductions: list) -> float:
        """Retry on tool error with corrected arguments.

        DeepSeek: "Start at 10. If a tool returns an error and the agent calls
        the same tool with identical arguments again, deduct 5. If a tool returns
        an error and the agent calls transfer_to_human_agents immediately, deduct 3.
        If a tool returns an error and the agent never retries with corrected
        arguments, deduct 5."

        Consolidated from two old rules (tool_error_retry + retry_quality).
        """
        error_names = [e.get("name", "") for e in errors]

        # Check if any error was followed by immediate transfer
        for i, c in enumerate(calls):
            name = c.get("name", c.get("function", {}).get("name", ""))
            if name in error_names and i + 1 < len(calls):
                next_c = calls[i + 1]
                next_name = next_c.get("name", next_c.get("function", {}).get("name", ""))
                if re.search(TRANSFER_TOOL, next_name, re.IGNORECASE):
                    deductions.append({"rule": "retry_with_correction", "deduction": 3,
                                       "reason": "Transferred to human immediately after tool error"})
                    return 3.0

        # Check for retry with same args
        for i, c in enumerate(calls):
            name = c.get("name", c.get("function", {}).get("name", ""))
            if name in error_names and i + 1 < len(calls):
                next_c = calls[i + 1]
                next_name = next_c.get("name", next_c.get("function", {}).get("name", ""))
                if next_name == name:
                    args1 = json.dumps(c.get("arguments", c.get("function", {}).get("arguments", {})), sort_keys=True)
                    args2 = json.dumps(next_c.get("arguments", next_c.get("function", {}).get("arguments", {})), sort_keys=True)
                    if args1 == args2:
                        deductions.append({"rule": "retry_with_correction", "deduction": 5,
                                           "reason": f"Retried {name} with identical arguments"})
                        return 5.0

        # Check for no retry at all
        retried = set()
        for c in calls:
            name = c.get("name", c.get("function", {}).get("name", ""))
            if name in error_names:
                retried.add(name)
        for ename in error_names:
            if ename not in retried:
                deductions.append({"rule": "retry_with_correction", "deduction": 5,
                                   "reason": f"Never retried after error on {ename}"})
                return 5.0

        return 0.0

    def _rule_no_premature_transfer(self, calls: list[dict], deductions: list) -> float:
        """No premature transfer to human agent.

        DeepSeek: "Start at 10. If the agent calls transfer_to_human_agents
        before making at least 3 tool calls (excluding think), deduct 5. If
        the agent calls transfer_to_human_agents after a single tool error
        without retrying, deduct 3."
        """
        non_think = _tool_call_names(calls)
        has_transfer = any(re.search(TRANSFER_TOOL, n, re.IGNORECASE) for n in non_think)

        if not has_transfer:
            return 0.0

        # Find position of first transfer
        transfer_idx = -1
        for i, n in enumerate(non_think):
            if re.search(TRANSFER_TOOL, n, re.IGNORECASE):
                transfer_idx = i
                break

        if transfer_idx >= 0 and transfer_idx < 3:
            deductions.append({"rule": "no_premature_transfer", "deduction": 5,
                               "reason": f"Transfer to human after only {transfer_idx} tool calls"})
            return 5.0
        return 0.0

    def _rule_give_up_handling(self, agent_msgs: list[str], deductions: list) -> float:
        """Agent gave up without trying alternatives."""
        if not agent_msgs:
            return 5.0
        last = agent_msgs[-1].lower()
        gave_up = bool(re.search(
            r"(?:can'?t help|cannot help|unable to|not possible|transfer|human agent)",
            last, re.IGNORECASE
        ))
        if gave_up and len(agent_msgs) < 3:
            deductions.append({"rule": "give_up_handling", "deduction": 5,
                               "reason": "Agent gave up too quickly"})
            return 5.0
        return 0.0

    # ==================================================================
    # Group: Task Completion (4 rules)
    # ==================================================================

    def _score_task_completion(self, calls: list[dict], results: list[dict],
                                agent_msgs: list[str], user_msgs: list[str],
                                deductions: list) -> float:
        score = 10.0
        score -= self._rule_main_task_done(calls, results, deductions)
        score -= self._rule_subtasks_covered(calls, results, deductions)
        score -= self._rule_verify_after_modify(calls, deductions)
        score -= self._rule_user_satisfied(user_msgs, deductions)
        return max(0.0, min(10.0, score))

    def _rule_main_task_done(self, calls: list[dict], results: list[dict], deductions: list) -> float:
        """Primary user request fulfilled.

        transfer_to_human_agents does NOT count as fulfilling the request.
        """
        has_action = _has_tool_call(calls, ACTION_TOOLS)
        has_success = any(
            r.get("status") == "success"
            and not re.search(TRANSFER_TOOL, r.get("name", ""), re.IGNORECASE)
            for r in results
        )
        if not has_action and not has_success:
            deductions.append({"rule": "main_task_done", "deduction": 8,
                               "reason": "No action taken to fulfill request"})
            return 8.0
        if has_action and not has_success:
            deductions.append({"rule": "main_task_done", "deduction": 5,
                               "reason": "Actions taken but no success"})
            return 5.0
        return 0.0

    def _rule_subtasks_covered(self, calls: list[dict], results: list[dict], deductions: list) -> float:
        """Multiple user requests all addressed.

        DeepSeek: "Start at 10. If the user makes multiple requests in their
        first message and the agent addresses fewer than all of them before
        the final response, deduct 3 per unaddressed request."

        Approximate: count successful action results vs distinct action tool calls.
        """
        success_count = sum(
            1 for r in results
            if r.get("status") == "success"
        )
        total_actions = _count_tool_call(calls, ACTION_TOOLS)
        if total_actions > 1 and success_count < total_actions:
            missed = total_actions - success_count
            deductions.append({"rule": "subtasks_covered", "deduction": min(9, missed * 3),
                               "reason": f"{missed} subtask(s) not completed"})
            return float(min(9, missed * 3))
        return 0.0

    def _rule_verify_after_modify(self, calls: list[dict], deductions: list) -> float:
        """Verify modifications after applying.

        DeepSeek: "Start at 10. If the agent calls modify_pending_order_items,
        modify_pending_order_address, modify_pending_order_payment,
        cancel_pending_order, or exchange_delivered_order_items and does not
        call get_order_details on the same order afterward, deduct 5."
        """
        has_mod_action = _has_tool_call(calls, MODIFY_TOOLS + "|" + CANCEL_TOOLS + "|" + EXCHANGE_TOOLS)
        if not has_mod_action:
            return 0.0

        names = _tool_call_names(calls)
        last_action_idx = -1
        for i in range(len(names) - 1, -1, -1):
            if re.search(MODIFY_TOOLS + "|" + CANCEL_TOOLS + "|" + EXCHANGE_TOOLS, names[i], re.IGNORECASE):
                last_action_idx = i
                break

        has_order_after = False
        for i in range(last_action_idx + 1, len(names)):
            if re.search(ORDER_QUERY_TOOLS, names[i], re.IGNORECASE):
                has_order_after = True
                break

        if not has_order_after:
            deductions.append({"rule": "verify_after_modify", "deduction": 5,
                               "reason": "Modification not verified with follow-up get_order_details"})
            return 5.0
        return 0.0

    def _rule_user_satisfied(self, user_msgs: list[str], deductions: list) -> float:
        """User signaled satisfaction."""
        if not user_msgs:
            return 2.0
        last_user = user_msgs[-1].lower()
        satisfied = bool(re.search(
            r"(?:thank|thanks|perfect|great|exactly|appreciate|wonderful|awesome)",
            last_user, re.IGNORECASE
        ))
        if not satisfied:
            deductions.append({"rule": "user_satisfied", "deduction": 2,
                               "reason": "No satisfaction signal from user"})
            return 2.0
        return 0.0

    # ==================================================================
    # Group: Communication Quality (4 rules)
    # ==================================================================

    def _score_communication(self, agent_msgs: list[str], all_text: str, deductions: list) -> float:
        score = 10.0
        score -= self._rule_include_order_ids(all_text, deductions)
        score -= self._rule_include_product_details(all_text, deductions)
        score -= self._rule_report_price_refund(all_text, deductions)
        score -= self._rule_confirm_changes(agent_msgs, deductions)
        return max(0.0, min(10.0, score))

    def _rule_include_order_ids(self, all_text: str, deductions: list) -> float:
        """Include order IDs in responses.

        DeepSeek: "Start at 10. If the agent's final response to the user does
        not contain any order ID (pattern O[0-9]{3} or #[A-Z0-9]+), deduct 5."
        """
        has_order_id = bool(re.search(r"O\d{3}|#\w+|order\s*(?:ID|id|number|#)?\s*[:\s]*\w+\d", all_text, re.IGNORECASE))
        if not has_order_id:
            deductions.append({"rule": "include_order_ids", "deduction": 5,
                               "reason": "Response contains no order IDs"})
            return 5.0
        return 0.0

    def _rule_include_product_details(self, all_text: str, deductions: list) -> float:
        """Include specific product details in responses.

        DeepSeek: "Start at 10. If the agent mentions a product but does not
        include its name or item number in the same message, deduct 3 per
        missing detail. If the agent's final response contains no product
        names or item numbers, deduct 5."
        """
        has_product_ref = bool(re.search(
            r"I\d{3}|P\d{3}|item\s*(?:ID|id|number|#)?\s*[:\s]*\w+\d|product\s*(?:name|ID|code)?\s*[:\s]*\w+",
            all_text, re.IGNORECASE
        ))
        if not has_product_ref:
            deductions.append({"rule": "include_product_details", "deduction": 5,
                               "reason": "Response contains no product names or item numbers"})
            return 5.0
        return 0.0

    def _rule_report_price_refund(self, all_text: str, deductions: list) -> float:
        """Report price or refund amounts when relevant.

        DeepSeek: "Start at 10. If the user asks about price, refund, or payment
        and the agent's response does not include a dollar amount ($[0-9]+),
        deduct 5. If the agent processes a return or exchange but does not tell
        the user the amount, deduct 3."

        Simplified: check if response contains dollar amounts when discussing
        modifications, returns, or exchanges.
        """
        has_financial_context = bool(re.search(
            r"price|refund|payment|charge|cost|amount|dollar|\$|credit|debit",
            all_text, re.IGNORECASE
        ))
        has_dollar_amount = bool(re.search(r"\$\s*\d+(?:\.\d{2})?", all_text))

        if has_financial_context and not has_dollar_amount:
            deductions.append({"rule": "report_price_refund", "deduction": 3,
                               "reason": "Financial context without dollar amounts"})
            return 3.0
        return 0.0

    def _rule_confirm_changes(self, agent_msgs: list[str], deductions: list) -> float:
        """Confirm changes with user before executing.

        DeepSeek: "Start at 10. If the agent applies a modification without
        asking the user to confirm first, deduct 3. If the user explicitly asks
        for confirmation and the agent does not provide it, deduct 5."
        """
        all_text = " ".join(agent_msgs)
        confirmed = bool(re.search(
            r"(?:would you like|shall I|do you want|confirm|are you sure|okay to|proceed|"
            r"I will|let me|go ahead|I'?ll go ahead)",
            all_text, re.IGNORECASE
        ))
        if not confirmed and len(agent_msgs) > 1:
            deductions.append({"rule": "confirm_changes", "deduction": 3,
                               "reason": "Changes applied without user confirmation"})
            return 3.0
        return 0.0

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        words_a = set(re.findall(r"\w+", a.lower()))
        words_b = set(re.findall(r"\w+", b.lower()))
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    @staticmethod
    def _get_conversation(trajectory: dict | Any) -> list[dict]:
        if hasattr(trajectory, "conversations"):
            return trajectory.conversations
        elif isinstance(trajectory, dict):
            return trajectory.get("conversations", trajectory.get("conversation", []))
        return []
