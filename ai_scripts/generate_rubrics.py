"""Generate tau-bench retail rubrics using DeepSeek API.

Phase 0.3 of self-evolution training plan.
Reads training prompts, clusters by complexity, calls DeepSeek to
generate 15-20 deterministic, task-specific scoring rubrics.

Usage:
    python ai_scripts/generate_rubrics.py [--output data/rubrics_retail.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRAIN_PATH = "data/tau_bench/train_prompts_augmented.jsonl"
OUTPUT_PATH = "data/rubrics_retail.json"

TOOLS_LIST = """calculate, cancel_pending_order, exchange_delivered_order_items,
find_user_id_by_email, find_user_id_by_name_zip, get_item_details,
get_order_details, get_product_details, get_user_details,
list_all_product_types, modify_pending_order_address,
modify_pending_order_items, modify_pending_order_payment,
modify_user_address, return_delivered_order_items, think,
transfer_to_human_agents"""

RUBRIC_GENERATION_PROMPT = """You are an expert evaluator for tau-bench retail customer service agents. Your job is to create scoring rubrics for GRPO reinforcement learning.

## Background
We are training a 4B-parameter LLM agent (Qwen3.5-4B) with GRPO. The agent handles retail customer service via tool calls — it receives a customer message, queries databases, modifies orders, and responds. Tasks take 3-7 turns with interleaved tool calls.

## Available Tools (17 total)
{tools}

## Training Task Samples (12 of 84)
{tasks_text}

## Rubric Requirements

Generate **15-20 deterministic, quantitative scoring rubrics**. Each rubric:
1. Checks ONE specific aspect of agent behavior
2. Uses 0-10 scale with specific numerical deductions
3. Is executable as a CODE RULE (parsing tool call names, args, conversation text) — no LLM needed

Dimensions to cover (distribute rubrics across all 6):

**tool_selection**: Right tools? Wrong tools? Correct order? (e.g., lookup user BEFORE modifying)
**info_sufficiency**: All needed info gathered before responding to user?
**step_efficiency**: Minimal steps? Redundant queries? Unnecessary turns?
**error_recovery**: Retry on tool error? Corrected args? Didn't give up prematurely?
**task_completion**: ALL user requests fulfilled? Verified results? User satisfied?
**communication**: Specific results (order numbers, dates, amounts)? Clear next steps?

## Critical Retail Patterns to Catch
- Agent MUST look up user identity FIRST (via name+zip or email) before ANY order operation
- Users often pack MULTIPLE requests in one message — all must be addressed
- Tool errors need retry with CORRECTED arguments (not the same bad call)
- Agent must VERIFY modifications were applied (re-query after change)
- Redundant tool calls (same tool+args twice) waste limited turns (max 7-10)
- Specific order IDs, dates, prices, and product names MUST be in agent's response
- The agent must NOT call transfer_to_human_agents as a first resort

## Output Format
Return ONLY a JSON array. No markdown, no explanation:
[
  {{
    "name": "Short unique name (5-8 words)",
    "dimension": "tool_selection|info_sufficiency|step_efficiency|error_recovery|task_completion|communication",
    "rule": "Start at 10. [List of specific deductions, each with point value]. Clamp final score to [0,10].",
    "applies_to": ["all"]
  }}
]

IMPORTANT RULE WRITING GUIDELINES:
- Deductions must reference SPECIFIC tool names, argument patterns, or conversation text
- Each deduction must have a clear NUMERICAL point value (-1, -2, -3, -5, etc.)
- NOT: "poor quality response" → YES: "agent did not include order_id in response to user"
- Focus on ERRORS THAT CAUSE TASK FAILURE, not minor style issues
- Generate 15-20 rubrics distributed across all 6 dimensions"""


def load_and_sample_tasks(train_path: str, n_samples: int = 12) -> list[dict]:
    """Load retail training tasks and select diverse samples."""
    with open(train_path, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if json.loads(line).get("domain") == "retail"]

    # Deduplicate by base task ID
    task_by_id: dict[str, dict] = {}
    for t in tasks:
        tid = t["id"].replace("_aug", "") if "_aug" in t["id"] else t["id"]
        if tid not in task_by_id:
            task_by_id[tid] = t

    unique = list(task_by_id.values())

    def assertion_count(t: dict) -> int:
        ev = t.get("evaluation", {})
        if isinstance(ev, dict):
            return len(ev.get("nl_assertions", []))
        return 0

    unique.sort(key=assertion_count, reverse=True)

    # Sample: 3 simplest, 6 medium, 3 most complex
    n = len(unique)
    samples = unique[-3:] + unique[n // 2 - 3 : n // 2 + 3] + unique[:3]
    print(f"Sampled {len(samples)} tasks from {len(unique)} unique retail tasks")
    return samples


def build_tasks_text(samples: list[dict]) -> str:
    """Build task description text for the prompt."""
    parts = []
    for i, t in enumerate(samples):
        ev = t.get("evaluation", {})
        if isinstance(ev, dict):
            assertions = ev.get("nl_assertions", [])
        else:
            assertions = []

        parts.append(
            f"### Task {i+1}: {t['id']}\n"
            f"**Customer Prompt:**\n{t['prompt'][:600]}\n\n"
            f"**Expected Behavior:** {json.dumps(assertions, ensure_ascii=False)}\n"
            f"---"
        )
    return "\n".join(parts)


def call_deepseek(prompt: str, temperature: float = 0.3) -> str:
    """Call DeepSeek API and return response text."""
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url=base_url)

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=4000,
    )
    return resp.choices[0].message.content


def parse_rubrics(raw: str) -> list[dict]:
    """Parse rubric JSON array from LLM response."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def validate_rubrics(rubrics: list[dict]) -> dict:
    """Validate rubric quality and coverage."""
    dims = {}
    for r in rubrics:
        d = r.get("dimension", "unknown")
        dims[d] = dims.get(d, 0) + 1

    issues = []
    if len(rubrics) < 12:
        issues.append(f"Only {len(rubrics)} rubrics, expected >= 15")
    if len(rubrics) > 25:
        issues.append(f"Too many rubrics: {len(rubrics)}, expected <= 22")

    for d in ["tool_selection", "info_sufficiency", "step_efficiency",
              "error_recovery", "task_completion", "communication"]:
        if dims.get(d, 0) == 0:
            issues.append(f"Missing dimension: {d}")

    return {
        "count": len(rubrics),
        "dimensions": dims,
        "issues": issues,
        "valid": len(issues) == 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate tau-bench retail rubrics via DeepSeek")
    parser.add_argument("--output", default=OUTPUT_PATH, help=f"Output path (default: {OUTPUT_PATH})")
    parser.add_argument("--dry-run", action="store_true", help="Build prompt but don't call API")
    parser.add_argument("--prompt-only", action="store_true", help="Print prompt and exit")
    args = parser.parse_args()

    samples = load_and_sample_tasks(TRAIN_PATH)
    tasks_text = build_tasks_text(samples)
    prompt = RUBRIC_GENERATION_PROMPT.format(tools=TOOLS_LIST, tasks_text=tasks_text)

    print(f"Prompt: {len(prompt)} chars, {len(samples)} task samples")

    if args.prompt_only:
        print("\n" + "=" * 60)
        print(prompt)
        return

    if args.dry_run:
        print("Dry run — skipping API call")
        prompt_path = "/tmp/rubric_generation_prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"Prompt saved to {prompt_path}")
        return

    # Estimate cost
    est_input_tokens = len(prompt) // 3
    est_cost = est_input_tokens * 0.14 / 1_000_000 + 3000 * 0.28 / 1_000_000
    print(f"Est. input: ~{est_input_tokens} tokens, est. cost: ~${est_cost:.4f}")

    print("Calling DeepSeek API...")
    raw = call_deepseek(prompt)
    print(f"Response: {len(raw)} chars")

    rubrics = parse_rubrics(raw)
    validation = validate_rubrics(rubrics)
    print(f"Generated {validation['count']} rubrics")
    print(f"Dimensions: {validation['dimensions']}")
    if validation["issues"]:
        print(f"Issues: {validation['issues']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
                "model": "deepseek-chat",
                "source": "84 retail training tasks (12 sampled)",
                "validation": validation,
            },
            "rubrics": rubrics,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")

    # Print first 3 for inspection
    print("\n=== Sample rubrics ===")
    for i, r in enumerate(rubrics[:3]):
        print(f"\n--- {r['name']} [{r['dimension']}] ---")
        print(r['rule'][:200] + "...")


if __name__ == "__main__":
    main()
