#!/usr/bin/env python3
"""
Align evaluation nl_assertions in augmented prompts with mock DB data.

For each augmented prompt:
1. Extract user identity (name + zip for retail, name for airline)
2. Look up the actual user in the mock DB
3. Fetch their orders/reservations
4. Use LLM (DeepSeek) to regenerate nl_assertions based on actual DB values
5. Write back updated evaluation fields to the JSONL files

Usage:
    python scripts/align_eval_assertions.py [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB helpers (inline to avoid circular imports)
# ---------------------------------------------------------------------------

def _lookup_retail_user(identity: dict) -> dict | None:
    """Look up a retail user by name + zip, or email, or name only."""
    from trainable_openclaw.agent.tau_bench_tools.mock_db import _seed_retail_users

    users = _seed_retail_users()
    name_raw = identity.get("name", "").strip()
    zipcode = identity.get("zipcode", "")
    email = identity.get("email", "").lower()

    # Normalize name: strip "user " prefix, convert snake_case to spaces
    name_raw = name_raw.strip()
    if name_raw.lower().startswith("user "):
        name_raw = name_raw[5:]
    name_parts = name_raw.replace("_", " ").strip().lower().split()
    first = name_parts[0] if len(name_parts) >= 1 else ""
    last = name_parts[-1] if len(name_parts) >= 2 else ""

    # 1. Exact match by name + zip
    if first and last and zipcode:
        for u in users:
            if (first == u["first_name"].lower()
                    and last == u["last_name"].lower()
                    and u.get("address", {}).get("zip") == zipcode):
                return u

    # 2. Match by email
    if email:
        for u in users:
            if u.get("email", "").lower() == email:
                return u

    # 3. Match by name only (first + last)
    if first and last:
        candidates = []
        for u in users:
            if (first == u["first_name"].lower()
                    and last == u["last_name"].lower()):
                candidates.append(u)
        if len(candidates) == 1:
            return candidates[0]

    # 4. Match by zipcode only (useful when name extraction fails but zip is known)
    if zipcode:
        candidates = [u for u in users if u.get("address", {}).get("zip") == zipcode]
        if len(candidates) == 1:
            return candidates[0]

    return None


def _lookup_airline_user(identity: dict) -> dict | None:
    """Look up an airline user by name or email."""
    from trainable_openclaw.agent.tau_bench_tools.mock_db import _seed_airline_users

    users = _seed_airline_users()
    name_raw = identity.get("name", "").strip()
    email = identity.get("email", "").lower()

    # Normalize name
    name_lower = name_raw.replace("_", " ").strip().lower()

    # 1. Exact full name match
    for u in users:
        if u["name"].lower() == name_lower:
            return u

    # 2. Match by email
    if email:
        for u in users:
            if u.get("email", "").lower() == email:
                return u

    return None


def _get_retail_orders(user_id: str) -> list[dict]:
    """Get all orders for a retail user."""
    from trainable_openclaw.agent.tau_bench_tools.mock_db import _seed_retail_orders

    return [o for o in _seed_retail_orders() if o["user_id"] == user_id]


def _get_airline_reservations(user_id: str) -> list[dict]:
    """Get all reservations for an airline user."""
    from trainable_openclaw.agent.tau_bench_tools.mock_db import _seed_airline_reservations

    return [r for r in _seed_airline_reservations() if r["user_id"] == user_id]


# ---------------------------------------------------------------------------
# Prompt parsing
# ---------------------------------------------------------------------------

def extract_identity(prompt: str, domain: str) -> dict[str, str]:
    """Extract user identity info from a prompt string.

    Uses multiple strategies because the augmented prompts have extremely
    varied formats.  Priority: email (most reliable) > name+zip > name only.
    """
    result = {}

    # --- Extract ALL emails from the prompt ---
    emails = re.findall(r'([a-z0-9_.]+@[a-z0-9_.]+)', prompt.lower())
    if emails:
        result["email"] = emails[0]

    # --- Extract ALL zipcodes ---
    zips = re.findall(r'(?:zip\s*(?:code|)\s*(?:is\s+|in\s+)?)(\d{5})', prompt[:500], re.IGNORECASE)
    if not zips:
        # "City, ST #####" or "City ST #####"
        zips = re.findall(r'[A-Z][a-z]+(?:\s*,\s*[A-Z]{2})?\s+(\d{5})', prompt[:500])
    if not zips:
        # Zip in parentheses: "(77027)"
        zips = re.findall(r'\((\d{5})\)', prompt[:500])
    if not zips:
        # Last resort: any 5-digit number in the first 300 chars (likely a zip)
        all_zips = re.findall(r'\b(\d{5})\b', prompt[:300])
        if all_zips:
            zips = [all_zips[0]]
    if zips:
        result["zipcode"] = zips[0]

    # --- Extract name ---
    NC = r"[A-Za-z][A-Za-z'\-]*"  # capitalized proper name part

    # Priority-ordered patterns
    name_patterns = [
        # "Your user id is '<name>'."  (snake_case identifiers)
        r"(?i)your user id is ['\"]?([a-z_']+)",
        r"(?i)user id is ['\"]?([a-z_']+)",
        # "You are <snake_case>(<email>)" or "You are <snake_case> with email"
        r'^You are ([a-z_]+)\s*\(',
        r'^You are ([a-z_]+) with email',
        r'^You are ([a-z_]+) in zipcode',
        r'^You are ([a-z_]+) living',
        r'^You are ([a-z_]+),? and\b',
        r'^You are user ([a-z_]+)',
        # "You're <Name>" contraction
        rf"^You're ({NC} {NC})",
        rf"^You're ({NC})",
        # "Your name is <Name>" / "You name is <Name>"
        rf'^Your name is ({NC} {NC})',
        rf'^You name is ({NC} {NC})',
        # "called <Name>" pattern (e.g., "an interesting guy called Nancy Davis")
        rf'called ({NC} {NC})',
        rf'called ({NC})',
        # Standard "You are <Name>" patterns
        rf'^You are ({NC} {NC})[,.\s]',
        rf'^You are ({NC} {NC}) in zip',
        rf'^You are ({NC} {NC}) living',
        rf'^You are ({NC} {NC}) from',
        rf'^You are ({NC} {NC}), and',
        rf'^You are ({NC} {NC})\.$',
        # Single name
        rf'^You are ({NC})[,.\s(]',
        # Generic
        r'^You are ([A-Za-z_]+ [A-Za-z_\']+)',
    ]
    for pat in name_patterns:
        m = re.search(pat, prompt, re.MULTILINE)
        if m:
            name = m.group(1).strip().rstrip(".").strip("'\"").rstrip("'\"")
            # Filter out obviously wrong "names"
            name_lower = name.lower()
            if name_lower not in ("an interesting", "also looking", "angry about", "also", "an", "pretty sure", "pretty"):
                result["name"] = name
                break

    return result


# ---------------------------------------------------------------------------
# LLM assertion generation
# ---------------------------------------------------------------------------

_ASSERTION_PROMPT = """You are updating evaluation criteria for a customer service task to match actual database values.

TASK PROMPT (what the customer asks):
{task_prompt}

ACTUAL USER DATA (from the database, these are the real values):
{user_data}

ACTUAL ORDER / RESERVATION DATA:
{order_data}

Generate new nl_assertions that match the ACTUAL database values above.
Rules:
1. Every assertion must reference values that EXIST in the actual user/order data above
2. Use exact numbers from the data (e.g., gift card balance $50.00, order total $142.99)
3. If the task asks about something not in the database (e.g., "Mastercard" but user only has Visa/Amex), adjust the assertion to match reality
4. Assertions should be verifiable from agent conversation + tool outputs
5. Focus on the customer's core goal: what must the agent accomplish for the task to be "complete"?
6. Keep assertions specific and measurable

Output ONLY a JSON array of assertion strings: ["assertion1", "assertion2"]
Do NOT include markdown fences, extra text, or commentary."""


def _format_user_data(user: dict) -> str:
    """Format user data as a readable string for the LLM prompt."""
    # Select key fields, format nicely
    fields = {}
    for key in ("user_id", "name", "email", "phone", "loyalty_tier", "loyalty_points",
                "gift_card_balance", "member_since"):
        if key in user:
            val = user[key]
            if isinstance(val, float):
                val = f"${val:.2f}"
            fields[key] = val
    if "address" in user:
        addr = user["address"]
        fields["address"] = f"{addr.get('address1','')}, {addr.get('city','')}, {addr.get('state','')} {addr.get('zip','')}"
    if "payment_methods" in user:
        pms = []
        for pm in user["payment_methods"]:
            ptype = pm.get("type", "")
            brand = pm.get("brand", "N/A")
            last4 = pm.get("last_four", "N/A")
            if ptype == "paypal":
                pms.append(f"PayPal ({pm.get('email','')})")
            else:
                pms.append(f"{ptype} {brand} (last 4: {last4})")
        fields["payment_methods"] = ", ".join(pms)
    return json.dumps(fields, indent=2)


def _format_order_data(orders: list[dict]) -> str:
    """Format order/reservation data as a readable string."""
    if not orders:
        return "(no orders/reservations)"
    summaries = []
    for o in orders:
        s = {
            "order_id": o.get("order_id", o.get("reservation_id", "")),
            "status": o.get("status", ""),
            "created_at": o.get("created_at", ""),
        }
        if "payment" in o:
            s["payment"] = o["payment"]
        if "payment_method" in o:
            s["payment_method"] = o["payment_method"]
        if "gift_card_applied" in o:
            s["gift_card_applied"] = o["gift_card_applied"]
        if "items" in o:
            s["items"] = [
                f"{it.get('name','')} x{it.get('quantity',0)} @ ${it.get('unit_price',0):.2f}"
                for it in o["items"]
            ]
        if "flights" in o:
            s["flights"] = [
                f"{seg.get('flight_number','')} {seg.get('date','')} {seg.get('origin','')}->{seg.get('destination','')} ({seg.get('cabin','economy')}) ${seg.get('fare',0):.2f}"
                for seg in o["flights"]
            ]
        summaries.append(s)
    return json.dumps(summaries, indent=2)


def generate_assertions(
    prompt_text: str,
    user_data: dict,
    order_data: list[dict],
    model: str = "deepseek-chat",
    api_key: str = "",
    base_url: str = "https://api.deepseek.com",
) -> list[str]:
    """Use LLM to generate nl_assertions matching actual DB data."""
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    if not api_key:
        logger.warning("No DEEPSEEK_API_KEY set — generating generic assertions")
        return _generate_fallback_assertions(prompt_text, user_data, order_data)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    system_prompt = _ASSERTION_PROMPT.format(
        task_prompt=prompt_text[:2000],
        user_data=_format_user_data(user_data),
        order_data=_format_order_data(order_data),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": system_prompt},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        return _parse_assertions(raw)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return _generate_fallback_assertions(prompt_text, user_data, order_data)


def _parse_assertions(raw: str) -> list[str]:
    """Parse the LLM output into a list of assertion strings."""
    text = raw.strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        result = json.loads(text)
        if isinstance(result, list) and all(isinstance(a, str) for a in result):
            return result
    except json.JSONDecodeError:
        pass
    # Try to extract JSON array from text
    m = re.search(r'\[[\s\S]*?\]', text)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    # Fallback: treat each non-empty line as an assertion
    lines = [l.strip().strip('"').strip("'") for l in text.split("\n") if l.strip()]
    lines = [l for l in lines if not l.startswith("```") and len(l) > 10]
    return lines if lines else ["Agent should complete the task correctly."]


def _generate_fallback_assertions(
    prompt_text: str,
    user_data: dict,
    order_data: list[dict],
) -> list[str]:
    """Generate basic assertions without LLM, using actual DB values."""
    assertions = []

    # Check what the user seems to want from the prompt
    prompt_lower = prompt_text.lower()

    # Gift card balance assertions
    if "gift card" in prompt_lower or "gift_card" in prompt_lower:
        balance = user_data.get("gift_card_balance", 0.0)
        assertions.append(
            f"Agent should correctly tell the user their gift card balance is ${balance:.2f}."
        )

    # Order-related assertions
    if any(kw in prompt_lower for kw in ("order", "tracking", "refund", "cancel", "return", "exchange")):
        if order_data:
            for o in order_data:
                oid = o.get("order_id", o.get("reservation_id", ""))
                status = o.get("status", "")
                if "cancel" in prompt_lower and status == "pending":
                    assertions.append(f"Agent should cancel order {oid} if the user requests cancellation.")
                elif "return" in prompt_lower and status == "delivered":
                    assertions.append(f"Agent should process the return for order {oid} if requested.")
                elif "exchange" in prompt_lower:
                    assertions.append(f"Agent should process the exchange for order {oid} if requested.")
        assertions.append("Agent should provide accurate order information based on tool query results.")

    # Payment method assertions
    if "payment" in prompt_lower or "credit card" in prompt_lower or "mastercard" in prompt_lower.lower():
        pms = user_data.get("payment_methods", [])
        if pms:
            brand_names = []
            for pm in pms:
                if pm.get("type") == "paypal":
                    brand_names.append("PayPal")
                elif pm.get("type") == "gift_card":
                    brand_names.append("Store Gift Card")
                else:
                    brand_names.append(pm.get("brand", pm.get("type", "Unknown")))
            if brand_names:
                assertions.append(
                    f"Agent should correctly identify the user's available payment methods: {', '.join(brand_names)}."
                )

    # Fallback
    if not assertions:
        if user_data.get("gift_card_balance", 0) > 0:
            assertions.append(
                f"Agent should tell the user their gift card balance is ${user_data['gift_card_balance']:.2f} if asked."
            )
        assertions.append("Agent should complete the task using available tools and provide accurate information.")

    return assertions


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_prompt(entry: dict, dry_run: bool = False) -> dict:
    """Process one augmented prompt entry: look up DB data and generate assertions."""
    prompt = entry.get("prompt", "")
    domain = entry.get("domain", "retail")
    identity = extract_identity(prompt, domain)

    user_data = None
    order_data = []

    if domain == "retail":
        user_data = _lookup_retail_user(identity)
    elif domain == "airline":
        user_data = _lookup_airline_user(identity)

    if user_data is None:
        logger.warning("Could not find user for prompt %s (identity=%s)", entry.get("id", "?"), identity)
        return entry

    # Get orders/reservations
    if domain == "retail":
        order_data = _get_retail_orders(user_data["user_id"])
    elif domain == "airline":
        order_data = _get_airline_reservations(user_data["user_id"])

    # Generate assertions
    assertions = generate_assertions(prompt, user_data, order_data)

    if dry_run:
        logger.info("DRY-RUN %s (%s): user=%s, orders=%d, assertions=%s",
                     entry.get("id", "?"), domain,
                     user_data.get("user_id", "?"), len(order_data),
                     json.dumps(assertions))
    else:
        ev = entry.get("evaluation", {})
        if isinstance(ev, str):
            try:
                ev = json.loads(ev)
            except (json.JSONDecodeError, ValueError):
                ev = {}
        ev["nl_assertions"] = assertions
        entry["evaluation"] = ev

    return entry


def main():
    parser = argparse.ArgumentParser(description="Align evaluation assertions with mock DB data")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N prompts per file (0=all)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM, use fallback assertions only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    base_dir = Path(__file__).resolve().parent.parent / "data" / "tau_bench"
    files = [
        base_dir / "train_prompts_augmented.jsonl",
        base_dir / "val_prompts_augmented.jsonl",
        base_dir / "test_prompts_augmented.jsonl",
    ]

    total_processed = 0
    total_updated = 0

    for filepath in files:
        if not filepath.exists():
            logger.warning("File not found: %s", filepath)
            continue

        entries = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        logger.info("Processing %s: %d entries", filepath.name, len(entries))

        limit = args.limit if args.limit > 0 else len(entries)
        updated = 0

        for i, entry in enumerate(entries[:limit]):
            # Skip entries that already have non-null assertions matching DB data
            # (we overwrite all to ensure consistency)
            original = json.dumps(entry.get("evaluation", {}))
            entry = process_prompt(entry, dry_run=args.dry_run)
            if json.dumps(entry.get("evaluation", {})) != original:
                updated += 1

        if not args.dry_run:
            with open(filepath, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info("  Updated %d/%d entries in %s", updated, limit, filepath.name)
        total_processed += limit
        total_updated += updated

    logger.info("Done: %d processed, %d updated across %d files",
                total_processed, total_updated, len(files))


if __name__ == "__main__":
    main()
