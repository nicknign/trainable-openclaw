#!/usr/bin/env python3
"""
augment_prompts.py — Entity-based prompt augmentation for trainable-openclaw.

Replaces tau-bench fictional entities with mock DB entities, ensuring every
augmented prompt is executable against MockDatabase tools (find_user_id_by_name_zip
resolves, orders belong to correct users, etc.).

Phase 1: Entity replacement (mandatory, pure Python)
Phase 2: LLM paraphrasing via DeepSeek (optional)
Phase 3: Train/val split (9:1 from augmented train, test stays holdout)

Usage:
    python scripts/data/augment_prompts.py --train-target 500 --test-target 50 --validate 20
    python scripts/data/augment_prompts.py --dry-run  # inspect entity extraction
    python scripts/data/augment_prompts.py --llm-paraphrase  # Phase 1 + Phase 2

Outputs:
    data/tau_bench/train_prompts_augmented.jsonl  — 450 entries (after 9:1 split)
    data/tau_bench/val_prompts_augmented.jsonl    — 50 entries (holdin, 9:1 from train)
    data/tau_bench/test_prompts_augmented.jsonl   — 50 entries (holdout)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PROMPTS = PROJECT_ROOT / "data" / "tau_bench" / "train_prompts.jsonl"
TEST_PROMPTS = PROJECT_ROOT / "data" / "tau_bench" / "test_prompts.jsonl"
TRAIN_OUT = PROJECT_ROOT / "data" / "tau_bench" / "train_prompts_augmented.jsonl"
VAL_OUT = PROJECT_ROOT / "data" / "tau_bench" / "val_prompts_augmented.jsonl"
TEST_OUT = PROJECT_ROOT / "data" / "tau_bench" / "test_prompts_augmented.jsonl"

SEED = 42
random.seed(SEED)

# Airport codes used in tau-bench prompts (3-letter uppercase)
_AIRPORT_CODES = {
    "SFO", "LAX", "JFK", "ORD", "MIA", "SEA", "DFW", "ATL", "BOS", "DEN",
    "IAD", "PHX", "LAS", "PDX", "SAN", "HNL", "LHR", "CDG", "NRT", "SYD",
    "DTW", "LGA", "PHL", "MCO", "IAH",
}

# US state names → abbreviations and reverse
_STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}
_STATE_ABBREV_TO_NAME = {v: k.title() for k, v in _STATE_NAMES.items()}
# Set of valid state abbreviations for detection
_VALID_STATE_ABBREVS = set(_STATE_NAMES.values())

# Words that look like airport codes but aren't (filter out from entity extraction)
_NOT_AIRPORTS = {
    "THE", "AND", "FOR", "NOT", "BUT", "ALL", "CAN", "HAS", "WAS", "ARE",
    "YOU", "YET", "NOW", "NEW", "OLD", "VIA", "WAY", "DAY", "USA", "USE",
    "ANY", "HAS", "HIS", "HER", "OUR", "OUT", "HOW", "WHO", "WHY", "YES",
    "GET", "GOT", "PUT", "SET", "SAY", "SEE", "SHE", "HE", "WE", "TO",
    "THEY", "THAT", "THIS", "THEN", "THAN", "WILL", "WITH", "FROM",
    "ITS", "IT", "IN", "ON", "AT", "BE", "BY", "DO", "GO", "NO", "SO",
    "UP", "US", "AM", "PM", "FEB", "MAR", "APR", "MAY", "JUN", "JUL",
    "JAN", "AUG", "SEP", "OCT", "NOV", "DEC",
}


# ===========================================================================
# Entity Catalog — mirrors MockDatabase seed data (mock_db.py)
# ===========================================================================

def _build_retail_users() -> list[dict]:
    """Retail users from mock_db _seed_retail_users()."""
    return [
        {"user_id": "U001", "name": "Alice Chen", "first_name": "Alice", "last_name": "Chen",
         "email": "alice.chen@email.com", "zip": "94102", "city": "San Francisco", "state": "CA",
         "orders": ["O001", "O002"]},
        {"user_id": "U002", "name": "Bob Williams", "first_name": "Bob", "last_name": "Williams",
         "email": "bob.w@email.com", "zip": "10001", "city": "New York", "state": "NY",
         "orders": ["O003", "O004"]},
        {"user_id": "U003", "name": "Carlos Rodriguez", "first_name": "Carlos", "last_name": "Rodriguez",
         "email": "carlos.r@email.com", "zip": "33101", "city": "Miami", "state": "FL",
         "orders": ["O005", "O006"]},
        {"user_id": "U004", "name": "Diana Park", "first_name": "Diana", "last_name": "Park",
         "email": "diana.park@email.com", "zip": "60601", "city": "Chicago", "state": "IL",
         "orders": ["O007", "O008"]},
        {"user_id": "U005", "name": "Edward Kim", "first_name": "Edward", "last_name": "Kim",
         "email": "ed.kim@email.com", "zip": "98101", "city": "Seattle", "state": "WA",
         "orders": ["O009", "O010"]},
        {"user_id": "U006", "name": "Fatima Hassan", "first_name": "Fatima", "last_name": "Hassan",
         "email": "fatima.h@email.com", "zip": "02108", "city": "Boston", "state": "MA",
         "orders": ["O011"]},
        {"user_id": "U007", "name": "George Thompson", "first_name": "George", "last_name": "Thompson",
         "email": "george.t@email.com", "zip": "80202", "city": "Denver", "state": "CO",
         "orders": ["O012"]},
        {"user_id": "U008", "name": "Hannah Lee", "first_name": "Hannah", "last_name": "Lee",
         "email": "hannah.lee@email.com", "zip": "90028", "city": "Los Angeles", "state": "CA",
         "orders": ["O013"]},
        {"user_id": "U009", "name": "Ivan Petrov", "first_name": "Ivan", "last_name": "Petrov",
         "email": "ivan.p@email.com", "zip": "77027", "city": "Houston", "state": "TX",
         "orders": []},
        {"user_id": "U010", "name": "Julia Martinez", "first_name": "Julia", "last_name": "Martinez",
         "email": "julia.m@email.com", "zip": "85004", "city": "Phoenix", "state": "AZ",
         "orders": ["O014"]},
        {"user_id": "U011", "name": "Kevin O'Brien", "first_name": "Kevin", "last_name": "O'Brien",
         "email": "kevin.ob@email.com", "zip": "97205", "city": "Portland", "state": "OR",
         "orders": []},
        {"user_id": "U012", "name": "Lisa Nakamura", "first_name": "Lisa", "last_name": "Nakamura",
         "email": "lisa.n@email.com", "zip": "96814", "city": "Honolulu", "state": "HI",
         "orders": ["O015"]},
        {"user_id": "U013", "name": "Michael Brown", "first_name": "Michael", "last_name": "Brown",
         "email": "michael.b@email.com", "zip": "30309", "city": "Atlanta", "state": "GA",
         "orders": []},
        {"user_id": "U014", "name": "Nancy Davis", "first_name": "Nancy", "last_name": "Davis",
         "email": "nancy.d@email.com", "zip": "75207", "city": "Dallas", "state": "TX",
         "orders": []},
        {"user_id": "U015", "name": "Oscar Garcia", "first_name": "Oscar", "last_name": "Garcia",
         "email": "oscar.g@email.com", "zip": "78205", "city": "San Antonio", "state": "TX",
         "orders": []},
    ]


def _build_airline_users() -> list[dict]:
    """Airline users from mock_db _seed_airline_users()."""
    return [
        {"user_id": "UA001", "name": "Alice Chen", "first_name": "Alice", "last_name": "Chen",
         "email": "alice.chen@email.com", "zip": "94102", "city": "San Francisco", "state": "CA",
         "reservations": ["RA001"]},
        {"user_id": "UA002", "name": "Bob Williams", "first_name": "Bob", "last_name": "Williams",
         "email": "bob.w@email.com", "zip": "10001", "city": "New York", "state": "NY",
         "reservations": ["RA002"]},
        {"user_id": "UA003", "name": "Carlos Rodriguez", "first_name": "Carlos", "last_name": "Rodriguez",
         "email": "carlos.r@email.com", "zip": "33101", "city": "Miami", "state": "FL",
         "reservations": ["RA003", "RA007"]},
        {"user_id": "UA004", "name": "Diana Park", "first_name": "Diana", "last_name": "Park",
         "email": "diana.park@email.com", "zip": "60601", "city": "Chicago", "state": "IL",
         "reservations": ["RA011"]},
        {"user_id": "UA005", "name": "Edward Kim", "first_name": "Edward", "last_name": "Kim",
         "email": "ed.kim@email.com", "zip": "98101", "city": "Seattle", "state": "WA",
         "reservations": ["RA004", "RA009"]},
        {"user_id": "UA006", "name": "Fatima Hassan", "first_name": "Fatima", "last_name": "Hassan",
         "email": "fatima.h@email.com", "zip": "02108", "city": "Boston", "state": "MA",
         "reservations": []},
        {"user_id": "UA007", "name": "George Thompson", "first_name": "George", "last_name": "Thompson",
         "email": "george.t@email.com", "zip": "80202", "city": "Denver", "state": "CO",
         "reservations": ["RA010"]},
        {"user_id": "UA008", "name": "Hannah Lee", "first_name": "Hannah", "last_name": "Lee",
         "email": "hannah.lee@email.com", "zip": "90028", "city": "Los Angeles", "state": "CA",
         "reservations": ["RA005"]},
        {"user_id": "UA009", "name": "Ivan Petrov", "first_name": "Ivan", "last_name": "Petrov",
         "email": "ivan.p@email.com", "zip": "77027", "city": "Houston", "state": "TX",
         "reservations": []},
        {"user_id": "UA010", "name": "Julia Martinez", "first_name": "Julia", "last_name": "Martinez",
         "email": "julia.m@email.com", "zip": "85004", "city": "Phoenix", "state": "AZ",
         "reservations": ["RA008"]},
        {"user_id": "UA011", "name": "Kevin O'Brien", "first_name": "Kevin", "last_name": "O'Brien",
         "email": "kevin.ob@email.com", "zip": "97205", "city": "Portland", "state": "OR",
         "reservations": []},
        {"user_id": "UA012", "name": "Lisa Nakamura", "first_name": "Lisa", "last_name": "Nakamura",
         "email": "lisa.n@email.com", "zip": "96814", "city": "Honolulu", "state": "HI",
         "reservations": ["RA006"]},
        {"user_id": "UA013", "name": "Michael Brown", "first_name": "Michael", "last_name": "Brown",
         "email": "michael.b@email.com", "zip": "30309", "city": "Atlanta", "state": "GA",
         "reservations": []},
        {"user_id": "UA014", "name": "Nancy Davis", "first_name": "Nancy", "last_name": "Davis",
         "email": "nancy.d@email.com", "zip": "75207", "city": "Dallas", "state": "TX",
         "reservations": ["RA012"]},
        {"user_id": "UA015", "name": "Oscar Garcia", "first_name": "Oscar", "last_name": "Garcia",
         "email": "oscar.g@email.com", "zip": "78205", "city": "San Antonio", "state": "TX",
         "reservations": []},
    ]


def _build_retail_orders() -> list[dict]:
    """Retail orders from mock_db _seed_retail_orders()."""
    return [
        {"order_id": "O001", "user_id": "U001", "status": "delivered"},
        {"order_id": "O002", "user_id": "U001", "status": "pending"},
        {"order_id": "O003", "user_id": "U002", "status": "delivered"},
        {"order_id": "O004", "user_id": "U002", "status": "shipped"},
        {"order_id": "O005", "user_id": "U003", "status": "delivered"},
        {"order_id": "O006", "user_id": "U003", "status": "pending"},
        {"order_id": "O007", "user_id": "U004", "status": "delivered"},
        {"order_id": "O008", "user_id": "U004", "status": "pending"},
        {"order_id": "O009", "user_id": "U005", "status": "delivered"},
        {"order_id": "O010", "user_id": "U005", "status": "cancelled"},
        {"order_id": "O011", "user_id": "U006", "status": "shipped"},
        {"order_id": "O012", "user_id": "U007", "status": "delivered"},
        {"order_id": "O013", "user_id": "U008", "status": "pending"},
        {"order_id": "O014", "user_id": "U010", "status": "delivered"},
        {"order_id": "O015", "user_id": "U012", "status": "pending"},
    ]


def _build_airline_reservations() -> list[dict]:
    """Airline reservations from mock_db _seed_airline_reservations()."""
    return [
        {"reservation_id": "RA001", "user_id": "UA001"},
        {"reservation_id": "RA002", "user_id": "UA002"},
        {"reservation_id": "RA003", "user_id": "UA003"},
        {"reservation_id": "RA004", "user_id": "UA005"},
        {"reservation_id": "RA005", "user_id": "UA008"},
        {"reservation_id": "RA006", "user_id": "UA012"},
        {"reservation_id": "RA007", "user_id": "UA003"},
        {"reservation_id": "RA008", "user_id": "UA010"},
        {"reservation_id": "RA009", "user_id": "UA005"},
        {"reservation_id": "RA010", "user_id": "UA007"},
        {"reservation_id": "RA011", "user_id": "UA004"},
        {"reservation_id": "RA012", "user_id": "UA014"},
    ]


def _build_flights() -> list[dict]:
    """Flights from mock_db _seed_airline_flights()."""
    return [
        {"flight_number": "AA101", "origin": "SFO", "destination": "JFK"},
        {"flight_number": "DL202", "origin": "SFO", "destination": "JFK"},
        {"flight_number": "UA303", "origin": "SFO", "destination": "LAX"},
        {"flight_number": "AA111", "origin": "ORD", "destination": "MIA"},
        {"flight_number": "DL404", "origin": "LAX", "destination": "HNL"},
        {"flight_number": "UA909", "origin": "SFO", "destination": "ORD"},
        {"flight_number": "DL606", "origin": "LAX", "destination": "JFK"},
        {"flight_number": "AS313", "origin": "SEA", "destination": "SFO"},
        {"flight_number": "AA515", "origin": "DFW", "destination": "ATL"},
        {"flight_number": "UA717", "origin": "BOS", "destination": "DEN"},
        {"flight_number": "BA707", "origin": "JFK", "destination": "LHR"},
    ]


def _build_product_names() -> dict[str, str]:
    """Map of product short names to mock DB product names (P001-P012)."""
    return {
        "headphones": "Wireless Noise-Cancelling Headphones",
        "watch": "Smart Fitness Watch",
        "fitness watch": "Smart Fitness Watch",
        "bluetooth speaker": "Portable Bluetooth Speaker",
        "speaker": "Portable Bluetooth Speaker",
        "running shoes": "Professional Running Shoes",
        "winter jacket": "Winter Insulated Jacket",
        "cotton t-shirt": "Premium Cotton T-Shirt",
        "coffee maker": "Programmable Coffee Maker",
        "toaster": "Stainless Steel Toaster",
        "vacuum": "Cordless Stick Vacuum",
        "stick vacuum": "Cordless Stick Vacuum",
        "office chair": "Ergonomic Office Chair",
    }


def build_entity_pools() -> dict:
    """Assemble all entity pools."""
    return {
        "retail_users": _build_retail_users(),
        "retail_orders": _build_retail_orders(),
        "airline_users": _build_airline_users(),
        "airline_reservations": _build_airline_reservations(),
        "flights": _build_flights(),
        "product_names": _build_product_names(),
    }


# ===========================================================================
# Entity Extraction — identify all replaceable entities in prompt text
# ===========================================================================

def extract_entities(prompt: str, domain: str) -> dict[str, Any]:
    """Parse entity values from a prompt that need replacement.

    Returns a dict with keys like original_user_id_str, original_first_name,
    original_last_name, original_full_name, original_zip, original_city,
    original_state, original_email, original_order_refs,
    original_reservation_refs, original_flight_refs, original_product_refs.
    Each value is the detected original string (or None/[]).
    """
    ent: dict[str, Any] = {
        "original_user_id_str": None,      # e.g. "noah_ito_3850"
        "original_first_name": None,
        "original_last_name": None,
        "original_full_name": None,        # "Noah Ito"
        "original_zip": None,
        "original_city": None,
        "original_state": None,
        "original_email": None,
        "original_order_refs": [],          # e.g. ["#W9300146", "W4284542"]
        "original_reservation_refs": [],    # e.g. ["XEHM4B", "59XX6W"]
        "original_flight_refs": [],         # e.g. ["HAT266"]
        "original_product_refs": [],        # e.g. ["Desk Lamp", "tea kettle"]
    }

    text = prompt

    # --- user_id: first_last_digits pattern ---
    m_uid = re.search(r'\b([a-z]+)_([a-z]+)_(\d{3,5})\b', text)
    if m_uid:
        ent["original_user_id_str"] = m_uid.group(0)
        # Derive first/last name from user_id if not found elsewhere
        if not ent["original_first_name"]:
            ent["original_first_name"] = m_uid.group(1).capitalize()
        if not ent["original_last_name"]:
            ent["original_last_name"] = m_uid.group(2).capitalize()

    # --- Full name: "First Last" from greeting patterns ---
    name_pats = [
        # "You are First Last" / "Your name is First Last" / "You name is First Last"
        r'(?:You are|you are|Your name is|You name is|you\'re|You\'re)\s+'
        r'(?:user\s+)?'  # skip "user" if present before uid
        r'(?:an interesting guy called\s+)?'  # "You are an interesting guy called Noah Patel"
        r'([A-Z][a-z]+(?:\'[A-Z][a-z]+)?)\s+([A-Z][a-z]+(?:\'[A-Z][a-z]+)?)',
        # "name is First Last" (mid-sentence)
        r'(?:name is|Name is)\s+([A-Z][a-z]+(?:\'[A-Z][a-z]+)?)\s+([A-Z][a-z]+(?:\'[A-Z][a-z]+)?)',
        # "called First Last" (standalone)
        r'(?:called|named)\s+([A-Z][a-z]+(?:\'[A-Z][a-z]+)?)\s+([A-Z][a-z]+(?:\'[A-Z][a-z]+)?)',
    ]
    for pat in name_pats:
        m = re.search(pat, text)
        if m:
            first = m.group(1)
            last = m.group(2)
            # Reject false matches on common non-name words
            if first in ("New", "San", "Los", "St", "El", "Big", "You", "Your"):
                continue
            ent["original_first_name"] = first
            ent["original_last_name"] = last
            ent["original_full_name"] = f"{first} {last}"
            break

    # Fallback: first two capitalized words at start of text (after optional "You are user")
    if not ent["original_full_name"]:
        m = re.search(
            r'(?:^You are (?:user )?|^Your user id is \S+\s*|^)'
            r'([A-Z][a-z]+(?:\'[A-Z][a-z]+)?)\s+([A-Z][a-z]+(?:\'[A-Z][a-z]+)?)\b',
            text)
        if m:
            cand_first, cand_last = m.group(1), m.group(2)
            if cand_first not in ("New", "San", "Los", "St", "El", "Big"):
                ent["original_first_name"] = cand_first
                ent["original_last_name"] = cand_last
                ent["original_full_name"] = f"{cand_first} {cand_last}"

    # --- Email(s) — some prompts have multiple emails ---
    all_emails = re.findall(r'[\w.+-]+@example\.com', text)
    if all_emails:
        ent["original_email"] = all_emails[0]
        ent["original_emails_all"] = list(set(all_emails))

    # --- Zip code (5-digit) in context ---
    for pat in [
        r'\bzip\s*code\s*(?:is\s*)?(\d{5})\b',
        r'\bzipcode\s+(\d{5})\b',
        r'[A-Z]{2}\s+(\d{5})\b',          # "WA 98187"
        r',?\s*(\d{5})\b',                 # "Philadelphia 19031"
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            zip_val = m.group(1)
            # Skip zips that look like street numbers (too low)
            if int(zip_val) >= 10000:
                ent["original_zip"] = zip_val
                break

    # --- City + State ---
    # "City, ST" or "City ST zip"
    m_cs = re.search(
        r'(?:in|at|from|to|residing in|living in|live in|residing)\s+'
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),?\s+([A-Z]{2})\b',
        text)
    if not m_cs:
        m_cs = re.search(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s+([A-Z]{2})', text)
    if m_cs:
        ent["original_city"] = m_cs.group(1)
        ent["original_state"] = m_cs.group(2)

    # Also detect full state names ("in Texas", "from Florida", etc.)
    if not ent.get("original_state"):
        for state_name, state_abbrev in _STATE_NAMES.items():
            # Only match multi-word or distinctive state names to avoid false
            # positives from city names ("Washington" could be DC or state)
            pattern = r'\b(' + re.escape(state_name) + r')\b'
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                # Don't match "New York" if it's clearly a city
                found = m.group(1)
                # For "New York": only treat as state if preceded by state-like context
                if found.lower() == "new york":
                    city_m = re.search(r'(?:in|at|from|to)\s+New York', text)
                    if not city_m or re.search(r'New York,\s*NY', text):
                        ent["original_state"] = state_abbrev
                        ent["original_state_name"] = found
                        break
                elif found.lower() == "washington":
                    # Could be DC or state - only treat as state in clear context
                    if re.search(r'(?:in|from)\s+Washington\s+(?:state|in)', text, re.IGNORECASE):
                        ent["original_state"] = state_abbrev
                        ent["original_state_name"] = found
                        break
                    elif "washington dc" not in text.lower():
                        # Ambiguous — extract anyway
                        ent["original_state"] = state_abbrev
                        ent["original_state_name"] = found
                        break
                else:
                    ent["original_state"] = state_abbrev
                    ent["original_state_name"] = found
                    break

    # Also detect bare state abbreviations in context
    if not ent.get("original_state"):
        for abbrev in _VALID_STATE_ABBREVS:
            # Look for "in ST" or "from ST" or "ST zipcode"
            if re.search(r'\b(in|from|to|at)\s+' + abbrev + r'\b', text):
                ent["original_state"] = abbrev
                break
            if re.search(r'\b' + abbrev + r'\s+\d{5}\b', text):
                ent["original_state"] = abbrev
                break

    # --- Order IDs: #Wnnnnnnn or Wnnnnnnn format ---
    order_refs = []
    for m in re.finditer(r'#?(W\d{5,10})\b', text):
        ref = m.group(0)  # includes # if present
        order_refs.append(ref)
    ent["original_order_refs"] = sorted(set(order_refs))

    # --- Reservation codes: 5-6 char uppercase alphanumeric (not airports/words) ---
    if domain == "airline":
        res_codes = []
        for m in re.finditer(r'\b([A-Z0-9]{5,6})\b', text):
            code = m.group(1)
            has_digit = any(c.isdigit() for c in code)
            has_alpha = any(c.isalpha() for c in code)
            if (has_digit and has_alpha and
                    code not in _AIRPORT_CODES and
                    code not in _NOT_AIRPORTS):
                res_codes.append(code)
        ent["original_reservation_refs"] = sorted(set(res_codes))

    # --- Flight numbers: XX#### or XX### format ---
    flight_refs = []
    for m in re.finditer(r'\b([A-Z]{2}\d{2,4})\b', text):
        fn = m.group(1)
        flight_refs.append(fn)
    if not flight_refs:
        # Some tau-bench prompts use non-standard flight formats like "HAT266"
        for m in re.finditer(r'\b([A-Z]{3}\d{3})\b', text):
            code = m.group(1)
            if code not in _AIRPORT_CODES:
                flight_refs.append(code)
    ent["original_flight_refs"] = sorted(set(flight_refs))

    # --- Product references: quoted or specific product names mentioned ---
    # Tau-bench prompts sometimes quote product names like 'Desk Lamp', 'Tea Kettle'
    product_refs = []
    for m in re.finditer(r"'([^']+)'|\"([^\"]+)\"", text):
        prod = m.group(1) or m.group(2)
        # Skip if this looks like a user_id (first_last_digits) or email
        if re.match(r'^[a-z]+_[a-z]+_\d{3,5}$', prod):
            continue
        if "@" in prod:
            continue
        if len(prod.split()) <= 4 and len(prod) > 2:
            product_refs.append(prod)
    ent["original_product_refs"] = sorted(set(product_refs))

    return ent


# ===========================================================================
# Entity Replacement — map original entities to mock DB entities
# ===========================================================================

def _pick_new_user(pools: dict, domain: str, used_user_ids: set[str]) -> dict | None:
    """Pick a mock DB user not yet used for this original prompt's variants."""
    if domain == "retail":
        candidates = [u for u in pools["retail_users"]
                      if u["user_id"] not in used_user_ids]
    else:
        candidates = [u for u in pools["airline_users"]
                      if u["user_id"] not in used_user_ids]
    if not candidates:
        return None
    return random.choice(candidates)


def _build_replacement_map(
    original_entities: dict[str, Any],
    new_user: dict,
    pools: dict,
    domain: str,
) -> dict[str, str]:
    """Build old→new text replacement map. Ensures consistency:
    user_id ↔ name ↔ zip ↔ orders/reservations all match.

    Returns dict of {old_text: new_text} sorted by key length desc for safe replacement.
    """
    rep: dict[str, str] = {}

    ent = original_entities
    new_uid = new_user["user_id"]
    new_first = new_user["first_name"]
    new_last = new_user["last_name"]
    new_full = new_user["name"]
    new_zip = new_user["zip"]
    new_city = new_user["city"]
    new_state = new_user["state"]
    new_email = new_user["email"]

    # --- user_id string replacement ---
    old_uid = ent.get("original_user_id_str")
    if old_uid:
        # Replace the fictional user_id_str (first_last_digits) with the new
        # user's human-readable snake_case name. Prompts use this string as both
        # a display identity ("You are X living in...") and as a system identifier
        # ("Your user id is X"). The snake_case name works for both contexts.
        new_username = f"{new_first.lower()}_{new_last.lower()}"
        rep[old_uid] = new_username
        # Also replace bare first_last (no digits) if it appears separately
        old_bare = "_".join(old_uid.split("_")[:2])
        new_bare = new_username
        if old_bare != new_bare and old_bare in old_uid:
            rep[old_bare] = new_bare

    # --- Name replacements (longest first to avoid partials) ---
    old_full = ent.get("original_full_name")
    old_first = ent.get("original_first_name")
    old_last = ent.get("original_last_name")

    if old_full and old_full != new_full:
        rep[old_full] = new_full
    if old_first and old_first != new_first:
        rep[old_first] = new_first
    if old_last and old_last != new_last:
        rep[old_last] = new_last
    # Lowercase name variants
    if old_first:
        rep[old_first.lower()] = new_first.lower()
    if old_last:
        rep[old_last.lower()] = new_last.lower()

    # --- Email(s) — replace ALL detected emails ---
    old_email = ent.get("original_email")
    all_old_emails = ent.get("original_emails_all", [])
    if all_old_emails:
        for old_em in all_old_emails:
            rep[old_em] = new_email

    # --- Zip ---
    old_zip = ent.get("original_zip")
    if old_zip and old_zip != new_zip:
        rep[old_zip] = new_zip

    # --- City ---
    old_city = ent.get("original_city")
    if old_city and old_city != new_city:
        rep[old_city] = new_city

    # --- State abbreviation ---
    old_state = ent.get("original_state")
    if old_state and old_state != new_state:
        rep[old_state] = new_state

    # --- State full name (only if different from city to avoid ambiguity) ---
    old_state_name = ent.get("original_state_name")
    old_city = ent.get("original_city")
    if old_state_name and old_state_name != old_city:
        new_state_name = _STATE_ABBREV_TO_NAME.get(new_state, new_state)
        if old_state_name.lower() != new_state_name.lower():
            rep[old_state_name] = new_state_name

    # --- Order references: map tau-bench #W#### to mock DB O### ---
    old_orders = ent.get("original_order_refs", [])
    if old_orders and domain == "retail":
        user_orders = new_user.get("orders", [])
        for i, old_oid in enumerate(old_orders):
            if user_orders:
                new_oid = user_orders[i % len(user_orders)]
                rep[old_oid] = new_oid
                # Also handle #-prefixed variant
                if old_oid.startswith("#"):
                    rep[old_oid] = f"#{new_oid}"
                else:
                    rep[f"#{old_oid}"] = f"#{new_oid}"

    # --- Reservation references: map tau-bench alphanumeric codes to RA### ---
    old_res = ent.get("original_reservation_refs", [])
    if old_res and domain == "airline":
        user_res = new_user.get("reservations", [])
        for i, old_rc in enumerate(old_res):
            if user_res:
                new_rc = user_res[i % len(user_res)]
                rep[old_rc] = new_rc

    # --- Flight references: map to mock DB flight numbers ---
    old_flights = ent.get("original_flight_refs", [])
    if old_flights:
        flight_pool = pools["flights"]
        for i, old_fn in enumerate(old_flights):
            # Pick a different flight from pool
            candidates = [f for f in flight_pool
                          if f["flight_number"] != old_fn]
            if candidates:
                new_fn = random.choice(candidates)["flight_number"]
                rep[old_fn] = new_fn

    # --- Product references: fuzzy match to mock DB product names ---
    old_products = ent.get("original_product_refs", [])
    if old_products:
        prod_map = pools.get("product_names", {})
        for old_prod in old_products:
            low = old_prod.lower()
            if low in prod_map:
                rep[old_prod] = prod_map[low]

    return rep


def apply_replacements(text: str, rep: dict[str, str]) -> str:
    """Apply replacements in a single pass, longest-match-first.

    Uses regex alternation with keys sorted by length descending. This prevents
    shorter keys (like "li") from matching inside already-replaced longer values
    (like "Julia").
    """
    if not rep:
        return text

    # Sort keys by length descending, then build a combined regex
    # Escape each key for regex literal matching
    sorted_keys = sorted(rep.keys(), key=len, reverse=True)
    pattern = "|".join(re.escape(k) for k in sorted_keys)

    # Single-pass substitution: for each match (longest first), look up the replacement
    result = re.sub(pattern, lambda m: rep[m.group(0)], text)
    return result


# ===========================================================================
# Variant Generation
# ===========================================================================

def generate_variant(
    original: dict,
    pools: dict,
    used_user_ids: set[str],
    variant_num: int,
) -> dict | None:
    """Generate one augmented variant of a single original prompt."""
    prompt = original["prompt"]
    domain = original["domain"]

    # 1. Extract original entities
    entities = extract_entities(prompt, domain)

    # 2. Pick a new mock DB user
    new_user = _pick_new_user(pools, domain, used_user_ids)
    if new_user is None:
        return None
    used_user_ids.add(new_user["user_id"])

    # 3. Build replacement mapping
    rep = _build_replacement_map(entities, new_user, pools, domain)

    if not rep:
        return None

    # 4. Apply replacements to prompt text
    new_prompt = apply_replacements(prompt, rep)

    # 5. Build output variant (evaluation/tools/source left unchanged per spec)
    variant = deepcopy(original)
    variant["prompt"] = new_prompt
    variant["original_id"] = original.get("id", "")
    variant["variant"] = variant_num
    variant["augmentation"] = {
        "method": "entity_replacement",
        "original_entities": {
            k: v for k, v in entities.items() if v
        },
        "new_user_id": new_user["user_id"],
        "new_name": new_user["name"],
        "replacements_applied": len(rep),
        "replacements": {
            k: v for k, v in sorted(rep.items(), key=lambda x: -len(x[0]))[:20]
        },
    }

    return variant


def augment_set(
    originals: list[dict],
    pools: dict,
    target_count: int,
    variants_per_original: int,
) -> list[dict]:
    """Generate augmented variants for a set of originals.

    Cycles through originals round-robin, generating up to variants_per_original
    variants each (each with a different mock DB user). Stops when target_count
    is reached.
    """
    collected: list[dict] = []
    n_originals = len(originals)

    # Shuffle originals to avoid bias from input order
    shuffled = list(originals)
    random.shuffle(shuffled)

    # Track used mock DB user_ids PER original (reset each original)
    # to ensure different variants of the same prompt use different users

    round_robin_idx = 0
    variants_done = {orig.get("id", i): 0 for i, orig in enumerate(shuffled)}

    while len(collected) < target_count:
        made_progress = False
        for orig in shuffled:
            if len(collected) >= target_count:
                break

            oid = orig.get("id", "")
            if variants_done[oid] >= variants_per_original:
                continue

            # Each original starts with a fresh used_id pool (reset per round)
            # but we track per-original to avoid duplicate mappings
            used = set()
            # Check already-generated variants for this original to avoid dup users
            for existing in collected:
                if existing.get("original_id") == oid:
                    aug = existing.get("augmentation", {})
                    used.add(aug.get("new_user_id", ""))

            variant = generate_variant(orig, pools, used, variants_done[oid] + 1)
            if variant:
                collected.append(variant)
                variants_done[oid] += 1
                made_progress = True

        if not made_progress:
            break  # no more variants possible

    return collected[:target_count]


# ===========================================================================
# Phase 2: LLM Paraphrasing (optional)
# ===========================================================================

_PARAPHRASE_SYSTEM = (
    "Rewrite this customer service request in different words. "
    "ALL specific values (user IDs like U003, order IDs like O005, names, "
    "numbers, zip codes, product names, emails) must remain EXACTLY unchanged. "
    "Only change phrasing, sentence structure, and tone. "
    "Output ONLY the rewritten text."
)


def paraphrase_batch(
    prompts: list[dict],
    api_key: str,
    model: str = "deepseek-chat",
) -> list[dict]:
    """Paraphrase prompt texts using DeepSeek API.

    Each prompt text is sent for rephrasing while entity values are locked.
    Failures are logged but the original text is kept.
    """
    if not api_key:
        print("  WARN: no API key provided, skipping paraphrase")
        return prompts

    try:
        from openai import OpenAI
    except ImportError:
        print("  WARN: openai package not installed, skipping paraphrase")
        return prompts

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    client = OpenAI(api_key=api_key, base_url=base_url)

    for i, pt in enumerate(prompts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _PARAPHRASE_SYSTEM},
                    {"role": "user", "content": f"Rewrite:\n\n{pt['prompt']}"},
                ],
                temperature=0.7,
                max_tokens=800,
            )
            new_text = resp.choices[0].message.content.strip()
            if new_text and len(new_text) > 10:
                pt["prompt"] = new_text
                pt["augmentation"]["method"] += "+llm_paraphrase"
        except Exception as e:
            print(f"  WARN: paraphrase failed for {pt.get('original_id', '?')} "
                  f"v{pt.get('variant', '?')}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(prompts)} paraphrased")

    return prompts


# ===========================================================================
# Phase 3: Train/Val Split
# ===========================================================================

def split_train_val(
    augmented: list[dict],
    val_ratio: float = 0.1,
    seed: int = SEED,
) -> tuple[list[dict], list[dict]]:
    """Split augmented train set 9:1 into train and val.

    IMPORTANT: split at the ORIGINAL prompt level. All variants of the same
    original go to the same split to prevent data leakage.
    """
    # Group variants by original_id
    groups: dict[str, list[dict]] = {}
    for pt in augmented:
        oid = pt.get("original_id", "unknown")
        groups.setdefault(oid, []).append(pt)

    # Shuffle group keys, split 9:1
    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_val = max(1, int(len(group_keys) * val_ratio))
    val_keys = set(group_keys[:n_val])
    train_keys = set(group_keys[n_val:])

    train_pts = [pt for key in train_keys for pt in groups[key]]
    val_pts = [pt for key in val_keys for pt in groups[key]]

    return train_pts, val_pts


# ===========================================================================
# Validation
# ===========================================================================

def validate_entities(prompts: list[dict], sample_size: int) -> dict:
    """Validate that all entity references in augmented prompts exist in mock DB.

    Checks:
    1. All user_ids exist in MockDatabase
    2. All order_ids exist and belong to correct user
    3. Names match user_ids
    4. No tau-bench fictional entities remain
    """
    from trainable_openclaw.agent.tau_bench_tools.mock_db import seed

    retail_db = seed("retail")
    airline_db = seed("airline")

    # Build lookup indexes
    retail_users = {u["user_id"]: u for u in retail_db["users"]}
    airline_users = {u["user_id"]: u for u in airline_db["users"]}
    retail_orders = {o["order_id"]: o for o in retail_db["orders"]}
    airline_res = {r["reservation_id"]: r for r in airline_db["reservations"]}
    retail_products = {p["product_id"]: p for p in retail_db["products"]}
    flights = {f["flight_number"]: f for f in airline_db["flights"].values()}

    sample = random.sample(prompts, min(sample_size, len(prompts)))
    results = []

    for pt in sample:
        domain = pt.get("domain", "retail")
        prompt = pt["prompt"]
        aug = pt.get("augmentation", {})
        new_uid = aug.get("new_user_id", "")
        errors = []

        # Check 1: user_id exists in mock DB
        if domain == "retail":
            if new_uid not in retail_users:
                errors.append(f"user_id {new_uid} not in retail DB")
        else:
            if new_uid not in airline_users:
                errors.append(f"user_id {new_uid} not in airline DB")

        # Check 2: order IDs exist and belong to user
        order_ids_in_prompt = re.findall(r'\b(O\d{3})\b', prompt)
        for oid in order_ids_in_prompt:
            if oid not in retail_orders:
                errors.append(f"order_id {oid} not in retail DB")
            elif domain == "retail" and retail_orders[oid]["user_id"] != new_uid:
                errors.append(f"order_id {oid} belongs to "
                              f"{retail_orders[oid]['user_id']}, not {new_uid}")

        # Check 3: reservation IDs exist and belong to user
        res_ids_in_prompt = re.findall(r'\b(RA\d{3})\b', prompt)
        for rid in res_ids_in_prompt:
            if rid not in airline_res:
                errors.append(f"reservation {rid} not in airline DB")
            elif domain == "airline" and airline_res[rid]["user_id"] != new_uid:
                errors.append(f"res {rid} belongs to "
                              f"{airline_res[rid]['user_id']}, not {new_uid}")

        # Check 4: No tau-bench fictional entities remain
        tau_uid = re.findall(r'\b([a-z]+_[a-z]+_\d{3,5})\b', prompt)
        if tau_uid:
            errors.append(f"tau-bench user_id remains: {tau_uid}")
        tau_email = re.findall(r'[\w.+-]+@example\.com', prompt)
        if tau_email:
            errors.append(f"tau-bench email remains: {tau_email}")

        # Check 5: Name matches user_id (only if prompt originally had a name)
        # Many prompts use uid-only format ("You are mia_garcia_4516") — those are
        # valid without a name mention, since the agent uses get_user_details(uid).
        if new_uid in retail_users:
            expected_name = retail_users[new_uid]["name"]
            expected_first = retail_users[new_uid]["first_name"]
            # Check if prompt text contains ANY person name (original or replacement)
            # by looking for two consecutive capitalized words that look like a name
            name_in_prompt = re.search(
                r'\b([A-Z][a-z]+(?:\'[A-Z][a-z]+)?)\s+([A-Z][a-z]+(?:\'[A-Z][a-z]+)?)\b',
                prompt)
            if name_in_prompt:
                found_name = f"{name_in_prompt.group(1)} {name_in_prompt.group(2)}"
                # Skip city/state names
                if name_in_prompt.group(1) not in ("New", "San", "Los", "St", "El", "Big"):
                    if expected_name.lower() not in prompt.lower():
                        errors.append(
                            f"name mismatch: found '{found_name}', expected '{expected_name}'")
        elif new_uid in airline_users:
            expected_name = airline_users[new_uid]["name"]
            name_in_prompt = re.search(
                r'\b([A-Z][a-z]+(?:\'[A-Z][a-z]+)?)\s+([A-Z][a-z]+(?:\'[A-Z][a-z]+)?)\b',
                prompt)
            if name_in_prompt:
                found_name = f"{name_in_prompt.group(1)} {name_in_prompt.group(2)}"
                if name_in_prompt.group(1) not in ("New", "San", "Los", "St", "El", "Big"):
                    if expected_name.lower() not in prompt.lower():
                        errors.append(
                            f"name mismatch: found '{found_name}', expected '{expected_name}'")

        results.append({
            "id": pt.get("original_id", "?"),
            "variant": pt.get("variant", "?"),
            "domain": domain,
            "new_user_id": new_uid,
            "errors": errors,
            "passed": len(errors) == 0,
        })

    passed = sum(1 for r in results if r["passed"])
    return {
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 3) if results else 0,
        "results": results,
    }


# ===========================================================================
# I/O helpers
# ===========================================================================

def load_jsonl(path: Path) -> list[dict]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line.strip()))
    return prompts


def write_jsonl(prompts: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


# ===========================================================================
# Dry-run / introspection
# ===========================================================================

def debug_entities(prompts: list[dict], limit: int = 10):
    """Print entity extraction results."""
    for i, pt in enumerate(prompts[:limit]):
        ent = extract_entities(pt["prompt"], pt["domain"])
        print(f"\n[{i}] {pt['domain']} | {pt.get('id', '?')}")
        print(f"  prompt[:150]: {pt['prompt'][:150]}")
        for key in ["original_user_id_str", "original_full_name",
                     "original_first_name", "original_last_name",
                     "original_zip", "original_city", "original_state",
                     "original_email", "original_order_refs",
                     "original_reservation_refs", "original_flight_refs",
                     "original_product_refs"]:
            val = ent.get(key)
            if val:
                print(f"  {key}: {val}")


# ===========================================================================
# Main CLI
# ===========================================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Augment tau-bench prompts")
    ap.add_argument("--train-target", type=int, default=500,
                    help="Target count for augmented train (before split)")
    ap.add_argument("--test-target", type=int, default=50,
                    help="Target count for augmented test")
    ap.add_argument("--train-variants", type=int, default=4,
                    help="Max variants per original train prompt")
    ap.add_argument("--test-variants", type=int, default=2,
                    help="Max variants per original test prompt")
    ap.add_argument("--llm-paraphrase", action="store_true",
                    help="Use DeepSeek to rephrase prompts (Phase 2)")
    ap.add_argument("--validate", type=int, default=0,
                    help="Run entity validation on N random samples")
    ap.add_argument("--val-split", type=float, default=0,
                    help="Val split ratio from augmented train (e.g. 0.1). 0 = no split.")
    ap.add_argument("--model", type=str, default="deepseek-chat",
                    help="Model for LLM paraphrasing")
    ap.add_argument("--dry-run", action="store_true",
                    help="Extract entities only, print results without generating")
    return ap.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  PROMPT AUGMENTATION — Entity Replacement (Phase 1-3)")
    print("=" * 60)

    # ---- Phase 1: Entity pools ----
    print("\n[Phase 1] Building entity pools from MockDatabase seed data...")
    pools = build_entity_pools()
    print(f"  Retail users: {len(pools['retail_users'])}")
    print(f"  Retail orders: {len(pools['retail_orders'])}")
    print(f"  Airline users: {len(pools['airline_users'])}")
    print(f"  Airline reservations: {len(pools['airline_reservations'])}")
    print(f"  Flights: {len(pools['flights'])}")

    # ---- Load originals ----
    print(f"\n  Loading original prompts...")
    train_orig = load_jsonl(TRAIN_PROMPTS)
    test_orig = load_jsonl(TEST_PROMPTS)
    print(f"  Train: {len(train_orig)} | Test: {len(test_orig)}")

    if args.dry_run:
        print("\n[DRY RUN] Entity extraction samples:")
        print("--- TRAIN ---")
        debug_entities(train_orig, 5)
        print("\n--- TEST ---")
        debug_entities(test_orig, 5)
        return

    # ---- Generate variants ----
    print(f"\n  Generating variants...")
    print(f"  Train target: {args.train_target} (max {args.train_variants}/orig)")
    print(f"  Test target:  {args.test_target} (max {args.test_variants}/orig)")

    train_aug = augment_set(train_orig, pools, args.train_target,
                            args.train_variants)
    test_aug = augment_set(test_orig, pools, args.test_target,
                           args.test_variants)

    # Task ID leakage check
    train_oid_set = {p["original_id"] for p in train_aug}
    test_oid_set = {p["original_id"] for p in test_aug}
    leak = train_oid_set & test_oid_set

    print(f"  Train generated: {len(train_aug)}")
    print(f"  Test generated:  {len(test_aug)}")
    print(f"  Train originals used: {len(train_oid_set)}")
    print(f"  Test originals used:  {len(test_oid_set)}")
    print(f"  Task ID leakage: {len(leak)}" +
          (" *** WARNING! ***" if leak else " OK"))

    # Show one sample variant
    if train_aug:
        s = train_aug[0]
        aug = s.get("augmentation", {})
        print(f"\n  Sample variant:")
        print(f"    original_id={s['original_id']}  variant={s['variant']}")
        print(f"    domain={s['domain']}")
        print(f"    new_user={aug.get('new_user_id')} ({aug.get('new_name')})")
        print(f"    replacements={aug.get('replacements_applied')}")
        print(f"    prompt[:200]: {s['prompt'][:200]}")

    # ---- Phase 2: LLM Paraphrasing (optional) ----
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if args.llm_paraphrase:
        if api_key:
            print(f"\n[Phase 2] LLM paraphrasing (model={args.model})...")
            train_aug = paraphrase_batch(train_aug, api_key, args.model)
            test_aug = paraphrase_batch(test_aug, api_key, args.model)
            print(f"  Paraphrasing complete.")
        else:
            print(f"\n[Phase 2] LLM paraphrasing SKIPPED (no DEEPSEEK_API_KEY)")

    # ---- Phase 3: Train/Val split (optional, controlled by --val-split) ----
    val_final: list[dict] = []
    if args.val_split > 0:
        vratio = args.val_split
        print(f"\n[Phase 3] Splitting train augmented {1-vratio:.0f}:{vratio:.0f} → train + val...")
        train_final, val_final = split_train_val(train_aug, val_ratio=vratio)
        print(f"  Train: {len(train_final)}")
        print(f"  Val:   {len(val_final)}")
        print(f"  Test:  {len(test_aug)}")
        write_jsonl(val_final, VAL_OUT)
        print(f"  -> {VAL_OUT} ({len(val_final)} entries)")
    else:
        train_final = train_aug
        print(f"\n[Phase 3] No val split (--val-split not set). Writing full train set.")
        print(f"  Train: {len(train_final)}")
        print(f"  Test:  {len(test_aug)}")

    # ---- Write outputs ----
    print(f"\n  Writing output files...")
    write_jsonl(train_final, TRAIN_OUT)
    write_jsonl(test_aug, TEST_OUT)
    print(f"  -> {TRAIN_OUT} ({len(train_final)} entries)")
    print(f"  -> {TEST_OUT} ({len(test_aug)} entries)")

    # ---- Validation ----
    if args.validate > 0:
        print(f"\n[Validate] Checking {args.validate} random samples for entity correctness...")
        all_prompts = train_final + val_final + test_aug
        result = validate_entities(all_prompts, args.validate)
        print(f"  Results: {result['passed']}/{result['total']} passed "
              f"({result['pass_rate']:.1%})")
        for r in result.get("results", []):
            status = "PASS" if r["passed"] else "FAIL"
            details = "; ".join(r["errors"]) if r["errors"] else ""
            print(f"    [{status}] {r['id']} v{r['variant']} ({r['domain']}) "
                  f"uid={r['new_user_id']}{' — ' + details if details else ''}")

    # ---- Summary ----
    print(f"\nDone.")
    print(f"  Train: {len(train_final)}/{args.train_target}")
    if val_final:
        print(f"  Val:   {len(val_final)}")
    print(f"  Test:  {len(test_aug)}/{args.test_target}")


if __name__ == "__main__":
    main()
