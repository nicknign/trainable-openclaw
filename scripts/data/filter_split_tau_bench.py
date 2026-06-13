#!/usr/bin/env python3
"""
Filter and split tau-bench samples into train/test sets + GRPO prompts.

Usage:
    python scripts/data/filter_split_tau_bench.py

Reads:  data/tau_bench/all_samples.json
        data/tau_bench/raw/tasks_airline.json
        data/tau_bench/raw/tasks_retail.json
        data/tau_bench/raw/split_tasks_airline.json
        data/tau_bench/raw/split_tasks_retail.json

Writes: data/tau_bench/train.jsonl
        data/tau_bench/test.jsonl
        data/tau_bench/grpo_prompts.jsonl
"""

import json
import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_DIR, 'data', 'tau_bench', 'raw')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'data', 'tau_bench')
ALL_SAMPLES_PATH = os.path.join(OUTPUT_DIR, 'all_samples.json')

SEED = 42


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(samples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  Saved {len(samples)} samples to {path}")


def filter_by_reward(samples, min_reward=0.5):
    """Filter samples by outcome.reward.final >= min_reward."""
    result = [s for s in samples if s['outcome']['reward']['final'] >= min_reward]
    removed = len(samples) - len(result)
    if removed:
        print(f"  Filtered {removed} samples with reward < {min_reward}")
    return result


def _composite_key(sample):
    """Return a (domain, task_id) key that uniquely identifies a task."""
    source = sample['source']
    domain = 'airline' if 'airline' in source else 'retail'
    return (domain, sample['task_id'])


def split_by_task_id(samples, train_ratio=0.8):
    """Split samples ensuring no (domain, task_id) appears in both train and test.

    Returns (train, test) lists.
    """
    groups = {}
    for s in samples:
        key = _composite_key(s)
        groups.setdefault(key, []).append(s)

    keys = sorted(groups.keys())
    rng = random.Random(SEED)
    rng.shuffle(keys)

    split_idx = int(len(keys) * train_ratio)
    train_keys = set(keys[:split_idx])
    test_keys = set(keys[split_idx:])

    train = []
    test = []
    for key, group in groups.items():
        if key in train_keys:
            train.extend(group)
        else:
            test.extend(group)

    # Verify no leakage
    train_tids = {_composite_key(s) for s in train}
    test_tids = {_composite_key(s) for s in test}
    overlap = train_tids & test_tids
    if overlap:
        print(f"  WARNING: {len(overlap)} composite task_ids in both train and test")

    return train, test


def split_using_tau_bench_splits(samples):
    """Use the official tau-bench train/test splits to partition samples.

    Historical trajectories have task_id matching the task definitions.
    We use split_tasks_airline.json and split_tasks_retail.json.
    """
    airline_split = load_json(os.path.join(RAW_DIR, 'split_tasks_airline.json'))
    retail_split = load_json(os.path.join(RAW_DIR, 'split_tasks_retail.json'))

    airline_train = set(airline_split.get('train', []))
    airline_test = set(airline_split.get('test', []))
    retail_train = set(retail_split.get('train', []))
    retail_test = set(retail_split.get('test', []))

    train = []
    test = []
    unassigned = []

    for s in samples:
        tid = s['task_id']
        source = s['source']

        if source in ('taubench_airline', 'apigen_airline'):
            if tid in airline_train:
                train.append(s)
            elif tid in airline_test:
                test.append(s)
            else:
                unassigned.append(s)
        elif source in ('taubench_retail', 'apigen_retail'):
            if tid in retail_train:
                train.append(s)
            elif tid in retail_test:
                test.append(s)
            else:
                unassigned.append(s)
        else:
            unassigned.append(s)

    if unassigned:
        print(f"  {len(unassigned)} samples not in official split — adding to train")
        train.extend(unassigned)

    return train, test


def _verify_no_overlap(train, test):
    """Verify no (domain, task_id) appears in both train and test."""
    train_keys = {_composite_key(s) for s in train}
    test_keys = {_composite_key(s) for s in test}
    overlap = train_keys & test_keys
    if overlap:
        print(f"  ERROR: {len(overlap)} composite task_ids in both train and test!")
        for key in sorted(overlap)[:10]:
            print(f"    {key}")
    else:
        print("  No task_id overlap (domain-aware) - OK")
    return len(overlap) == 0


def build_grpo_prompts(tasks_dir, airline_train_ids, airline_test_ids,
                       retail_train_ids, retail_test_ids):
    """Build GRPO prompts from task definitions.

    Returns (train_prompts, test_prompts) suitable for GRPO.
    """
    airline_tasks = load_json(os.path.join(RAW_DIR, 'tasks_airline.json'))
    retail_tasks = load_json(os.path.join(RAW_DIR, 'tasks_retail.json'))

    # Load tool definitions
    sys.path.insert(0, SCRIPT_DIR)
    from convert_tau_bench import DOMAIN_TOOLS

    train_prompts = []
    test_prompts = []

    for task in airline_tasks:
        tid = str(task['id'])
        domain = 'airline'
        prompt = _task_to_prompt(task, domain, DOMAIN_TOOLS.get(domain, []))

        if tid in airline_train_ids:
            train_prompts.append(prompt)
        elif tid in airline_test_ids:
            test_prompts.append(prompt)

    for task in retail_tasks:
        tid = str(task['id'])
        domain = 'retail'
        prompt = _task_to_prompt(task, domain, DOMAIN_TOOLS.get(domain, []))

        if tid in retail_train_ids:
            train_prompts.append(prompt)
        elif tid in retail_test_ids:
            test_prompts.append(prompt)

    return train_prompts, test_prompts


def _task_to_prompt(task, domain, tools):
    """Convert a task definition to a GRPO prompt."""
    instructions = task.get('user_scenario', {}).get('instructions', {})
    reason = instructions.get('reason_for_call', '')
    known_info = instructions.get('known_info', '')
    evaluation = task.get('evaluation_criteria', {})
    nl_assertions = evaluation.get('nl_assertions', [])
    reward_basis = evaluation.get('reward_basis', [])
    description = task.get('description', {})

    user_prompt = known_info + '\n\n' + reason if known_info else reason
    user_prompt = user_prompt.strip()

    return {
        'id': f"{domain}_task_{task['id']}",
        'source': f'taubench_{domain}',
        'task_id': str(task['id']),
        'domain': domain,
        'prompt': user_prompt,
        'evaluation': {
            'purpose': description.get('purpose', ''),
            'nl_assertions': nl_assertions,
            'reward_basis': reward_basis,
        },
        'tools': tools,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Filtering and splitting tau-bench samples ===\n")

    if not os.path.exists(ALL_SAMPLES_PATH):
        print(f"ERROR: {ALL_SAMPLES_PATH} not found. Run convert_tau_bench.py first.")
        return 1

    samples = load_json(ALL_SAMPLES_PATH)
    print(f"Loaded {len(samples)} samples")

    # Step 1: Filter by reward >= 0.5
    print("\n1. Filter by reward >= 0.5 ...")
    filtered = filter_by_reward(samples, min_reward=0.5)

    # Step 2: Split train/test
    print("\n2. Splitting train/test ...")

    # Use official tau-bench splits when available
    airline_split_path = os.path.join(RAW_DIR, 'split_tasks_airline.json')
    retail_split_path = os.path.join(RAW_DIR, 'split_tasks_retail.json')

    if os.path.exists(airline_split_path) and os.path.exists(retail_split_path):
        print("  Using official tau-bench train/test splits")
        train, test = split_using_tau_bench_splits(filtered)
    else:
        print("  Using random task_id-based split (80/20)")
        train, test = split_by_task_id(filtered, train_ratio=0.8)

    print(f"  Train: {len(train)} samples ({len({_composite_key(s) for s in train})} unique tasks)")
    print(f"  Test:  {len(test)} samples ({len({_composite_key(s) for s in test})} unique tasks)")

    # Verify no task_id overlap (domain-aware)
    _verify_no_overlap(train, test)

    # Step 3: Save
    print("\n3. Saving ...")
    save_jsonl(train, os.path.join(OUTPUT_DIR, 'train.jsonl'))
    save_jsonl(test, os.path.join(OUTPUT_DIR, 'test.jsonl'))

    # Step 4: Build GRPO prompts
    print("\n4. Building GRPO prompts ...")
    airline_split = load_json(airline_split_path) if os.path.exists(airline_split_path) else {'train': [], 'test': []}
    retail_split = load_json(retail_split_path) if os.path.exists(retail_split_path) else {'train': [], 'test': []}

    train_prompts, test_prompts = build_grpo_prompts(
        RAW_DIR,
        set(airline_split.get('train', [])),
        set(airline_split.get('test', [])),
        set(retail_split.get('train', [])),
        set(retail_split.get('test', [])),
    )

    print(f"  Train prompts: {len(train_prompts)}")
    print(f"  Test prompts:  {len(test_prompts)}")

    save_jsonl(train_prompts + test_prompts, os.path.join(OUTPUT_DIR, 'grpo_prompts.jsonl'))

    print("\n=== Done ===")
    return 0


if __name__ == '__main__':
    sys.exit(main())
