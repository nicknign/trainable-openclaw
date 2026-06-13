#!/usr/bin/env python3
"""
Validate tau-bench converted data integrity.

Runs 13+ automated checks on train.jsonl and test.jsonl.
Exit code 1 if any check fails.
"""

import json
import os
import sys
import uuid

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'tau_bench')

VALID_SOURCES = {
    'taubench_airline', 'taubench_retail',
    'apigen_airline', 'apigen_retail',
}


def load_jsonl(path):
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


class Checker:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name, condition, detail=''):
        if condition:
            self.passed += 1
            print(f"  PASS: {name}")
        else:
            self.failed += 1
            msg = f"  FAIL: {name}"
            if detail:
                msg += f" — {detail}"
            print(msg)
            self.errors.append(msg)


def validate_samples(samples, label, checker):
    """Run checks 1-12 on a set of samples."""
    print(f"\n--- {label} ({len(samples)} samples) ---")

    # 1: All samples have valid UUID id
    invalid_ids = []
    for s in samples:
        sid = s.get('id', '')
        try:
            uuid.UUID(sid)
        except (ValueError, AttributeError):
            invalid_ids.append(sid)
    checker.check(
        '1. Valid UUID id',
        len(invalid_ids) == 0,
        f'{len(invalid_ids)} invalid: {invalid_ids[:3]}...' if invalid_ids else '',
    )

    # 2: All samples have valid source field
    invalid_sources = [s.get('source', 'MISSING') for s in samples
                       if s.get('source') not in VALID_SOURCES]
    checker.check(
        '2. Valid source field',
        len(invalid_sources) == 0,
        f'{len(invalid_sources)} invalid: {invalid_sources[:3]}...' if invalid_sources else '',
    )

    # 3: context.system_prompt is non-empty string
    bad_sp = sum(1 for s in samples
                 if not isinstance(s.get('context', {}).get('system_prompt'), str)
                 or not s['context']['system_prompt'].strip())
    checker.check('3. context.system_prompt non-empty', bad_sp == 0,
                  f'{bad_sp} samples with empty/missing system_prompt' if bad_sp else '')

    # 4: context.tools is non-empty list with valid JSON Schema
    bad_tools = 0
    for s in samples:
        tools = s.get('context', {}).get('tools', [])
        if not isinstance(tools, list) or len(tools) == 0:
            bad_tools += 1
            continue
        for t in tools:
            if not isinstance(t, dict):
                bad_tools += 1
                break
            if t.get('type') != 'function':
                bad_tools += 1
                break
            fn = t.get('function', {})
            if not fn.get('name'):
                bad_tools += 1
                break
    checker.check('4. context.tools valid non-empty', bad_tools == 0,
                  f'{bad_tools} samples with invalid tools' if bad_tools else '')

    # 5: context.user_request is non-empty string
    bad_ur = sum(1 for s in samples
                 if not isinstance(s.get('context', {}).get('user_request'), str)
                 or not s['context']['user_request'].strip())
    checker.check('5. context.user_request non-empty', bad_ur == 0,
                  f'{bad_ur} samples with empty/missing user_request' if bad_ur else '')

    # 6: trajectory is non-empty list
    bad_traj = sum(1 for s in samples
                   if not isinstance(s.get('trajectory'), list) or len(s['trajectory']) == 0)
    checker.check('6. trajectory non-empty', bad_traj == 0,
                  f'{bad_traj} samples with empty trajectory' if bad_traj else '')

    # 7: First trajectory message has role "assistant" or "user"
    bad_first = 0
    for s in samples:
        traj = s.get('trajectory', [])
        if traj:
            first_role = traj[0].get('role')
            if first_role not in ('assistant', 'user'):
                bad_first += 1
    checker.check('7. First trajectory role is assistant/user', bad_first == 0,
                  f'{bad_first} samples with wrong first role' if bad_first else '')

    # 8: All tool_calls have valid {name, arguments} structure
    bad_tc = 0
    for s in samples:
        for msg in s.get('trajectory', []):
            for tc in msg.get('tool_calls', []):
                fn = tc.get('function', {})
                if not fn.get('name') or 'arguments' not in fn:
                    bad_tc += 1
                    break
    checker.check('8. All tool_calls have valid {name, arguments}', bad_tc == 0,
                  f'{bad_tc} tool_calls with invalid structure' if bad_tc else '')

    # 9: All tool responses have valid {tool_call_id, name, content}
    bad_tr = 0
    for s in samples:
        for msg in s.get('trajectory', []):
            if msg.get('role') == 'tool':
                if not msg.get('tool_call_id') or not msg.get('name'):
                    bad_tr += 1
    checker.check('9. All tool responses have {tool_call_id, name}', bad_tr == 0,
                  f'{bad_tr} tool responses missing fields' if bad_tr else '')

    # 11: outcome.reward.final between 0 and 1
    bad_reward = sum(1 for s in samples
                     if not (0.0 <= s['outcome']['reward']['final'] <= 1.0))
    checker.check('11. outcome.reward.final in [0, 1]', bad_reward == 0,
                  f'{bad_reward} samples with out-of-range final reward' if bad_reward else '')

    # 12: outcome.reward has all 4 fields
    required_fields = {'layer1', 'layer2', 'layer3', 'final'}
    bad_fields = sum(1 for s in samples
                     if required_fields - set(s['outcome']['reward'].keys()))
    checker.check('12. outcome.reward has all 4 layers', bad_fields == 0,
                  f'{bad_fields} samples missing reward fields' if bad_fields else '')

    # 13: outcome.reward_weights sum to approx 1.0
    bad_weights = sum(1 for s in samples
                      if abs(sum(s['outcome']['reward_weights'].values()) - 1.0) > 0.01)
    checker.check('13. reward_weights sum to ~1.0', bad_weights == 0,
                  f'{bad_weights} samples with bad weight sum' if bad_weights else '')


def main():
    print("=== Validating tau-bench data ===\n")

    checker = Checker()

    train_path = os.path.join(DATA_DIR, 'train.jsonl')
    test_path = os.path.join(DATA_DIR, 'test.jsonl')

    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found")
        return 1
    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found")
        return 1

    train = load_jsonl(train_path)
    test = load_jsonl(test_path)

    validate_samples(train, 'Train', checker)
    validate_samples(test, 'Test', checker)

    # 10: No domain+task_id overlap between train and test
    print(f"\n--- Cross-set checks ---")
    def _domain_key(s):
        source = s.get('source', '')
        domain = 'airline' if 'airline' in source else 'retail'
        return (domain, s['task_id'])
    train_ids = {_domain_key(s) for s in train}
    test_ids = {_domain_key(s) for s in test}
    overlap = train_ids & test_ids
    checker.check('10. No domain+task_id overlap train/test', len(overlap) == 0,
                  f'{len(overlap)} overlapping (domain,task_id): {sorted(overlap)[:5]}...' if overlap else '')

    # Summary
    total = checker.passed + checker.failed
    print(f"\n{'='*50}")
    print(f"Results: {checker.passed}/{total} PASSED, {checker.failed}/{total} FAILED")

    if checker.failed > 0:
        print("\nFailures:")
        for err in checker.errors:
            print(f"  {err}")

    return 1 if checker.failed > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
