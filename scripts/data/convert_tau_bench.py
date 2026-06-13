#!/usr/bin/env python3
"""
Convert tau-bench raw data to the project unified training sample format.

Handles:
  - Historical trajectories (GPT-4o, Sonnet) for airline and retail
  - APIGen tau-bench samples

Output: data/tau_bench/all_samples.json
"""

import json
import os
import sys
import uuid

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_DIR, 'data', 'tau_bench', 'raw')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'data', 'tau_bench')

AIRLINE_TOOL_NAMES = {
    'book_reservation', 'cancel_reservation', 'get_reservation_details',
    'search_direct_flight', 'search_onestop_flight', 'get_flight_status',
    'update_reservation_baggages', 'update_reservation_flights',
    'update_reservation_passengers', 'send_certificate',
    'list_all_airports', 'get_user_details', 'calculate', 'think',
    'transfer_to_human_agents',
}

RETAIL_TOOL_NAMES = {
    'cancel_pending_order', 'exchange_delivered_order_items',
    'find_user_id_by_name_zip', 'find_user_id_by_email',
    'get_order_details', 'get_product_details', 'get_item_details',
    'get_user_details', 'list_all_product_types',
    'modify_pending_order_address', 'modify_pending_order_items',
    'modify_pending_order_payment', 'modify_user_address',
    'return_delivered_order_items', 'calculate', 'think',
    'transfer_to_human_agents',
}


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _classify_tools(tools):
    """Classify a list of tool dicts as airline, retail, or mixed."""
    names = {t['function']['name'] for t in tools}
    n_airline = len(names & AIRLINE_TOOL_NAMES)
    n_retail = len(names & RETAIL_TOOL_NAMES)
    if n_airline > n_retail:
        return 'airline'
    if n_retail > n_airline:
        return 'retail'
    return 'unknown'


def _load_cached_tool_defs():
    """Load tool definitions from APIGen data, keyed by domain.

    Falls back to mock tool schemas for domains not found in APIGen data.
    """
    cache = {}
    apigen_path = os.path.join(RAW_DIR, 'apigen_sample.json')
    if os.path.exists(apigen_path):
        data = load_json(apigen_path)
        for sample in data:
            tools = sample.get('tools', [])
            domain = _classify_tools(tools)
            if domain not in cache and domain != 'unknown':
                cache[domain] = tools
            if len(cache) >= 2:
                break

    # Fallback: generate from mock tools for any missing domain
    sys.path.insert(0, PROJECT_DIR)
    try:
        from trainable_openclaw.agent.tau_bench_tools.registry import register_tau_bench_tools
        for domain in ('airline', 'retail'):
            if domain not in cache:
                mock_tools = register_tau_bench_tools(domain)
                cache[domain] = [t.to_schema() for t in mock_tools]
    except ImportError:
        pass

    return cache


DOMAIN_TOOLS = _load_cached_tool_defs()


def _extract_tools_from_traj(traj):
    """Build minimal tool definitions from tool calls and responses in a trajectory."""
    seen = set()
    tools = []
    for msg in traj:
        for tc in (msg.get('tool_calls') or []):
            if 'function' in tc:
                name = tc['function']['name']
                if name and name not in seen:
                    seen.add(name)
                    tools.append({
                        'type': 'function',
                        'function': {'name': name, 'description': '', 'parameters': '{}'},
                    })
        if msg.get('role') == 'tool':
            name = msg.get('name', '')
            if name and name not in seen:
                seen.add(name)
                tools.append({
                    'type': 'function',
                    'function': {'name': name, 'description': '', 'parameters': '{}'},
                })
    return tools


def convert_historical_trajectory(traj_obj, source):
    """Convert one GPT-4o/Sonnet trajectory to unified format."""
    task_id = str(traj_obj.get('task_id', 'unknown'))
    trial = traj_obj.get('trial', 0)
    reward = float(traj_obj.get('reward', 0.0))
    traj = traj_obj.get('traj', [])

    domain = 'airline' if 'airline' in source else 'retail'
    tools = DOMAIN_TOOLS.get(domain, [])
    if not tools:
        tools = _extract_tools_from_traj(traj)

    context = _extract_context(traj, tools)
    normalized_traj = _normalize_messages(traj)

    outcome = {
        'task_completed': reward >= 1.0,
        'reward': {
            'layer1': reward,
            'layer2': reward,
            'layer3': reward,
            'final': reward,
        },
        'reward_weights': {'layer1': 0.5, 'layer2': 0.3, 'layer3': 0.2},
    }

    return {
        'id': str(uuid.uuid4()),
        'source': source,
        'task_id': task_id,
        'trial': trial,
        'context': context,
        'trajectory': normalized_traj,
        'outcome': outcome,
    }


def convert_apigen_sample(sample, source):
    """Convert one APIGen sample to unified format."""
    msgs = sample.get('messages', [])
    tools = sample.get('tools', [])

    context = _extract_context(msgs, tools)
    normalized_traj = _normalize_messages(msgs)

    task_id = str(sample.get('task_id', sample.get('id', 'unknown')))

    outcome = _compute_reward(normalized_traj, None)

    return {
        'id': str(uuid.uuid4()),
        'source': source,
        'task_id': task_id,
        'context': context,
        'trajectory': normalized_traj,
        'outcome': outcome,
    }


def _extract_context(trajectory, tools):
    """Extract system prompt, tools, and first user request from trajectory."""
    system_prompt = ''
    user_request = ''

    for msg in trajectory:
        role = msg.get('role')
        if role == 'system' and not system_prompt:
            system_prompt = msg.get('content') or ''
        elif role == 'user' and not user_request:
            user_request = msg.get('content') or ''
        if system_prompt and user_request:
            break

    return {
        'system_prompt': system_prompt,
        'tools': tools,
        'user_request': user_request,
    }


def _normalize_messages(msgs):
    """Normalize message list to unified trajectory format.

    - Strips system messages (they live in context.system_prompt)
    - Ensures tool_call `id` is non-None
    - Synthesizes missing `name`/`tool_call_id` on tool responses
    """
    result = []
    for msg in msgs:
        role = msg.get('role')
        if role == 'system':
            continue

        normalized = {'role': role}

        if role == 'user':
            normalized['content'] = msg.get('content') or ''

        elif role == 'assistant':
            content = msg.get('content')
            if content is not None:
                normalized['content'] = content

            tool_calls = msg.get('tool_calls')
            if tool_calls:
                normalized['tool_calls'] = []
                for tc in tool_calls:
                    tc_id = tc.get('id')
                    if not tc_id:
                        tc_id = str(uuid.uuid4())
                    normalized['tool_calls'].append({
                        'id': tc_id,
                        'type': tc.get('type', 'function'),
                        'function': {
                            'name': tc['function']['name'],
                            'arguments': tc['function']['arguments'],
                        },
                    })

        elif role == 'tool':
            normalized['content'] = msg.get('content') or ''
            normalized['name'] = msg.get('name', '')
            normalized['tool_call_id'] = msg.get('tool_call_id', '')

            # Synthesize missing name/tool_call_id from preceding assistant
            if not normalized.get('name') or not normalized.get('tool_call_id'):
                for prev in reversed(result):
                    if prev.get('role') == 'assistant' and prev.get('tool_calls'):
                        last_tc = prev['tool_calls'][-1]
                        if not normalized.get('name'):
                            normalized['name'] = last_tc['function']['name']
                        if not normalized.get('tool_call_id'):
                            normalized['tool_call_id'] = last_tc['id']
                        break

        result.append(normalized)

    return result


def _compute_reward(trajectory, _task_gt):
    """Compute Layer 1 reward from trajectory for APIGen data (no explicit reward)."""
    if not trajectory:
        return {
            'task_completed': False,
            'reward': {'layer1': 0.0, 'layer2': 0.0, 'layer3': 0.0, 'final': 0.0},
            'reward_weights': {'layer1': 0.5, 'layer2': 0.3, 'layer3': 0.2},
        }

    score = 0.5

    has_tool_calls = any(
        m.get('role') == 'assistant' and m.get('tool_calls') for m in trajectory
    )
    if has_tool_calls:
        score += 0.3

    last_role = trajectory[-1].get('role')
    if last_role == 'assistant':
        score += 0.2

    score = min(score, 1.0)

    return {
        'task_completed': score >= 0.8,
        'reward': {
            'layer1': score,
            'layer2': score,
            'layer3': score,
            'final': score,
        },
        'reward_weights': {'layer1': 0.5, 'layer2': 0.3, 'layer3': 0.2},
    }


def _is_broken(sample):
    """Return True if the sample is broken/incomplete."""
    if not sample.get('trajectory'):
        return True
    if not sample['context'].get('user_request'):
        return True
    return False


def load_all():
    """Load and convert all raw data."""
    all_samples = []

    historical_files = [
        ('gpt-4o-airline.json', 'taubench_airline'),
        ('gpt-4o-retail.json', 'taubench_retail'),
        ('sonnet-35-new-airline.json', 'taubench_airline'),
        ('sonnet-35-new-retail.json', 'taubench_retail'),
    ]

    for fname, source in historical_files:
        filepath = os.path.join(RAW_DIR, fname)
        if not os.path.exists(filepath):
            print(f"  SKIP {fname}: file not found")
            continue
        print(f"Loading {fname} ...")
        data = load_json(filepath)
        count = 0
        for traj_obj in data:
            try:
                sample = convert_historical_trajectory(traj_obj, source)
                if not _is_broken(sample):
                    all_samples.append(sample)
                    count += 1
            except Exception as e:
                print(f"  Skipping trajectory task_id={traj_obj.get('task_id', '?')} (error: {e})")
        print(f"  -> {count} valid samples")

    apigen_path = os.path.join(RAW_DIR, 'apigen_sample.json')
    if os.path.exists(apigen_path):
        print("Loading apigen_sample.json ...")
        data = load_json(apigen_path)
        count = 0
        for sample in data:
            try:
                tools = sample.get('tools', [])
                domain = _classify_tools(tools)
                if domain == 'unknown':
                    # Fallback: check the system prompt content
                    sys_msg = sample['messages'][0] if sample.get('messages') else {}
                    content = sys_msg.get('content', '') if sys_msg.get('role') == 'system' else ''
                    if 'airline' in content.lower() or 'flight' in content.lower():
                        domain = 'airline'
                    elif 'order' in content.lower() or 'product' in content.lower():
                        domain = 'retail'
                    else:
                        domain = 'retail'

                source = f'apigen_{domain}'
                result = convert_apigen_sample(sample, source)
                if not _is_broken(result):
                    all_samples.append(result)
                    count += 1
            except Exception as e:
                print(f"  Skipping APIGen sample (error: {e})")
        print(f"  -> {count} valid samples")

    return all_samples


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Converting tau-bench data to unified format ===\n")
    samples = load_all()

    print(f"\nTotal valid samples: {len(samples)}")

    # Reward breakdown
    rewards = [s['outcome']['reward']['final'] for s in samples]
    print(f"Reward 0.0: {rewards.count(0.0)}")
    print(f"Reward 1.0: {rewards.count(1.0)}")
    print(f"Reward 0<r<1: {len(rewards) - rewards.count(0.0) - rewards.count(1.0)}")

    output_path = os.path.join(OUTPUT_DIR, 'all_samples.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
