"""
Debug Judge scoring: test individual vs merged scoring on coding test prompts.
Runs LOCALLY using DeepSeek API.
"""
import json
import os
import sys
import time

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-906ad0dc48354e7aba594ef6d9aa5be6")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"

# Load test prompts
test_path = "data/coding_debug/test.jsonl"
with open(test_path, 'r', encoding='utf-8') as f:
    test_prompts = [json.loads(line)['prompt'] for line in f if line.strip()]
print(f"Test prompts: {len(test_prompts)}")

# Load rubrics
rubrics_path = "data/coding_debug/rubrics_coding.json"
with open(rubrics_path, 'r', encoding='utf-8') as f:
    rubric_data = json.load(f)
print(f"Rubrics: {len(rubric_data)}")
for r in rubric_data:
    print(f"  {r['id'][:12]}: {r['名称']} ({len(r['评分提示词'])} chars)")

# Load baseline results for responses
baseline_path = "data/coding_debug/baseline_eval.json"
with open(baseline_path, 'r', encoding='utf-8') as f:
    baseline_data = json.load(f)

# Pick 3 test cases to debug
test_cases = baseline_data['results'][:3]

print("\n" + "=" * 60)
print("DEBUG: Test individual scoring on 3 test prompts × 6 rubrics")
print("=" * 60)

from openai import OpenAI
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

for case_idx, case in enumerate(test_cases):
    prompt = case['prompt']
    answer = case['response']
    print(f"\n{'='*60}")
    print(f"Case {case_idx+1}: {prompt[:80]}...")
    print(f"Answer length: {len(answer)} chars")
    print(f"{'='*60}")

    for rub_idx, rubric in enumerate(rubric_data):
        # Build individual scoring prompt
        scoring_prompt = rubric['评分提示词'].replace('{content}', answer)

        # Truncate answer if prompt is too long
        if len(scoring_prompt) > 4000:
            # Truncate answer to fit
            max_answer_len = 4000 - len(rubric['评分提示词']) + len('{content}')
            truncated_answer = answer[:max_answer_len] + "\n... (truncated)"
            scoring_prompt = rubric['评分提示词'].replace('{content}', truncated_answer)

        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": scoring_prompt}],
                temperature=0.0,
                max_tokens=500,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = resp.choices[0].message.content.strip()
            elapsed = time.time() - t0

            # Parse score
            try:
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1]
                    if raw.endswith("```"):
                        raw = raw.rsplit("\n", 1)[0]
                result = json.loads(raw)
                score = result.get("分数", 0)
            except json.JSONDecodeError:
                # Try regex fallback
                import re
                m = re.search(r'"分数"\s*:\s*([\d.]+)', raw)
                score = float(m.group(1)) if m else 0.0

            print(f"  Rubric {rub_idx+1} [{rubric['名称']}]: score={score:.1f} ({elapsed:.1f}s) | ({len(raw)} chars)")
            if score == 0 and len(raw) > 5:
                print(f"    RAW: {raw[:150]}...")
        except Exception as e:
            print(f"  Rubric {rub_idx+1}: ERROR: {e}")

print("\nDone!")
