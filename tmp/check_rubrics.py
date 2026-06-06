import json

# Check rubric format
with open('/data/wangye/trainable-openclaw/data/rubrics_coding.json', 'r', encoding='utf-8') as f:
    rubrics = json.load(f)

for r in rubrics:
    prompt = r['评分提示词']
    print(f'=== {r["名称"]} ===')
    print(f'Length: {len(prompt)} chars')
    # Check for JSON format instruction
    has_json = 'JSON' in prompt or 'json' in prompt or '{' in prompt
    print(f'Has JSON format instruction: {has_json}')
    # Show last 150 chars
    print(f'End: ...{prompt[-150:]}')
    print()

# Check baseline results
with open('/data/wangye/trainable-openclaw/data/coding/baseline_eval.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'\n=== Baseline Summary ===')
print(f"Mean: {data['summary']['mean_score']}")
print(f"Scores: {[r['score'] for r in data['results']]}")
# Show first 2 responses
for r in data['results'][:2]:
    print(f"\nPrompt: {r['prompt'][:80]}")
    print(f"Score: {r['score']}")
    print(f"Response (first 100): {r['response'][:100]}")
