import json
from collections import Counter

base = '/data/wangye/trainable-openclaw/data/phase3_datasets'

for fname in ['all_prompts.jsonl', 'train_prompts.jsonl', 'test_prompts.jsonl', 'training_pairs.jsonl']:
    coding = []
    non_coding = Counter()
    total = 0
    with open(f'{base}/{fname}', 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            d = json.loads(line)
            cat = d.get('类别', d.get('source_category', d.get('category', '')))
            if 'coding' in str(cat).lower() and 'debugging' not in str(cat).lower():
                prompt = d.get('prompt', d.get('种子提示词', ''))
                coding.append(prompt[:80])
            else:
                non_coding[str(cat)] += 1

    print(f'=== {fname} (total {total}) ===')
    print(f'  Coding: {len(coding)}')
    print(f'  Non-coding categories: {sum(non_coding.values())}')
    if len(coding) <= 3:
        for c in coding:
            print(f'    -> {c}')
    print()

# Overall across all sources (deduplicated by prompt)
print('=== Overall unique coding prompts ===')
all_prompts = set()
for fname in ['train_prompts.jsonl', 'test_prompts.jsonl']:
    with open(f'{base}/{fname}', 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            cat = d.get('类别', d.get('source_category', d.get('category', '')))
            if 'coding' in str(cat).lower() and 'debugging' not in str(cat).lower():
                prompt = d.get('prompt', d.get('种子提示词', ''))
                all_prompts.add(prompt)

print(f'Total unique coding prompts (train+test): {len(all_prompts)}')
