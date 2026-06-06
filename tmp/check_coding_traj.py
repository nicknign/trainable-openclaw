import json
from collections import Counter

fpath = '/data/wangye/trainable-openclaw/data/trajectories_high_error.jsonl'
coding = []
with open(fpath, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        cat = d.get('category', d.get('source_category', ''))
        # Also try Chinese field name
        for key in ['category', 'source_category', 'class']:
            v = d.get(key, '')
            if v:
                cat = v
                break
        if not cat:
            # Try Chinese field
            for k, v in d.items():
                if 'class' in str(k).lower() or 'cat' in str(k).lower() or 'type' in str(k).lower():
                    cat = v
                    break

        cat_str = str(cat).lower()
        if 'coding' in cat_str and 'debugging' not in cat_str:
            coding.append(d)

print(f'Total trajectories: ', end='')
with open(fpath, 'r', encoding='utf-8') as f:
    total = sum(1 for _ in f)
print(total)

print(f'Coding trajectories: {len(coding)}')
if coding:
    verdicts = Counter(str(d.get('verdict', d.get('result', 'unknown'))) for d in coding)
    print(f'Verdicts: {dict(verdicts)}')

    # Check first record keys
    print(f'First record keys: {list(coding[0].keys())[:10]}')
