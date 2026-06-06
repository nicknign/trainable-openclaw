import json
from collections import Counter

fpath = '/data/wangye/trainable-openclaw/data/trajectories_high_error.jsonl'
cats = Counter()
coding_trajs = []

with open(fpath, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        cat = d.get('类别', '')
        cats[cat] += 1
        if str(cat) == 'coding':
            coding_trajs.append(d)

print('All categories:')
for c, n in cats.most_common():
    print(f'  {c}: {n}')

print(f'\nCoding trajectories: {len(coding_trajs)}')
if coding_trajs:
    from collections import Counter
    verdicts = Counter(d.get('最终判定', '') for d in coding_trajs)
    print(f'Verdicts: {dict(verdicts)}')
