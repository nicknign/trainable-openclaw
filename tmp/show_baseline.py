import json

with open('/data/wangye/trainable-openclaw/data/coding/baseline_eval.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print('=== BASELINE SUMMARY ===')
for k, v in d['summary'].items():
    print(f'  {k}: {v}')

print(f'\n=== PER-ITEM SCORES ({len(d["results"])} prompts) ===')
for i, r in enumerate(d['results']):
    resp = r['response'].replace('\n', ' ')
    print(f'{i+1:2d}. score={r["score"]:.3f} | {r["prompt"][:70]}')
    print(f'    response: {resp[:120]}...')

print(f'\nNon-zero scores: {sum(1 for r in d["results"] if r["score"] > 0)}/{len(d["results"])}')
print(f'Scores: {[r["score"] for r in d["results"]]}')
