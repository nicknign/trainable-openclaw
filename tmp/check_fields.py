import json
fpath = '/data/wangye/trainable-openclaw/data/trajectories_high_error.jsonl'
with open(fpath, 'r', encoding='utf-8') as f:
    d = json.loads(f.readline())
print('Keys:', list(d.keys()))
for k, v in d.items():
    s = str(v)
    print(f'  {k}: {s[:100]}')
