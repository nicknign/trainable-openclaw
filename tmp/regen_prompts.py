import json

base = '/data/wangye/trainable-openclaw/data/coding'

# Regenerate test_prompts_only.txt from test.jsonl
with open(f'{base}/test.jsonl', 'r', encoding='utf-8') as f:
    prompts = []
    for line in f:
        d = json.loads(line)
        p = d.get('prompt', '').strip()
        if p:
            prompts.append(p)
with open(f'{base}/test_prompts_only.txt', 'w', encoding='utf-8') as f:
    for p in prompts:
        f.write(p + '\n')
print(f'Regenerated test_prompts_only.txt: {len(prompts)} prompts')

# Regenerate train_prompts_only.txt from train_all.jsonl
with open(f'{base}/train_all.jsonl', 'r', encoding='utf-8') as f:
    prompts = []
    for line in f:
        d = json.loads(line)
        p = d.get('prompt', '').strip()
        if p:
            prompts.append(p)
with open(f'{base}/train_prompts_only.txt', 'w', encoding='utf-8') as f:
    for p in prompts:
        f.write(p + '\n')
print(f'Regenerated train_prompts_only.txt: {len(prompts)} prompts')
