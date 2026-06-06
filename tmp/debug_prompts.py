import json

base = '/data/wangye/trainable-openclaw/data/coding'

# Check test_prompts_only.txt
with open(f'{base}/test_prompts_only.txt', 'r', encoding='utf-8') as f:
    lines = [l for l in f if l.strip()]
print(f'test_prompts_only.txt non-empty lines: {len(lines)}')
# Print first 3 lines
for i, l in enumerate(lines[:3]):
    print(f'  Line {i}: {l[:80]}...')

# Check test.jsonl
with open(f'{base}/test.jsonl', 'r', encoding='utf-8') as f:
    prompts = []
    for line in f:
        d = json.loads(line)
        p = d.get('prompt', '')
        prompts.append(p)
print(f'test.jsonl prompts: {len(prompts)}')
multi = sum(1 for p in prompts if '\n' in p)
print(f'Multi-line prompts: {multi}')

# Check train_prompts_only.txt
with open(f'{base}/train_prompts_only.txt', 'r', encoding='utf-8') as f:
    lines = [l for l in f if l.strip()]
print(f'train_prompts_only.txt non-empty lines: {len(lines)}')
