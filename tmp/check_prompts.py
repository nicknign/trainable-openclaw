import json
fpath = '/data/wangye/trainable-openclaw/data/coding/test_prompts_only.txt'
with open(fpath, 'r', encoding='utf-8') as f:
    prompts = [l.strip() for l in f if l.strip()]
lengths = sorted([len(p) for p in prompts], reverse=True)
print(f'Test prompts: {len(prompts)}')
print(f'Max length: {lengths[0]} chars')
print(f'Top 5 longest: {lengths[:5]}')
longest = max(prompts, key=len)
print(f'Longest prompt (first 200): {longest[:200]}')
# Also count how many > 3000 chars (to keep room for response)
over = sum(1 for l in lengths if l > 3000)
print(f'Prompts > 3000 chars: {over}')
