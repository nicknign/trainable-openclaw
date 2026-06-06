import json
import random
import os
from collections import defaultdict

random.seed(42)
base = '/data/wangye/trainable-openclaw/data/phase3_datasets'
out_dir = '/data/wangye/trainable-openclaw/data/coding'
os.makedirs(out_dir, exist_ok=True)

# Step 1: Collect all unique coding prompts from train + test files
all_coding = {}  # prompt -> full record
for fname in ['train_prompts.jsonl', 'test_prompts.jsonl']:
    with open(f'{base}/{fname}', 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            cat = d.get('类别', d.get('source_category', d.get('category', '')))
            if 'coding' in str(cat).lower() and 'debugging' not in str(cat).lower():
                prompt = d.get('prompt', d.get('种子提示词', ''))
                if prompt not in all_coding:
                    all_coding[prompt] = d

prompts = list(all_coding.keys())
random.shuffle(prompts)

split = int(len(prompts) * 0.8)
train_prompts = set(prompts[:split])
test_prompts = set(prompts[split:])

print(f'Total unique coding prompts: {len(prompts)}')
print(f'Training set: {len(train_prompts)}')
print(f'Test set: {len(test_prompts)}')
print(f'Train/Test overlap: {len(train_prompts & test_prompts)}')

# Step 2: Load full training pairs and match to train prompts
train_pairs = []
with open(f'{base}/training_pairs.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        prompt = d.get('prompt', d.get('种子提示词', ''))
        if prompt in train_prompts:
            train_pairs.append(d)
            train_prompts.discard(prompt)  # mark as matched

missing = [p for p in train_prompts if p in all_coding]
print(f'Training pairs matched: {len(train_pairs)}')
print(f'Training prompts WITHOUT pairs (raw only): {len(missing)}')

# Step 3: Write training pairs (with correction data)
train_output = []
for d in train_pairs:
    entry = {
        'prompt': d.get('prompt', d.get('种子提示词', '')),
        'category': d.get('类别', d.get('source_category', 'coding')),
        'correction': d.get('纠错意见', d.get('纠错意�', '')),
        'wrong_answer': d.get('错误回答', d.get('错�answer', '')),
        'corrected_answer': d.get('修正回答', d.get('修�answer', '')),
        'corrected_reasoning': d.get('修正思考', d.get('修�思考', '')),
    }
    train_output.append(entry)

with open(f'{out_dir}/train.jsonl', 'w', encoding='utf-8') as f:
    for entry in train_output:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f'Wrote {len(train_output)} entries to train.jsonl')

# Step 4: Write raw train prompts (without pairs) as simple prompts
raw_prompts = []
for p in missing:
    raw_prompts.append({'prompt': p, 'category': all_coding[p].get('类别', 'coding')})

# Combine: training pairs + raw prompts
all_train = train_output + raw_prompts
with open(f'{out_dir}/train_all.jsonl', 'w', encoding='utf-8') as f:
    for entry in all_train:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f'Wrote {len(all_train)} entries to train_all.jsonl (with pairs: {len(train_output)}, raw: {len(raw_prompts)})')

# Step 5: Write test set
test_output = []
for p in test_prompts:
    test_output.append({
        'prompt': p,
        'category': all_coding[p].get('类别', d.get('source_category', 'coding')),
    })

with open(f'{out_dir}/test.jsonl', 'w', encoding='utf-8') as f:
    for entry in test_output:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f'Wrote {len(test_output)} entries to test.jsonl')

# Step 6: Write prompt-only files for training (just the prompt strings)
with open(f'{out_dir}/train_prompts_only.txt', 'w', encoding='utf-8') as f:
    for entry in all_train:
        f.write(entry['prompt'] + '\n')

with open(f'{out_dir}/test_prompts_only.txt', 'w', encoding='utf-8') as f:
    for entry in test_output:
        f.write(entry['prompt'] + '\n')

print('Done!')
print(f'\nFiles created in {out_dir}/:')
for fn in sorted(os.listdir(out_dir)):
    size = os.path.getsize(f'{out_dir}/{fn}')
    print(f'  {fn} ({size} bytes)')
