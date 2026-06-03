"""Remove train/test overlap from training set."""
import json, os, shutil

os.chdir('/data/wangye/trainable-openclaw')
TRAIN = 'data/phase3_datasets/train_prompts.jsonl'
TEST = 'data/phase3_datasets/test_prompts.jsonl'
BACKUP = 'data/phase3_datasets/train_prompts.jsonl.bak'

# Load test prompts
test_pairs = [json.loads(l) for l in open(TEST)]
test_prompts = set()
for p in test_pairs:
    seed = p.get('prompt', '') or p.get('种子提示词', '')
    test_prompts.add(seed.strip())

print('Test prompts:', len(test_prompts))

# Load train pairs
train_pairs = [json.loads(l) for l in open(TRAIN)]
print('Train pairs before:', len(train_pairs))

# Find overlap
removed = []
clean = []
for p in train_pairs:
    seed = p.get('prompt', '') or p.get('种子提示词', '')
    if seed.strip() in test_prompts:
        removed.append(seed.strip()[:80])
    else:
        clean.append(p)

print('Overlapping prompts:', len(set(removed)))
print('Removed pairs:', len(removed))
for r in sorted(set(removed)):
    print(' ', r)

print('\nTrain pairs after:', len(clean))

# Count unique prompts in clean set
clean_prompts = set()
for p in clean:
    seed = p.get('prompt', '') or p.get('种子提示词', '')
    clean_prompts.add(seed.strip())
print('Unique prompts after:', len(clean_prompts))

# Backup and save
shutil.copy(TRAIN, BACKUP)
with open(TRAIN, 'w', encoding='utf-8') as f:
    for p in clean:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')

# Verify
verify_pairs = [json.loads(l) for l in open(TRAIN)]
verify_prompts = set()
for p in verify_pairs:
    seed = p.get('prompt', '') or p.get('种子提示词', '')
    verify_prompts.add(seed.strip())
overlap = verify_prompts & test_prompts
print('\nVerification - remaining overlap:', len(overlap))
print('Backup saved to:', BACKUP)
print('Done.')
