import json
from collections import Counter

# Check train.jsonl
with open("data/coding/train.jsonl") as f:
    train = [json.loads(l) for l in f if l.strip()]
print(f"train.jsonl: {len(train)} entries")
print(f"Keys: {list(train[0].keys())}")
print(f"Sample prompt: {train[0].get('prompt', '')[:150]}")
print()

# Check train_all.jsonl
with open("data/coding/train_all.jsonl") as f:
    train_all = [json.loads(l) for l in f if l.strip()]
print(f"train_all.jsonl: {len(train_all)} entries")
print(f"Keys: {list(train_all[0].keys())}")
print()

# Check unique prompts
prompts = [t['prompt'] for t in train]
print(f"Unique prompts in train: {len(set(prompts))}")

# Check overlap with test
with open("data/coding/test.jsonl") as f:
    test_prompts = [json.loads(l)['prompt'] for l in f if l.strip()]
overlap = set(prompts) & set(test_prompts)
print(f"Overlap with test: {len(overlap)}")
