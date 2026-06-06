import json
from collections import Counter
cats = Counter()
with open("data/phase3_datasets/train_prompts.jsonl") as f:
    for line in f:
        d = json.loads(line)
        cats[d.get("category", "unknown")] += 1
for cat, n in cats.most_common():
    print(f"{cat:30s} {n:4d}")
print(f"{'TOTAL':30s} {sum(cats.values()):4d}")
print(f"\nCoding prompts: {cats.get('coding', 0)} / {sum(cats.values())}")
