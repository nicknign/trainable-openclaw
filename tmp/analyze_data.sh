#!/bin/bash
# Analyze training/test dataset on remote server
cd /data/wangye/trainable-openclaw

echo "=== Data Directory ==="
find data -name "*.jsonl" -o -name "*.json" | sort

echo ""
echo "=== File Sizes ==="
wc -l data/phase3_datasets/train_prompts.jsonl 2>/dev/null
ls -lh data/phase3_datasets/ 2>/dev/null

echo ""
echo "=== Training Pairs Analysis ==="
python3 << 'PYEOF'
import json, os

# Train data
train_path = "data/phase3_datasets/train_prompts.jsonl"
if os.path.exists(train_path):
    pairs = [json.loads(l) for l in open(train_path)]
    print(f"train_prompts.jsonl: {len(pairs)} pairs")

    prompts = set()
    for p in pairs:
        seed = (p.get("prompt", "") or p.get("种子提示词", "")).strip()
        prompts.add(seed)
    print(f"Unique prompts: {len(prompts)}")

    # Categories
    cats = {}
    for p in pairs:
        c = p.get("类别", "") or p.get("category", "")
        cats[c] = cats.get(c, 0) + 1
    print(f"Categories ({len(cats)}):")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")

    # Sample a few prompts
    print("\nSample prompts:")
    for i, p in enumerate(pairs[:3]):
        seed = (p.get("prompt", "") or p.get("种子提示词", "")).strip()
        cat = p.get("类别", "") or p.get("category", "")
        print(f"\n  [{cat}] {seed[:120]}...")

# Check test data
print("\n=== Test Data ===")
test_files = [
    "data/phase3_datasets/test_prompts.jsonl",
    "data/seed_test_50.jsonl",
    "data/test_pairs.jsonl",
]
for f in test_files:
    if os.path.exists(f):
        n = sum(1 for _ in open(f))
        print(f"  {f}: {n} lines")

# Check rubric file
print("\n=== Rubric Files ===")
for f in ["data/rubrics_dynamic.json", "data/rubrics_v2.json", "data/rubrics.json"]:
    if os.path.exists(f):
        data = json.load(open(f))
        if isinstance(data, list):
            active = sum(1 for r in data if isinstance(r, dict) and r.get("状态") == "活跃")
            print(f"  {f}: {len(data)} total, {active} active")
        else:
            print(f"  {f}: {type(data).__name__}")

print("\nDone.")
PYEOF
