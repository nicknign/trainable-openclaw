"""Quick remote rubric test - does scoring work with local judge.py?"""
import json, sys, os
sys.path.insert(0, '/data/wangye/trainable-openclaw')
from trainable_openclaw.evaluation.rubric import Rubric
from trainable_openclaw.evaluation.judge import JudgeExecutor

# Load rubrics
with open("data/rubrics_coding_v4.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} rubrics")
for i, d in enumerate(data):
    r = Rubric.from_dict(d)
    print(f"  [{i}] {r.名称} | prompt_len={len(r.评分提示词)} | has_content={'{content}' in r.评分提示词}")

# Test scoring with a simple answer
rubric_objs = [Rubric.from_dict(d) for d in data]
judge = JudgeExecutor(
    api_key="sk-906ad0dc48354e7aba594ef6d9aa5be6",
    base_url="https://api.deepseek.com",
    model="deepseek-v4-flash",
    enable_thinking=False,
    use_merged=False,
)

test_code = """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[-1]
    left = [x for x in arr[:-1] if x <= pivot]
    right = [x for x in arr[:-1] if x > pivot]
    return quicksort(left) + [pivot] + quicksort(right)"""

print(f"\nTest code length: {len(test_code)}")
print("Testing individual scoring...")

results = judge.score_answers_sync(
    prompt="write a quicksort function",
    answers=[test_code],
    rubrics=rubric_objs,
)

for r in results:
    print(f"\nMean: {r.get('平均分', 0):.1f}")
    print(f"Scores: {r.get('分数向量', [])}")
    for rs in r.get("评分", []):
        print(f"  {rs.get('rubric名称', '?')}: score={rs.get('分数', 0)}, error={rs.get('解析错误', '')[:80]}")
