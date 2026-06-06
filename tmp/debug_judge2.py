"""Debug: what does DeepSeek return for a single rubric call?"""
import json, sys
sys.path.insert(0, "/data/wangye/trainable-openclaw")
from trainable_openclaw.evaluation.judge import JudgeExecutor
from trainable_openclaw.evaluation.rubric import Rubric

with open("data/rubrics_coding_v4.json", "r") as f:
    data = json.load(f)
code_rubrics = [Rubric.from_dict(r) for r in data]

judge = JudgeExecutor(
    api_key="sk-906ad0dc48354e7aba594ef6d9aa5be6",
    base_url="https://api.deepseek.com",
    model="deepseek-v4-flash",
    enable_thinking=False,
    use_merged=False,
)

with open("data/coding/testset_eval.json", "r") as f:
    prev = json.load(f)

# Test prompt 1: reverse linked list (no code blocks, 8433 chars)
r = prev["results"][0]
code = r["extracted_code"]
print(f"Code length: {len(code)}")
print(f"Has code blocks: {r.get('has_code_blocks', False)}")

# Test with full vs truncated
for name, test_code in [("full (8433 chars)", code), ("truncated 3000", code[:3000]), ("truncated 1500", code[:1500])]:
    rubric = code_rubrics[0]
    print(f"\n{'='*60}")
    print(f"Testing with {name} against rubric: {rubric.名称}")
    result = judge.score_one_sync(rubric, test_code)
    print(f"  Score: {result.分数}")
    print(f"  Parse error: {result.解析错误[:200] if result.解析错误 else 'None'}")
    print(f"  Summary: {result.总结[:200] if result.总结 else 'None'}")
    if result.原始输出:
        print(f"  Raw output (first 300): {result.原始输出[:300]}")
    else:
        print(f"  Raw output: None/empty")

# Also test prompt 7 (quicksort, 347 chars, code blocks)
r7 = prev["results"][6]
code7 = r7["extracted_code"]
print(f"\n{'='*60}")
print(f"Testing quicksort (347 chars, has code blocks)")
print(f"Code:\n{code7[:500]}")
for rubric in code_rubrics:
    result = judge.score_one_sync(rubric, code7)
    print(f"  {rubric.名称}: score={result.分数}, error={result.解析错误[:100] if result.解析错误 else 'None'}")
