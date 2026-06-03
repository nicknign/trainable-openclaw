import sys, json, os, time
sys.path.insert(0, '/data/wangye/trainable-openclaw')
os.chdir('/data/wangye/trainable-openclaw')

from trainable_openclaw.evaluation.rubric import Rubric
from trainable_openclaw.evaluation.judge import JudgeExecutor

# Load rubrics
with open('data/rubrics_dynamic.json', 'r') as f:
    data = json.load(f)

rubrics = [Rubric.from_dict(item) for item in data if item.get('状态') == '活跃']
print(f"Loaded {len(rubrics)} active rubrics")

judge = JudgeExecutor(
    api_key="sk-906ad0dc48354e7aba594ef6d9aa5be6",
    base_url="https://api.deepseek.com",
    model="deepseek-v4-flash",
    enable_thinking=False,
    use_merged=True,
)

# Test with 2 answers
answers = [
    "The capital of France is Paris. The Eiffel Tower is there.",
    "France's capital is London. It's known for Big Ben.",
]
prompt = "What is the capital of France?"

t0 = time.time()
results = judge.score_answers_sync(prompt, answers, rubrics)
elapsed = time.time() - t0

print(f"\nSync scoring took {elapsed:.1f}s")
for r in results:
    print(f"  回答: '{r['回答'][:60]}...'")
    print(f"  平均分: {r['平均分']}, 总分: {r['总分']}")
    print(f"  分数向量: {r['分数向量']}")

rewards = judge.compute_grpo_rewards(results, reward_mode="mean")
print(f"\nRewards: {rewards}")
print("SUCCESS" if any(r > 0 for r in rewards) else "FAILED - all zero rewards")
