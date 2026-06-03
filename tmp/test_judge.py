import sys, json, asyncio, os
sys.path.insert(0, '/data/wangye/trainable-openclaw')
os.chdir('/data/wangye/trainable-openclaw')

async def test():
    from trainable_openclaw.evaluation.rubric import Rubric
    from trainable_openclaw.evaluation.judge import JudgeExecutor

    # Load rubrics
    with open('data/rubrics_dynamic.json', 'r') as f:
        data = json.load(f)

    rubrics = []
    for item in data[:2]:  # First 2 rubrics
        r = Rubric.from_dict(item)
        rubrics.append(r)
        print(f"Rubric: {r.名称} (active={r.状态})")

    print(f"\nLoaded {len(rubrics)} rubrics")

    # Test judge
    judge = JudgeExecutor(
        api_key="sk-906ad0dc48354e7aba594ef6d9aa5be6",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        enable_thinking=False,
        use_merged=True,
    )

    # Test scoring one answer
    answer = "The capital of France is Paris. It is known for the Eiffel Tower."
    print(f"\nScoring: '{answer}'")

    scores = await judge.score_answers(
        prompt="What is the capital of France?",
        answers=[answer],
        rubrics=rubrics,
    )

    for s in scores:
        print(f"\nAnswer scores:")
        for rs in s.rubric_scores:
            print(f"  {rs.rubric_name}: 分数={rs.分数}, error={rs.解析错误}")

    rewards = judge.compute_grpo_rewards(scores, reward_mode="mean")
    print(f"\nRewards: {rewards}")

asyncio.run(test())
