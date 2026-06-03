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
    for item in data:
        r = Rubric.from_dict(item)
        if r.状态 == "活跃":
            rubrics.append(r)

    print(f"Loaded {len(rubrics)} active rubrics")

    # Test judge
    judge = JudgeExecutor(
        api_key="sk-906ad0dc48354e7aba594ef6d9aa5be6",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        enable_thinking=False,
        use_merged=True,
    )

    answer = "The capital of France is Paris. It is known for the Eiffel Tower."
    print(f"Scoring answer...")

    results = await judge.score_answers(
        prompt="What is the capital of France?",
        answers=[answer],
        rubrics=rubrics,
    )

    print(f"Result type: {type(results)}")
    print(f"Result len: {len(results)}")
    for r in results:
        print(f"\nKeys: {list(r.keys())}")
        print(f"平均分: {r.get('平均分')}")
        print(f"总分: {r.get('总分')}")
        print(f"分数向量: {r.get('分数向量')}")
        for s in r.get('评分', []):
            print(f"  {s['rubric名称']}: 分数={s['分数']}, error={s['解析错误'][:100] if s.get('解析错误') else 'none'}")

    rewards = judge.compute_grpo_rewards(results, reward_mode="mean")
    print(f"\nRewards: {rewards}")

asyncio.run(test())
