"""Validate category rubrics — sample prompts, generate answers, score, check distribution."""
import json, sys, os, time, urllib.request

sys.path.insert(0, '/data/wangye/trainable-openclaw')

# Sample prompts from each category group
TEST_PROMPTS = {
    "coding": [
        "Write a Python function to check if a string is a palindrome.",
        "How do I read a CSV file with pandas and filter rows where age > 30?",
    ],
    "debugging": [
        "My Python code gives 'KeyError: 0' when accessing a dictionary. What's wrong?",
        "I get 'ImportError: No module named requests' but I already installed it. Help.",
    ],
    "logical reasoning": [
        "If all dogs are mammals and all mammals are animals, are all dogs animals? Explain.",
        "In a race, if A finishes before B, and C finishes after B, who finishes last?",
    ],
    "math": [
        "What is the probability of rolling a sum of 7 with two dice?",
        "Solve for x: 2x + 5 = 13",
    ],
    "brainstorming": [
        "List 5 creative uses for a paperclip besides holding papers.",
        "What are some eco-friendly alternatives to plastic packaging?",
    ],
    "explanation": [
        "Explain how photosynthesis works in simple terms.",
        "What is machine learning and how is it different from traditional programming?",
    ],
    "creative writing": [
        "Write a short story about a robot learning to paint.",
        "Describe a sunset from the perspective of a blind person who just regained sight.",
    ],
    "copywriting": [
        "Write a product description for a wireless noise-cancelling headphone.",
        "Create a slogan for an eco-friendly water bottle brand.",
    ],
    "debating": [
        "Is remote work better than office work? Give arguments for both sides.",
        "Should AI be regulated? Present pros and cons.",
    ],
    "translation": [
        "Translate 'It's raining cats and dogs' to Chinese.",
        "How do you say 'thank you very much' in Japanese?",
    ],
    "instruction following": [
        "Sort these words by length: elephant, cat, dog, butterfly, ant",
        "Rank these from coldest to hottest: boiling water, ice, room temperature, lava",
    ],
}

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_PATH = "/data/models/Qwen3-4B"

def generate_answer(prompt):
    data = json.dumps({
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 1.0,
        "top_p": 1.0,
    }).encode()
    try:
        r = urllib.request.Request(VLLM_URL, data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(r, timeout=60)
        body = json.loads(resp.read())
        return body["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  Generation failed: {e}")
        return ""

def validate():
    from trainable_openclaw.training.reward_bridge import RewardBridge

    bridge = RewardBridge(
        rubrics_path="data/rubrics_category.json",
        api_key="sk-906ad0dc48354e7aba594ef6d9aa5be6",
        model="deepseek-v4-flash",
        enable_thinking=False,
        use_merged=True,
    )

    print(f"Loaded {len(bridge.rubrics)} rubrics")
    if bridge._category_map:
        print(f"Category map: {len(bridge._category_map)} entries")

    all_scores = []
    group_results = {}

    for cat, prompts in TEST_PROMPTS.items():
        cat_scores = []
        print(f"\n{'='*50}")
        print(f"Category: {cat} ({len(prompts)} prompts)")
        print(f"{'='*50}")

        for pi, prompt in enumerate(prompts):
            print(f"\n  Prompt {pi+1}: {prompt[:80]}...")

            # Generate answer
            answer = generate_answer(prompt)
            if not answer:
                print(f"  SKIP: generation failed")
                cat_scores.append(0.0)
                continue
            print(f"  Answer: {answer[:100]}...")

            # Score with category filter
            try:
                rewards = bridge.score_responses(prompt, [answer], category=cat)
                score = rewards[0]
                print(f"  Reward: {score:.3f}")
                cat_scores.append(score)
                all_scores.append(score)
            except Exception as e:
                print(f"  Scoring failed: {e}")
                cat_scores.append(0.0)
                all_scores.append(0.0)

        if cat_scores:
            mean = sum(cat_scores) / len(cat_scores)
            positive = sum(1 for s in cat_scores if s > 0.3)
            spread = max(cat_scores) - min(cat_scores) if len(cat_scores) > 1 else 0
            group_results[cat] = {"mean": mean, "positive": positive, "total": len(cat_scores), "spread": spread}
            print(f"\n  Category '{cat}': mean={mean:.3f}, positive(>0.3)={positive}/{len(cat_scores)}, spread={spread:.3f}")

    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS")
    print(f"{'='*60}")
    if all_scores:
        overall_mean = sum(all_scores) / len(all_scores)
        overall_positive = sum(1 for s in all_scores if s > 0.3)
        overall_spread = max(all_scores) - min(all_scores) if len(all_scores) > 1 else 0
        print(f"Overall: mean={overall_mean:.3f}, positive(>0.3)={overall_positive}/{len(all_scores)}, spread={overall_spread:.3f}")

        # Per-group summary
        for cat, res in sorted(group_results.items()):
            status = "PASS" if res["mean"] >= 0.15 and res["spread"] > 0.05 else "WEAK"
            print(f"  {cat}: mean={res['mean']:.3f}, pos={res['positive']}/{res['total']}, spread={res['spread']:.3f} [{status}]")

        # Pass criteria (relaxed for Qwen3-4B)
        passed = overall_mean >= 0.15 and overall_spread > 0.05
        print(f"\nVALIDATION: {'PASS' if passed else 'FAIL'} (need mean>=0.15, spread>0.05)")
        return passed
    return False

if __name__ == "__main__":
    print("Checking vLLM health...")
    try:
        r = urllib.request.urlopen("http://localhost:8000/v1/health", timeout=5)
        print(f"Health: {r.read().decode()}")
    except Exception as e:
        print(f"vLLM not accessible: {e}")
        print("Please start serve_ppo first (inference mode)")
        sys.exit(1)

    ok = validate()
    sys.exit(0 if ok else 1)
