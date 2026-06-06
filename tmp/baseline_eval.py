"""基线评测：生成测试集回答 → Judge 评分 → 报告 baseline 指标."""
import json
import time
import sys
import os
from openai import OpenAI

# Config
API_BASE = "http://localhost:8000/v1"
TEST_PATH = "data/coding/test.jsonl"
RUBRICS_PATH = "data/rubrics_coding.json"
JUDGE_API_KEY = "sk-906ad0dc48354e7aba594ef6d9aa5be6"
JUDGE_BASE_URL = "https://api.deepseek.com"
JUDGE_MODEL = "deepseek-v4-flash"
MAX_TOKENS = 2048
TEMPERATURE = 1.0

# Load test prompts
with open(TEST_PATH, 'r', encoding='utf-8') as f:
    prompts = [json.loads(line).get('prompt', '') for line in f if line.strip()]
print(f"[BASELINE] {len(prompts)} test prompts loaded from {TEST_PATH}")

# Load rubrics
with open(RUBRICS_PATH, 'r', encoding='utf-8') as f:
    rubrics_data = json.load(f)
active_rubrics = [r for r in rubrics_data if r.get('状态', '活跃') == '活跃']
print(f"[BASELINE] {len(active_rubrics)} active rubrics: {[r['名称'] for r in active_rubrics]}")

# Step 1: Generate responses via serve_ppo API
vllm_client = OpenAI(api_key="EMPTY", base_url=API_BASE)
print(f"\n{'='*60}")
print("Step 1: Generating responses from Qwen3-4B (baseline)")
print(f"{'='*60}")

results = []
t_start = time.time()
for i, prompt in enumerate(prompts):
    t0 = time.time()
    try:
        resp = vllm_client.chat.completions.create(
            model="qwen",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        response_text = resp.choices[0].message.content
        results.append({"prompt": prompt, "response": response_text, "error": None})
        t1 = time.time()
        print(f"  [{i+1:2d}/{len(prompts)}] {t1-t0:.1f}s | {prompt[:60]}... → {len(response_text)} chars")
    except Exception as e:
        results.append({"prompt": prompt, "response": "", "error": str(e)})
        print(f"  [{i+1:2d}/{len(prompts)}] ERROR: {e}")

gen_time = time.time() - t_start
print(f"\nGeneration done: {gen_time:.1f}s ({gen_time/len(prompts):.1f}s/prompt)")

# Step 2: Score with Judge
print(f"\n{'='*60}")
print("Step 2: Scoring with Judge (coding rubrics)")
print(f"{'='*60}")

from trainable_openclaw.evaluation.judge import JudgeExecutor
from trainable_openclaw.evaluation.rubric import Rubric

rubric_objs = [Rubric.from_dict(r) for r in active_rubrics]
judge = JudgeExecutor(
    api_key=JUDGE_API_KEY,
    base_url=JUDGE_BASE_URL,
    model=JUDGE_MODEL,
    enable_thinking=False,
    use_merged=True,
)

all_scores = []
t_start = time.time()
for i, r in enumerate(results):
    if r["error"]:
        all_scores.append(0.0)
        print(f"  [{i+1:2d}/{len(results)}] SKIP (gen error)")
        continue
    try:
        t0 = time.time()
        score_results = judge.score_answers_sync(
            prompt=r["prompt"],
            answers=[r["response"]],
            rubrics=rubric_objs,
        )
        rewards = judge.compute_grpo_rewards(score_results, reward_mode="mean")
        score = rewards[0] if rewards else 0.0
        all_scores.append(score)
        t1 = time.time()
        print(f"  [{i+1:2d}/{len(results)}] {t1-t0:.1f}s | score={score:.3f} | {r['prompt'][:50]}...")
    except Exception as e:
        all_scores.append(0.0)
        print(f"  [{i+1:2d}/{len(results)}] ERROR: {e}")

score_time = time.time() - t_start

# Summary
mean_score = sum(all_scores) / len(all_scores) if all_scores else 0
min_score = min(all_scores)
max_score = max(all_scores)
above_5 = sum(1 for s in all_scores if s >= 0.5)
above_7 = sum(1 for s in all_scores if s >= 0.7)

print(f"\n{'='*60}")
print("BASELINE RESULTS")
print(f"{'='*60}")
print(f"Test prompts:       {len(prompts)}")
print(f"Valid responses:    {sum(1 for r in results if not r['error'])}")
print(f"Mean score:         {mean_score:.4f}")
print(f"Score range:        {min_score:.4f} ~ {max_score:.4f}")
print(f"Score >= 0.5:       {above_5}/{len(prompts)} ({above_5/len(prompts):.1%})")
print(f"Score >= 0.7:       {above_7}/{len(prompts)} ({above_7/len(prompts):.1%})")
print(f"Gen time:           {gen_time:.1f}s")
print(f"Judge time:         {score_time:.1f}s")
print(f"Total time:         {gen_time+score_time:.1f}s")

# Save for comparison
baseline_output = {
    "timestamp": time.time(),
    "model": "Qwen3-4B (baseline, before coding training)",
    "config": {
        "rubrics": RUBRICS_PATH,
        "test_prompts": len(prompts),
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    },
    "results": [
        {
            "prompt": r["prompt"],
            "response": r["response"][:500],
            "score": all_scores[i] if i < len(all_scores) else 0,
            "error": r["error"],
        }
        for i, r in enumerate(results)
    ],
    "summary": {
        "mean_score": mean_score,
        "min_score": min_score,
        "max_score": max_score,
        "above_0.5": above_5,
        "above_0.7": above_7,
        "gen_time_s": gen_time,
        "judge_time_s": score_time,
    },
}

os.makedirs("data/coding", exist_ok=True)
with open("data/coding/baseline_eval.json", 'w', encoding='utf-8') as f:
    json.dump(baseline_output, f, ensure_ascii=False, indent=2)

print(f"\nSaved to data/coding/baseline_eval.json")
