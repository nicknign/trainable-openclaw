"""Debug: test with higher max_tokens to fix empty responses."""
import json, sys
sys.path.insert(0, "/data/wangye/trainable-openclaw")
from openai import OpenAI
from trainable_openclaw.evaluation.rubric import Rubric

with open("data/rubrics_coding_v4.json", "r") as f:
    data = json.load(f)
rubrics = [Rubric.from_dict(r) for r in data]

with open("data/coding/testset_eval.json", "r") as f:
    prev = json.load(f)
r = prev["results"][0]
code = r["full_response"].replace("<think>", "").replace("</think>", "").strip()[:3000]

client = OpenAI(api_key="sk-906ad0dc48354e7aba594ef6d9aa5be6", base_url="https://api.deepseek.com")

for max_tok in [500, 1024, 2048]:
    print(f"\n{'='*60}")
    print(f"max_tokens={max_tok}")
    for i in [2, 3]:  # rubrics 3 and 4 (0-indexed)
        rubric = rubrics[i]
        prompt = rubric.评分提示词.replace("{content}", code)
        try:
            resp = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tok,
                temperature=0.0,
                timeout=30,
            )
            raw = resp.choices[0].message.content
            print(f"  R{i+1}: {len(raw) if raw else 0} chars | {raw[:150] if raw else 'EMPTY'}")
        except Exception as e:
            print(f"  R{i+1}: ERROR: {e}")
