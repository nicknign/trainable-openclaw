"""Debug: capture raw DeepSeek response for a specific rubric."""
import json, sys
sys.path.insert(0, "/data/wangye/trainable-openclaw")
from openai import OpenAI
from trainable_openclaw.evaluation.rubric import Rubric

with open("data/rubrics_coding_v4.json", "r") as f:
    data = json.load(f)
rubrics = [Rubric.from_dict(r) for r in data]

# Get a short test code
with open("data/coding/testset_eval.json", "r") as f:
    prev = json.load(f)

# Pick prompt 1 which had scores [0, 9, 10, 9]
r = prev["results"][0]
code = r["full_response"]
# Clean think tags
code = code.replace("<think>", "").replace("</think>", "").strip()
# Truncate
code = code[:3000]

print(f"Code length for judge: {len(code)}")
print(f"Code preview: {code[:200]}...")

client = OpenAI(api_key="sk-906ad0dc48354e7aba594ef6d9aa5be6", base_url="https://api.deepseek.com")

# Test each rubric individually and capture raw response
for i, rubric in enumerate(rubrics):
    prompt = rubric.评分提示词.replace("{content}", code)
    print(f"\n{'='*60}")
    print(f"Rubric {i+1}: {rubric.名称}")
    print(f"Prompt length: {len(prompt)}")

    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    print(f"Raw response ({len(raw)} chars):")
    print(raw[:500])
    print()
