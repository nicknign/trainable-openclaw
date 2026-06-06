"""Test rubric scoring locally to see actual DeepSeek responses."""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-906ad0dc48354e7aba594ef6d9aa5be6")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Load rubrics
with open("data/coding_debug/rubrics_v2.json", "r", encoding="utf-8") as f:
    rubrics = json.load(f)

# Sample code answer
sample_code = """def reverse_linked_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Example usage
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
reversed_head = reverse_linked_list(head)"""

# Test each rubric individually
for i, rubric in enumerate(rubrics):
    prompt = rubric['评分提示词'].replace('{content}', sample_code)
    print(f"\n{'='*60}")
    print(f"Rubric {i+1}: {rubric['名称']}")
    print(f"Prompt length: {len(prompt)}")
    print(f"{'='*60}")

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
        extra_body={"thinking": {"type": "disabled"}},
    )
    raw = resp.choices[0].message.content
    print(f"RAW RESPONSE:\n{raw}\n")

    # Try parsing
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if len(lines) > 1 else lines
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        result = json.loads(text)
        print(f"PARSED: score={result.get('分数', '?')}, deductions={result.get('扣分项', [])}")
    except json.JSONDecodeError as e:
        print(f"JSON PARSE FAILED: {e}")
        import re
        m = re.search(r'"分数"\s*:\s*([\d.]+)', raw)
        if m:
            print(f"Regex found score: {m.group(1)}")
        else:
            print("No score found in raw response")
