"""Compare high-scoring vs low-scoring responses."""
import json

with open("data/coding/testset_eval.json") as f:
    d = json.load(f)

with open("data/coding/test.jsonl") as f:
    prompts = [json.loads(l)["prompt"] for l in f if l.strip()]

# Show P7 (quicksort, score 8.80) vs P5 (pygame clock, score 0.20)
pairs = [
    (6, "P7 quicksort 8.80"),
    (0, "P1 reverse_ll 6.60"),
    (4, "P5 pygame 0.20"),
    (13, "P14 swift 0.20"),
]

for idx, label in pairs:
    r = d["results"][idx]
    resp = r["full_response"]
    cleaned = resp.replace("<think>", "").replace("</think>", "").strip()
    print(f"=== {label} ===")
    print(f"think_len={r['think_len']} code_len={r['code_len']} extracted={r['extracted_len']} blocks={r['has_code_blocks']}")
    print(f"PROMPT: {prompts[idx][:200]}")
    print(f"RESPONSE (first 400 chars):")
    print(cleaned[:400])
    print(f"... (total {len(cleaned)} chars)")
    print()
