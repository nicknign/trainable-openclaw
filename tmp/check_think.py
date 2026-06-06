import json, re
with open("data/coding/testset_eval.json") as f:
    d = json.load(f)

# Check first 3 results for think tag handling
for i in range(min(3, len(d["results"]))):
    r = d["results"][i]
    resp = r["full_response"]
    print(f"\n--- Result {i+1} ---")
    print(f"has_code_blocks: {r['has_code_blocks']}")
    print(f"extracted_len: {r['extracted_len']}")
    print(f"score: {r['score']}")
    print(f"has <think>: {'<think>' in resp}")
    print(f"has </think>: {'</think>' in resp}")
    if "<think>" in resp:
        idx_start = resp.find("<think>")
        idx_end = resp.find("</think>")
        print(f"<think> at pos {idx_start}, </think> at pos {idx_end}")
        if idx_end > idx_start:
            print(f"think content len: {idx_end - idx_start - 7}")
    # Check code_part from detail if available
    detail = r.get("detail", {})
    if "format_score" in detail:
        print(f"detail keys: {list(detail.keys())}")
    print(f"first 200 chars of full_response:")
    print(resp[:200])
    print("...")
    print(f"last 200 chars of full_response:")
    print(resp[-200:])
