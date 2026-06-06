import json, re
with open("data/coding/testset_eval.json") as f:
    d = json.load(f)

r = d["results"][0]  # P1: reverse linked list
resp = r["full_response"]
print(f"Total length: {len(resp)}")
print(f"Has </think>: {'</think>' in resp}")

# Show sections
print("\n=== First 500 chars ===")
print(resp[:500])
print("\n=== Middle (chars 4000-4500) ===")
print(resp[4000:4500])
print("\n=== Last 500 chars ===")
print(resp[-500:])

# Find any code-like content (lines with common code patterns)
lines = resp.split("\n")
code_lines = [l for l in lines if l.strip() and not l.strip().startswith(("Okay", "I need", "First", "Let me", "The user", "Wait", "But", "So", "Now", "Then", "In the", "This"))]
print(f"\n=== Potential code lines ({len(code_lines)}/{len(lines)}) ===")
for l in code_lines[:30]:
    print(f"  {l[:120]}")
