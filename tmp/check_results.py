import json, re
with open("data/coding/testset_eval.json") as f:
    d = json.load(f)
print("Top keys:", list(d.keys()))
r = d["results"][0]
print("Result keys:", list(r.keys()))
print("has_code_blocks:", r.get("has_code_blocks"))
resp = r.get("full_response", "")
print("full_response len:", len(resp))
code = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
print("code_part len:", len(code))
blocks = re.findall(r"```(?:\w+)?\s*\n(.*?)```", code, re.DOTALL)
print("code_blocks found:", len(blocks))
print("first 500 of code_part:", code[:500])
