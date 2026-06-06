import json, re
with open("data/coding/testset_eval.json") as f:
    d = json.load(f)

for i in range(5):
    r = d["results"][i]
    resp = r["full_response"]
    code_part = resp
    think_part = ""
    if "<think>" in resp:
        think_blocks = re.findall(r"<think>(.*?)</think>", resp, re.DOTALL)
        think_part = "\n".join(think_blocks)
        code_part = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
        # Handle unclosed <think>
        if "<think>" in code_part:
            code_part = re.sub(r"<think>.*$", "", code_part, flags=re.DOTALL).strip()
        if not code_part:
            code_part = resp
    code_blocks = re.findall(r"```(?:\w+)?\s*\n(.*?)```", code_part, re.DOTALL)
    if code_blocks:
        extracted = "\n\n".join(code_blocks).strip()
    else:
        extracted = code_part
    if len(extracted) > 4000:
        extracted = extracted[:4000]
    print(f"P{i+1}: think={len(think_part)} code_part={len(code_part)} extracted={len(extracted)} blocks={len(code_blocks)} orig_score={r['score']}")
    if extracted:
        print(f"  First 150: {extracted[:150]}")
    else:
        print(f"  EMPTY extracted!")
    print()
