"""Smoke test for compute_score — pure local, no API calls."""
import json, time, sys
sys.path.insert(0, "/data/wangye/trainable-openclaw")
from trainable_openclaw.training.grpo_reward import compute_score

with open("/data/wangye/trainable-openclaw/data/tau_bench/train_split_66.jsonl") as f:
    tasks = [json.loads(l) for l in f if l.strip()]
print(f"Loaded {len(tasks)} tasks")

for t in tasks[:5]:
    tid = t.get("id", "?")
    resp = '<function_call>{"name": "find_user_id_by_name_zip", "arguments": {"name": "test", "zip": "12345"}}</function_call>'
    t0 = time.time()
    r = compute_score(
        data_source=f"retail_{tid}",
        solution_str=resp,
        ground_truth=json.dumps(t),
        extra_info=t,
    )
    print(f"  {tid}: reward={r:.4f}  ({time.time()-t0:.2f}s)")

print("\nAll passed")
