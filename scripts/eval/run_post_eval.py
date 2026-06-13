"""Run post-training evaluation on checkpoint and compare with baseline."""
import asyncio, json, os, sys, time
from pathlib import Path

# Config
MODEL_SERVER = "http://localhost:8000/v1"
MODEL_NAME = "ckpt10"  # LoRA adapter name
TEST_DATA = "data/phase3_datasets/test_prompts.jsonl"
BASELINE_PATH = "data/phase3_datasets/baseline_eval.json"
OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/post_eval_ckpt10.json"

# API config
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
SIM_MODEL = "deepseek-v4-flash"

sys.path.insert(0, os.getcwd())

from trainable_openclaw.evaluation.correction_rate import (
    CorrectionRateEvaluator, CorrectionRateResult, PromptEvalResult
)

class LoRACorrectionRateEvaluator(CorrectionRateEvaluator):
    """Evaluator that uses a specific LoRA adapter model name."""

    def __init__(self, lora_model_name, **kwargs):
        super().__init__(**kwargs)
        self.lora_model_name = lora_model_name

    async def _call_model(self, prompt):
        import aiohttp
        url = f"{self.model_server_url}/chat/completions"
        payload = {
            "model": self.lora_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.7,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        print(f"  Model error {resp.status}: {body[:200]}")
                        return ""
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  Model call failed: {e}")
            return ""

    async def _call_model_with_history(self, prompt, history):
        import aiohttp
        url = f"{self.model_server_url}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        for h in history:
            messages.append({"role": h["speaker"], "content": h["content"]})
        payload = {
            "model": self.lora_model_name,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.7,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        print(f"  Model error {resp.status}: {body[:200]}")
                        return ""
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  Model call failed: {e}")
            return ""


async def main():
    # Load test prompts
    test_prompts = []
    with open(TEST_DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                prompt = d.get("种子提示词", d.get("prompt", ""))
                if prompt:
                    test_prompts.append({
                        "prompt": prompt,
                        "category": d.get("类别", d.get("category", "")),
                    })

    # Dedup
    seen = set()
    unique = []
    for p in test_prompts:
        if p["prompt"] not in seen:
            seen.add(p["prompt"])
            unique.append(p)
    test_prompts = unique

    print(f"Test prompts: {len(test_prompts)}")

    # Run evaluation
    evaluator = LoRACorrectionRateEvaluator(
        lora_model_name=MODEL_NAME,
        model_server_url=MODEL_SERVER,
        api_key=API_KEY,
        base_url=BASE_URL,
        sim_model=SIM_MODEL,
        max_concurrent=3,
    )

    print(f"Starting post-training evaluation with model={MODEL_NAME}...")
    t0 = time.time()
    result = await evaluator.evaluate(test_prompts)
    elapsed = time.time() - t0

    # Load baseline
    baseline = {}
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, encoding="utf-8") as f:
            baseline = json.load(f)

    # Build output
    post_dict = result.to_dict()
    post_dict["model"] = f"Qwen3-4B + LoRA (checkpoint global_step_10)"
    post_dict["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    post_dict["eval_time_seconds"] = round(elapsed, 1)

    # Compute delta vs baseline
    if baseline:
        bs = baseline["summary"]
        pre_rate = bs[list(bs.keys())[4]]  # 纠错率
        post_dict["pre_纠错率"] = pre_rate
        post_dict["Δ纠错率"] = round(post_dict["纠错率"] - pre_rate, 4)
        post_dict["baseline_timestamp"] = baseline["timestamp"]

    # Per-category breakdown
    per_cat = {}
    for r in result.per_prompt:
        cat = r.category if r.category else "unknown"
        if cat not in per_cat:
            per_cat[cat] = {"total": 0, "direct_pass": 0, "corrected_pass": 0, "failed": 0}
        per_cat[cat]["total"] += 1
        if r.outcome == "直接通过":
            per_cat[cat]["direct_pass"] += 1
        elif r.outcome == "纠错后通过":
            per_cat[cat]["corrected_pass"] += 1
        else:
            per_cat[cat]["failed"] += 1

    # Add correction_rate per category
    for cat in per_cat:
        t = per_cat[cat]["total"]
        d = per_cat[cat]["direct_pass"]
        per_cat[cat]["correction_rate"] = round((t - d) / t, 4) if t > 0 else 0.0

    post_dict["per_category"] = per_cat

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(post_dict, f, ensure_ascii=False, indent=2)

    print(f"\n=== Results ===")
    print(f"Total: {result.total}")
    print(f"Direct Pass: {result.直接通过}")
    print(f"Corrected Pass: {result.纠错后通过}")
    print(f"Failed: {result.失败}")
    print(f"Correction Rate: {result.纠错率:.4f}")
    if "Δ纠错率" in post_dict:
        print(f"Pre Rate (baseline): {post_dict['pre_纠错率']:.4f}")
        print(f"Delta: {post_dict['Δ纠错率']:+.4f}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"\nSaved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
