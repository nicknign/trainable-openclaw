# Training Summary — Coding GRPO on Qwen3-4B

**Date**: 2026-06-02 to 2026-06-05 (4 days, mostly automated)
**Base Model**: Qwen3-4B + LoRA rank=16, alpha=32
**Training Method**: GRPO (Group Relative Policy Optimization) with LLM Judge reward
**Training Data**: 80 coding prompts (`data/coding/train_all.jsonl`), zero overlap with test set
**Rubrics**: 4 coding rubrics (`data/rubrics_coding_v4.json`), individual scoring, DeepSeek-v4-flash, no thinking

---

## Key Metric: 5x Reward Jump After Fixing Response Truncation

| Phase | Steps | Config | Response Len | Mean Reward | TPS |
|-------|-------|--------|-------------|-------------|-----|
| 1: Debug | 15 | 2p×2r=4 | **504 tok** (truncated) | 0.093 | 86 |
| 2: Batch48 | 27 | 48p×4r=192 | **497 tok** (truncated) | 0.184 | 120 |
| 3: Batch32 | 25 | 32p×4r=128 | **500 tok** (truncated) | 0.031 | 132 |
| 4: Transition | 6 | mixed | **492 tok** (truncated) | 0.096 | 122 |
| 5: Full Response | 31 | 16p×4r=64 | **1925 tok** | **0.588** | 247 |

**The critical fix**: Changing `response_length` from 512 (default in rollout.yaml) to 4096 tokens. Before this fix, ~75% of generated code was truncated — the judge was scoring incomplete code fragments, resulting in near-zero rewards. After the fix, rewards jumped from 0.06–0.14 to 0.67 in the first full-length step.

---

## Phase 5 Detail (31 Steps, Full Response Length)

**Config**: 16 prompts/step × 4 rollouts = 64 answers/step, prompt_length=2048, response_length=4096, max_model_len=8192, lr=1e-5, temperature=0.6

**Results**:
- Mean reward: **0.588** (range: 0.427–0.692)
- Reward >0.5: **60.8%** (1206/1984 answers)
- Mean loss: 0.0450 (range: 0.0002–0.1155)
- Mean grad_norm: 0.0845
- Mean step duration: 516s (~8.6 min)
- Total Phase 5 time: 4.4 hours

**3-Step Moving Average Trend**:
```
Steps 1-3:   0.577  ████████████████████████████████
Steps 5-7:   0.579  ████████████████████████████████
Steps 10-12: 0.638  ████████████████████████████████████████  (peak)
Steps 15-17: 0.630  ███████████████████████████████████████
Steps 20-22: 0.641  ████████████████████████████████████████  (peak)
Steps 25-27: 0.631  ███████████████████████████████████████
Steps 29-31: 0.566  ████████████████████████████████
```

**Trend assessment**: Oscillating between 0.52–0.64, no clear monotonic upward trend. First 5 mean: 0.571, Last 5 mean: 0.599 — very slight improvement (+0.028). The oscillation is likely driven by prompt sampling variation (different 16 prompts each step from the 80-prompt pool).

---

## Response Length (rlen) Evolution

The most telling metric across all 104 steps:

```
Phase 1-4:  rlen = 492–504 tokens  (ALL truncated at 512 default)
Phase 5:    rlen = 1372–2588 tokens (full code generation, response_length=4096)
```

Before the fix, the model was training on fragments. After, it sees complete code with think blocks, code fences, and full implementations.

---

## Loss and Gradient Behavior

- **Phase 1-4** (truncated): Loss often stuck at 0.0000. When non-zero, very small (-0.06 to +0.02). Grad norm 0.02–0.15.
- **Phase 5** (full): Loss consistently non-zero (0.0002–0.1155), mean 0.045. Grad norm 0.05–0.23. The model is actually learning — advantage signal comes from real code quality differences, not truncation artifacts.

---

## Checkpoints

| Step | Time | Phase Context |
|------|------|---------------|
| `global_step_10` | Jun 5 17:48 | Phase 5 (full response), ~step 10 of final run |
| `global_step_20` | Jun 5 19:17 | Phase 5 (full response), ~step 20 of final run |

Both checkpoints are from Phase 5 with full response length. LoRA adapter extracted: 504 params, rank=16, alpha=32, 126MB safetensors.

No checkpoint was saved from the truncated phases (1-4), since the `save_ckpt_interval=10` was only configured in the final `start_train.sh`.

---

## Total Compute

| Metric | Value |
|--------|-------|
| Total training steps | 104 |
| Total GPU time | ~16 hours |
| Effective training (Phase 5) | 4.4 hours |
| Wasted (truncated phases) | ~11.6 hours |
| DeepSeek API calls (Phase 5) | 31 × 64 × 5 = ~9,920 calls |
| Estimated API cost | ~$10–15 |

---

## Key Takeaways

1. **Response truncation was the #1 blocker**. The rollout.yaml default of `response_length=512` silently truncated 75% of generated code. Always verify `rlen` in training logs matches expectations.

2. **Reward signal is healthy at ~0.59**. With full response length and 4 coding rubrics + individual scoring, the judge provides meaningful differentiation (60% of answers score >0.5).

3. **No clear training improvement trend**. 31 steps of Phase 5 show oscillation without monotonic improvement. Possible causes:
   - Prompt sampling noise (different 16/80 prompts each step)
   - Learning rate too low (1e-5) for 31 steps
   - Rubric scoring has inherent variance
   - Need more steps to see trend

4. **Loss is learning**. Non-zero loss (mean 0.045) and meaningful grad norms confirm the model is updating. This is a major improvement over earlier phases where loss was stuck at 0.

5. **Phase 3 (32p×4r) had near-zero reward (0.031)**. Likely caused by a bug or config issue in that server restart — needs investigation.

---

## Next Steps

- Run test set evaluation (20 prompts, `data/coding/test.jsonl`) on the trained checkpoint vs baseline
- Compare results to quantify actual coding improvement
- Consider: more steps, higher LR, or pairwise ranking instead of scalar rubric scoring
- Expand training data beyond 80 prompts for more stable gradients

---

## Files Saved

| File | Description |
|------|-------------|
| `reports/phase3_train.log` (6.7MB) | Full server log |
| `reports/serve_ppo_train.log` (1.3MB) | Per-step training metrics |
| `reports/generation_samples.json` (160KB) | First 100 cached generations with token counts |
| `reports/start_train.sh` | Training launch script |
| `reports/training_summary_2026-06-06.md` | This report |
