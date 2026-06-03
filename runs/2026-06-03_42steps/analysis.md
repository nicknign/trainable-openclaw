# Training Record: 2026-06-02 ~ 2026-06-03

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-4B + LoRA rank=16, alpha=32 |
| Learning Rate | 1e-5 |
| Prompts per Step | 48 (sampled from 496 unique) |
| Rollouts per Prompt | 4 |
| Answers per Step | 192 |
| Mini Batch Size | 8 |
| Max Rounds | 5 rounds x 20 steps = 100 |
| Rubric | Category-matched (5 groups x 4 = 20), DeepSeek-v4-flash judge |
| Checkpoint Interval | Every 10 steps |
| GPU | RTX 4090 (24GB free / 48GB total) |

## Phase 1: Small-Batch Debug (Steps 1-15)

- Time: 2026-06-02 22:20:06 ~ 2026-06-02 22:42:24
- Batch: 4 prompts x 1 rollout = 4 answers/step
- Steps: 15
- Step Duration: ~25s
- Reward Range: 0.000 ~ 0.350

## Phase 2: Full-Batch Training (Steps 16-42)

- Time: 2026-06-02 23:14:02 ~ 2026-06-03 11:41:29
- Batch: 48 prompts x 4 rollout = 192 answers/step
- Steps: 27
- Step Duration: ~887s (14.8 min)
- Reward Mean: 0.184
- Reward Range: 0.142 ~ 0.297
- Loss Range: -0.0016 ~ 0.0132
- >0.5 Ratio: 352/5184 (6.8%)

### Reward Trend

- Step 16 (2026-06-02 23:14): reward=0.297 ##############
- Step 17 (2026-06-02 23:31): reward=0.287 ##############
- Step 18 (2026-06-02 23:49): reward=0.191 #########
- Step 19 (2026-06-03 05:04): reward=0.179 ########
- Step 20 (2026-06-03 05:17): reward=0.186 #########
- Step 21 (2026-06-03 06:09): reward=0.185 #########
- Step 22 (2026-06-03 06:35): reward=0.225 ###########
- Step 23 (2026-06-03 06:53): reward=0.179 ########
- Step 24 (2026-06-03 07:06): reward=0.184 #########
- Step 25 (2026-06-03 07:19): reward=0.142 #######
- Step 26 (2026-06-03 07:32): reward=0.156 #######
- Step 27 (2026-06-03 07:45): reward=0.180 #########
- Step 28 (2026-06-03 07:59): reward=0.171 ########
- Step 29 (2026-06-03 08:13): reward=0.185 #########
- Step 30 (2026-06-03 08:40): reward=0.158 #######
- Step 31 (2026-06-03 08:55): reward=0.178 ########
- Step 32 (2026-06-03 09:09): reward=0.154 #######
- Step 33 (2026-06-03 09:24): reward=0.177 ########
- Step 34 (2026-06-03 09:40): reward=0.173 ########
- Step 35 (2026-06-03 09:55): reward=0.190 #########
- Step 36 (2026-06-03 10:10): reward=0.190 #########
- Step 37 (2026-06-03 10:25): reward=0.176 ########
- Step 38 (2026-06-03 10:40): reward=0.186 #########
- Step 39 (2026-06-03 10:55): reward=0.154 #######
- Step 40 (2026-06-03 11:12): reward=0.146 #######
- Step 41 (2026-06-03 11:27): reward=0.170 ########
- Step 42 (2026-06-03 11:41): reward=0.174 ########

## Baseline Eval (Pre-Training)

- Model: Qwen3-4B (raw weights, untrained)
- Test Set: 80 prompts
- Correction Rate: 0.6
- Direct Pass: 32
- Corrected Pass: 36
- Failed: 12
- Avg Correction Rounds: 0.95

### Per-Category Correction Rate

- brainstorming: 0.67 (5+7/15)
- coding: 0.62 (6+7/16)
- copywriting: 0.58 (5+6/12)
- creative writing: 0.44 (5+2/9)
- debating: 0.67 (1+2/3)
- debugging: 0.67 (1+2/3)
- explanation: 0.25 (3+1/4)
- instruction following: 0.33 (2+1/3)
- logical reasoning: 1.00 (0+3/3)
- math: 0.67 (3+3/9)
- translation: 0.67 (1+2/3)

## Post-Training Eval (Checkpoint global_step_10)

- Evaluated Model: Qwen3-4B + LoRA from checkpoint step_10
- Test Set: 78 prompts
- Correction Rate: **0.8846** (higher = worse)
- Direct Pass: 9 (11.5%)
- Corrected Pass: 34 (43.6%)
- Failed: 35 (44.9%)
- Avg Correction Rounds: 2.04
- Eval Time: 888.9s (~15 min)

### Baseline vs Post-Training Comparison

| Metric | Baseline (Raw Qwen3-4B) | Post-Training (ckpt step_10) | Delta |
|--------|------------------------|------------------------------|-------|
| Correction Rate | 0.6000 | 0.8846 | **+0.2846 <<<** |
| Direct Pass | 32 (40%) | 9 (11.5%) | **-23 <<<** |
| Failed | 12 (15%) | 35 (44.9%) | **+23 <<<** |
| Avg Rounds | 0.95 | 2.04 | +1.09 |

**Result: Significant degradation.** Correction rate increased from 60% to 88%, meaning the model needs corrections on 88% of test prompts (vs 60% baseline). Direct passes dropped from 32 to 9.

### Per-Category Delta

| Category | Baseline | Post | Delta | Verdict |
|----------|----------|------|-------|---------|
| explanation | 0.2500 | 1.0000 | +0.7500 | Severe degradation |
| instruction following | 0.3333 | 1.0000 | +0.6667 | Severe degradation |
| brainstorming | 0.6667 | 1.0000 | +0.3333 | Degradation |
| debugging | 0.6667 | 1.0000 | +0.3333 | Degradation |
| translation | 0.6667 | 1.0000 | +0.3333 | Degradation |
| coding | 0.6250 | 0.9375 | +0.3125 | Degradation |
| copywriting | 0.5833 | 0.8333 | +0.2500 | Degradation |
| creative writing | 0.4444 | 0.6667 | +0.2223 | Degradation |
| math | 0.6667 | 0.8889 | +0.2222 | Degradation |
| debating | 0.6667 | 0.6667 | +0.0000 | No change |
| logical reasoning | 1.0000 | 0.6667 | **-0.3333** | Improvement |

10 of 11 categories worsened. Only logical reasoning improved.

## Summary

| Metric | Value |
|--------|-------|
| Total Steps | 42 |
| Total Training Time | 6.8 hours |
| P2 Reward Mean | 0.184 |
| P2 Reward Trend | Slight downward |
| Checkpoints Saved | Only global_step_10 (from Phase 1) |
| Baseline Correction Rate | 0.60 (60% need correction) |
| Post-Training Correction Rate | **0.88 (88% need correction)** |
| Delta Correction Rate | **+0.28 (model got WORSE)** |

## Conclusions

1. **Training degraded model quality on the available checkpoint**: The only saved checkpoint (step_10) shows significantly worse correction rate than baseline (0.88 vs 0.60). Direct passes dropped from 32 to 9.
2. **Phase 1 checkpoint was too early**: Step 10 was saved during small-batch debugging when rewards were mostly 0 and the model was likely learning noise rather than signal. It captured an intermediate poor state.
3. **Phase 2 checkpoints were lost**: Steps 16-42 ran full-batch training with better rewards (up to 0.297), but `save_ckpt_interval=10` failed to trigger after the restart. The P2 checkpoint gap means we cannot evaluate the later, potentially better weights.
4. **GRPO training can be destructive**: Without proper checkpoint management, early-stage training can degrade model quality through catastrophic forgetting or overfitting to reward noise.
5. **Qwen3-4B capacity likely insufficient**: Reward signal stayed at ~0.17 with no upward trend across 27 full-batch steps. The 11-category multi-dimensional rubric task appears beyond 4B parameter capability.
6. **Recommendations for next run**:
   - Fix checkpoint saving to persist across restarts
   - Start with larger model (7B+ or 14B+)
   - Narrow task scope to 2-3 categories initially
   - Run evaluation at every checkpoint for progress tracking
   - Increase learning rate or use learning rate warmup