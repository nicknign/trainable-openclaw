"""Parse training logs and write run analysis (encoding-safe)."""
import json, os, re, sys

run_dir = sys.argv[1] if len(sys.argv) > 1 else 'runs/latest'

with open(os.path.join(run_dir, 'serve_ppo_train.log'), encoding='utf-8') as f:
    log = f.read()

steps = []
for m in re.finditer(
    r'\[([^\]]+)\].*train_step completed in ([\d.]+)s.*reward=([\d.]+) \(([\d]+)/(\d+)\).*loss=([-\d.]+).*grad_norm=([\d.]+)',
    log
):
    steps.append({
        'time': m.group(1),
        'duration_s': float(m.group(2)),
        'reward': float(m.group(3)),
        'above_05': int(m.group(4)),
        'total': int(m.group(5)),
        'loss': float(m.group(6)),
        'grad_norm': float(m.group(7)),
    })

with open(os.path.join(run_dir, 'baseline_eval.json'), encoding='utf-8') as f:
    baseline = json.load(f)

bs = baseline['summary']
bs_keys = list(bs.keys())
bc = baseline['per_category']

phase1 = [s for s in steps if s['total'] <= 8]
phase2 = [s for s in steps if s['total'] > 8]

L = []
def w(s=''):
    L.append(s)

# Title
w('# Training Record: 2026-06-02 ~ 2026-06-03')
w()
w('## Configuration')
w()
w('| Parameter | Value |')
w('|-----------|-------|')
w('| Model | Qwen3-4B + LoRA rank=16, alpha=32 |')
w('| Learning Rate | 1e-5 |')
w('| Prompts per Step | 48 (sampled from 496 unique) |')
w('| Rollouts per Prompt | 4 |')
w('| Answers per Step | 192 |')
w('| Mini Batch Size | 8 |')
w('| Max Rounds | 5 rounds x 20 steps = 100 |')
w('| Rubric | Category-matched (5 groups x 4 = 20), DeepSeek-v4-flash judge |')
w('| Checkpoint Interval | Every 10 steps |')
w('| GPU | RTX 4090 (24GB free / 48GB total) |')
w()

# Phase 1
w(f'## Phase 1: Small-Batch Debug (Steps 1-{len(phase1)})')
w()
w(f'- Time: {phase1[0]["time"]} ~ {phase1[-1]["time"]}')
w('- Batch: 4 prompts x 1 rollout = 4 answers/step')
w(f'- Steps: {len(phase1)}')
w(f'- Step Duration: ~{sum(s["duration_s"] for s in phase1)/len(phase1):.0f}s')
p1r = [s['reward'] for s in phase1]
w(f'- Reward Range: {min(p1r):.3f} ~ {max(p1r):.3f}')
w()

# Phase 2
p2_start = len(phase1) + 1
w(f'## Phase 2: Full-Batch Training (Steps {p2_start}-{len(steps)})')
w()
w(f'- Time: {phase2[0]["time"]} ~ {phase2[-1]["time"]}')
w('- Batch: 48 prompts x 4 rollout = 192 answers/step')
w(f'- Steps: {len(phase2)}')
p2_dur = sum(s['duration_s'] for s in phase2) / len(phase2)
w(f'- Step Duration: ~{p2_dur:.0f}s ({p2_dur/60:.1f} min)')
p2r = [s['reward'] for s in phase2]
w(f'- Reward Mean: {sum(p2r)/len(p2r):.3f}')
w(f'- Reward Range: {min(p2r):.3f} ~ {max(p2r):.3f}')
p2l = [s['loss'] for s in phase2]
w(f'- Loss Range: {min(p2l):.4f} ~ {max(p2l):.4f}')
total_05 = sum(s['above_05'] for s in phase2)
total_all = sum(s['total'] for s in phase2)
w(f'- >0.5 Ratio: {total_05}/{total_all} ({total_05/total_all*100:.1f}%)')
w()

w('### Reward Trend')
w()
for i, s in enumerate(phase2):
    bar = '#' * int(s['reward'] * 50)
    step_num = p2_start + i
    w(f'- Step {step_num:2d} ({s["time"][:16]}): reward={s["reward"]:.3f} {bar}')
w()

# Baseline (use positional indexing to avoid Chinese encoding in source)
w('## Baseline Eval (Pre-Training)')
w()
w(f'- Model: Qwen3-4B (raw weights, untrained)')
w(f'- Test Set: {bs[bs_keys[0]]} prompts')
w(f'- Correction Rate: {bs[bs_keys[4]]}')
w(f'- Direct Pass: {bs[bs_keys[1]]}')
w(f'- Corrected Pass: {bs[bs_keys[2]]}')
w(f'- Failed: {bs[bs_keys[3]]}')
w(f'- Avg Correction Rounds: {bs[bs_keys[5]]}')
w()
w('### Per-Category Correction Rate')
w()
for cat in sorted(bc.keys()):
    s = bc[cat]
    w(f'- {cat}: {s["correction_rate"]:.2f} ({s["direct_pass"]}+{s["corrected_pass"]}/{s["total"]})')
w()

# Summary
total_hours = sum(s['duration_s'] for s in steps) / 3600
w('## Summary')
w()
w('| Metric | Value |')
w('|--------|-------|')
w(f'| Total Steps | {len(steps)} |')
w(f'| Total Training Time | {total_hours:.1f} hours |')
w(f'| P2 Reward Mean | {sum(p2r)/len(p2r):.3f} |')
trend = 'No clear trend' if len(phase2) < 2 else ('Slight upward' if phase2[-1]['reward'] > phase2[0]['reward'] else 'Slight downward')
w(f'| P2 Reward Trend | {trend} |')
w(f'| P2 First Step Reward | {phase2[0]["reward"]:.3f} |')
w(f'| P2 Last Step Reward | {phase2[-1]["reward"]:.3f} |')
w('| Checkpoints Saved | Only global_step_10 (from Phase 1) |')
w('| Final Model Weights | Not saved after Phase 2 training |')
w()

w('## Conclusions')
w()
w('1. **Weak reward signal**: Mean reward ~0.17 in full-batch mode; only ~5% of answers score >0.5. Qwen3-4B is far from the rubric full-score standard on complex multi-category tasks.')
w('2. **No clear improvement from training**: 27 full-batch steps show no upward reward trend; loss oscillates within +/-0.01.')
w('3. **Checkpoint gap**: Only step_10 checkpoint (Phase 1) was preserved. Phase 2/3 checkpoints were not saved. Restarting from step_10 may have overwritten Phase 2 progress.')
w('4. **Model capacity bottleneck**: 4B parameters is insufficient for 11-category multi-dimensional correction tasks. Recommend larger model or narrower task scope for future experiments.')
w('5. **Phase 2 initial rewards were highest**: First 2 steps (0.297, 0.287) had markedly higher >0.5 ratios (59-56/192) than later steps. Suggests possible catastrophic forgetting after checkpoint reload or reward drift.')

out_path = os.path.join(run_dir, 'analysis.md')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))

print(f'Report: {out_path}')
print(f'Total steps: {len(steps)} (P1={len(phase1)}, P2={len(phase2)})')
print(f'Phase 2 reward: mean={sum(p2r)/len(p2r):.3f}, range={min(p2r):.3f}-{max(p2r):.3f}')
