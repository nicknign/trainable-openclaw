#!/usr/bin/env python3
"""Quick import verification for Phase 3 modules."""
import sys
sys.path.insert(0, '/data/wangye/trainable-openclaw')
sys.path.insert(0, '/root/autodl-tmp/wangye/trainable-openclaw/verl-main-0516')

tests = [
    ('reward_bridge', 'trainable_openclaw.training.reward_bridge', 'RewardBridge'),
    ('correction_rate', 'trainable_openclaw.evaluation.correction_rate', 'CorrectionRateEvaluator'),
    ('serve_ppo new fn', 'verl.trainer.serve_ppo', '_load_trajectory_data'),
    ('judge', 'trainable_openclaw.evaluation.judge', 'JudgeExecutor'),
    ('rubric', 'trainable_openclaw.evaluation.rubric', 'Rubric'),
    ('feedback', 'trainable_openclaw.evaluation.feedback', 'FeedbackAnalyzer'),
    ('trajectory_eval', 'trainable_openclaw.evaluation.trajectory_eval', 'grade_trajectory'),
]

passed = 0
for name, mod, attr in tests:
    try:
        m = __import__(mod, fromlist=[attr])
        getattr(m, attr)
        print(f"  {name:25s} OK")
        passed += 1
    except Exception as e:
        print(f"  {name:25s} FAIL: {e}")

print(f"\n{passed}/{len(tests)} imports successful")
