"""Validate rubrics_dynamic.json — check scoring distribution on test answers."""
import json, sys
sys.path.insert(0, '/data/wangye/trainable-openclaw')

from trainable_openclaw.evaluation.judge import JudgeExecutor
from trainable_openclaw.evaluation.rubric import Rubric

data = json.load(open('data/rubrics_dynamic.json'))
rubric_objs = [Rubric.from_dict(r) for r in data]
active = rubric_objs

tests = [
    {
        'prompt': 'what is machine learning',
        'answer': 'ML is AI that makes computers learn. Supervised and unsupervised. Makes machines smart.'
    },
    {
        'prompt': 'sort list python',
        'answer': 'def sort_list(arr): return sorted(arr)'
    },
    {
        'prompt': 'describe photosynthesis process',
        'answer': (
            'Photosynthesis is the process where plants convert light energy, water, '
            'and CO2 into glucose and oxygen. It occurs in chloroplasts. The light '
            'reactions (on thylakoid membranes) photolyze water to produce ATP and '
            'NADPH. The dark reactions (Calvin cycle in stroma) fix CO2 into glucose. '
            'Photosynthesis provides the energy foundation for nearly all life on Earth.'
        )
    },
]

judge = JudgeExecutor(
    api_key='sk-906ad0dc48354e7aba594ef6d9aa5be6',
    base_url='https://api.deepseek.com',
    model='deepseek-v4-flash',
    enable_thinking=False,
    use_merged=True,
)

all_scores = []
for t in tests:
    try:
        results = judge.score_answers_sync(
            prompt=t['prompt'], answers=[t['answer']], rubrics=active
        )
        rewards = judge.compute_grpo_rewards(results, reward_mode='mean')
        print("prompt={} answer={}... reward={:.3f}".format(
            t['prompt'][:40], t['answer'][:50], rewards[0]))
        all_scores.append(rewards[0])
    except Exception as e:
        print('Error: {}'.format(e))
        all_scores.append(0.0)

mean = sum(all_scores) / len(all_scores) if all_scores else 0
spread = max(all_scores) - min(all_scores) if len(all_scores) > 1 else 0
positive = sum(1 for s in all_scores if s > 0.5)
passed = mean >= 0.3 and positive >= 1 and spread > 0.1
print('')
print('mean={:.3f}  positive(>0.5)={}/{}  spread={:.3f}  PASS={}'.format(
    mean, positive, len(all_scores), spread, passed))
