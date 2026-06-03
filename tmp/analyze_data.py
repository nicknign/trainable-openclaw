import json, os
os.chdir('/data/wangye/trainable-openclaw')

# Train
pairs = [json.loads(l) for l in open('data/phase3_datasets/train_prompts.jsonl')]
print('Train pairs:', len(pairs))
prompts = set()
for p in pairs:
    seed = p.get('prompt','') or p.get('种子提示词','')
    prompts.add(seed.strip())
print('Unique train prompts:', len(prompts))
cats = {}
for p in pairs:
    c = p.get('类别','') or p.get('category','')
    cats[c] = cats.get(c,0)+1
for c,n in sorted(cats.items(), key=lambda x:-x[1]):
    print('  {}: {}'.format(c, n))
print('Categories:', len(cats))

# Test
tp = [json.loads(l) for l in open('data/phase3_datasets/test_prompts.jsonl')]
print('')
print('Test pairs:', len(tp))
tprompts = set()
for p in tp:
    seed = p.get('prompt','') or p.get('种子提示词','')
    tprompts.add(seed.strip())
print('Unique test prompts:', len(tprompts))
overlap = prompts & tprompts
print('Train/Test overlap:', len(overlap), 'prompts')

# Baseline eval
if os.path.exists('data/phase3_datasets/baseline_eval.json'):
    be = json.load(open('data/phase3_datasets/baseline_eval.json'))
    if isinstance(be, list):
        print('\nBaseline eval entries:', len(be))
    else:
        print('Baseline eval keys:', list(be.keys()))
