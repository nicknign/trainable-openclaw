"""Evaluate model on test set — code-block extraction + format rubric + individual scoring."""
import json, sys, os, re, time

API_BASE = 'http://localhost:8000/v1'
RUBRICS_PATH = 'data/rubrics_coding_v4.json'
TEST_PATH = 'data/coding/test.jsonl'
JUDGE_API_KEY = 'sk-906ad0dc48354e7aba594ef6d9aa5be6'
JUDGE_BASE_URL = 'https://api.deepseek.com'
JUDGE_MODEL = 'deepseek-v4-flash'
MAX_TOKENS = 2048
TEMPERATURE = 0.6
SYSTEM_PROMPT = 'You are a code generator. Output clean, runnable code in ``` fences. No explanations unless the user asks for them.'
OUTPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else 'data/coding/eval_result.json'
MODEL_LABEL = sys.argv[2] if len(sys.argv) > 2 else 'Qwen3-4B'

from openai import OpenAI
vllm_client = OpenAI(api_key='EMPTY', base_url=API_BASE)

with open(TEST_PATH) as f:
    prompts = [json.loads(l)['prompt'] for l in f if l.strip()]
print(f'Test prompts: {len(prompts)}')

with open(RUBRICS_PATH) as f:
    rubric_data = json.load(f)
active = [r for r in rubric_data if r.get('状态', 'active') in ('活跃', 'active')] or rubric_data
print(f'Code rubrics: {len(active)}')

# Generate
print(f'\n===== Step 1: Generate ({MODEL_LABEL}) =====')
results = []
t_start = time.time()
for i, prompt in enumerate(prompts):
    t0 = time.time()
    try:
        resp = vllm_client.chat.completions.create(
            model='qwen', messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt},
            ], max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        rt = resp.choices[0].message.content
        think_part = ''
        if '<think>' in rt:
            think_part = '\n'.join(re.findall(r'<think>(.*?)</think>', rt, re.DOTALL))
        clean = rt.replace('<think>', '').replace('</think>', '').strip()
        code_blocks = re.findall(r'```(?:\w+)?\s*\n(.*?)```', clean, re.DOTALL)
        extracted = '\n\n'.join(code_blocks).strip() if code_blocks else clean
        if len(extracted) > 3000:
            extracted = extracted[:3000]
        results.append({'prompt': prompt, 'full_response': rt, 'think_part': think_part,
            'code_part': clean, 'extracted_code': extracted, 'has_code_blocks': len(code_blocks) > 0, 'error': None})
        print(f'  [{i+1:2d}/{len(prompts)}] {time.time()-t0:.1f}s | think={len(think_part)} blocks={len(code_blocks)} code={len(extracted)}')
    except Exception as e:
        results.append({'prompt': prompt, 'full_response': '', 'extracted_code': '', 'error': str(e)})
        print(f'  [{i+1:2d}/{len(prompts)}] ERROR: {e}')
gen_time = time.time() - t_start
n_ok = sum(1 for r in results if not r['error'])
print(f'Gen done: {gen_time:.1f}s, {n_ok}/{len(prompts)} OK')

# Score
print(f'\n===== Step 2: Individual Scoring =====')
sys.path.insert(0, '/data/wangye/trainable-openclaw')
from trainable_openclaw.evaluation.judge import JudgeExecutor
from trainable_openclaw.evaluation.rubric import Rubric

code_rubrics = [Rubric.from_dict(r) for r in active]
format_rubric = Rubric(
    id='format_check',
    名称='回答格式合规性',
    评分提示词='评估以下回答的格式合规性，给出10分制：\n- 9-10分：直接提供可执行代码，无冗余解释，格式干净专业\n- 7-8分：主要提供代码，附简要说明或注释，不影响代码提取\n- 4-6分：解释和代码混杂，解释占比较大，需人工提取代码\n- 0-3分：主要是解释/教程/思路，没有可直接使用的代码\n\n返回JSON: {"score": <0-10分数>, "deductions": ["扣分项"], "summary": "一句话"}\n\n待评估内容：{content}',
    来源模式='manual_format_check')

judge = JudgeExecutor(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL, model=JUDGE_MODEL,
    enable_thinking=False, use_merged=False)

all_scores, all_details = [], []
t_start = time.time()
total_calls = 0
for i, r in enumerate(results):
    if r['error']:
        all_scores.append(0.0); all_details.append({'error': r['error']}); continue
    t0 = time.time()
    try:
        code_for_judge = r['extracted_code'] if r['extracted_code'].strip() else r['code_part']
        code_results = judge.score_answers_sync(prompt=r['prompt'], answers=[code_for_judge], rubrics=code_rubrics)
        format_results = judge.score_answers_sync(prompt=r['prompt'], answers=[r['code_part']], rubrics=[format_rubric])
        all_rubric_scores = code_results[0].get('评分', []) + format_results[0].get('评分', [])
        score_vector = code_results[0].get('分数向量', []) + format_results[0].get('分数向量', [])
        score = sum(score_vector) / len(score_vector) if score_vector else 0
        all_scores.append(score)
        detail = {'prompt': r['prompt'][:60], 'mean_score': score,
            'code_scores': code_results[0].get('分数向量', []),
            'format_score': format_results[0].get('分数向量', [0])[0],
            'has_code_blocks': r['has_code_blocks'], 'extracted_len': len(r['extracted_code']),
            'rubric_scores': [{'name': rs.get('rubric名称', ''), '分数': rs.get('分数', 0),
                '扣分项': rs.get('扣分项', [])[:2]} for rs in all_rubric_scores]}
        all_details.append(detail)
        total_calls += len(code_rubrics) + 1
        print(f'  [{i+1:2d}/{len(results)}] {time.time()-t0:.1f}s | mean={score:.2f} code={detail["code_scores"]} fmt={detail["format_score"]:.0f}')
    except Exception as e:
        all_scores.append(0.0); all_details.append({'error': str(e)})
        print(f'  [{i+1:2d}/{len(results)}] ERROR: {e}')
score_time = time.time() - t_start

out = {
    'timestamp': time.time(), 'model': MODEL_LABEL,
    'config': {'rubrics': RUBRICS_PATH, 'code_rubric_count': len(active),
        'scoring_mode': 'individual+format', 'judge_model': JUDGE_MODEL},
    'results': [{'prompt': r['prompt'], 'full_response': r['full_response'],
        'think_len': len(r['think_part']), 'code_len': len(r['code_part']),
        'extracted_len': len(r['extracted_code']), 'has_code_blocks': r['has_code_blocks'],
        'score': all_scores[i] if i < len(all_scores) else 0,
        'detail': all_details[i] if i < len(all_details) else {}} for i, r in enumerate(results)],
    'summary': {'mean_score': sum(all_scores)/len(all_scores) if all_scores else 0,
        'min_score': min(all_scores) if all_scores else 0, 'max_score': max(all_scores) if all_scores else 0,
        'above_0.5': sum(1 for s in all_scores if s >= 0.5),
        'gen_time_s': gen_time, 'judge_time_s': score_time}}

os.makedirs('data/coding', exist_ok=True)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f'\n===== {MODEL_LABEL} RESULTS =====')
s = out['summary']
print(f'Mean: {s["mean_score"]:.4f}  Range: {s["min_score"]:.4f}-{s["max_score"]:.4f}')
print(f'>=0.5: {s["above_0.5"]}/{len(prompts)}')
print(f'Scores: {[round(x,2) for x in all_scores]}')
print(f'Gen: {gen_time:.1f}s  Judge: {score_time:.1f}s  Total: {gen_time+score_time:.1f}s')
print(f'Saved to {OUTPUT_FILE}')
