"""Test set evaluation — code-block extraction + format rubric + individual scoring."""
import json
import time
import os
import sys
import re

API_BASE = "http://localhost:8000/v1"
RUBRICS_PATH = "data/rubrics_coding_v4.json"
TEST_PATH = "data/coding/test.jsonl"
JUDGE_API_KEY = "sk-906ad0dc48354e7aba594ef6d9aa5be6"
JUDGE_BASE_URL = "https://api.deepseek.com"
JUDGE_MODEL = "deepseek-v4-flash"
MAX_TOKENS = 2048
TEMPERATURE = 0.6

SYSTEM_PROMPT = "You are a code generator. Output clean, runnable code in ``` fences. No explanations unless the user asks for them."

from openai import OpenAI
vllm_client = OpenAI(api_key="EMPTY", base_url=API_BASE)

# Load test prompts
with open(TEST_PATH, 'r', encoding='utf-8') as f:
    prompts = [json.loads(line)['prompt'] for line in f if line.strip()]
print(f'Test prompts: {len(prompts)}')

# Load rubrics
with open(RUBRICS_PATH, 'r', encoding='utf-8') as f:
    rubric_data = json.load(f)
active = [r for r in rubric_data if r.get('状态', 'active') in ('活跃', 'active')]
if not active:
    active = rubric_data
print(f'Code rubrics: {len(active)}: {[r["名称"] for r in active]}')

# ---- Step 1: Generate full responses ----
print(f'\n{"="*60}')
print('Step 1: Generating FULL responses from Qwen3-4B')
print(f'{"="*60}')

results = []
t_start = time.time()
for i, prompt in enumerate(prompts):
    t0 = time.time()
    try:
        resp = vllm_client.chat.completions.create(
            model="qwen",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        response_text = resp.choices[0].message.content

        # Clean up response: remove think tags (keep content for code extraction)
        think_part = ""
        if "<think>" in response_text:
            # Extract well-formed think blocks
            think_blocks = re.findall(r'<think>(.*?)</think>', response_text, re.DOTALL)
            think_part = "\n".join(think_blocks)
        # Strip think XML tags but keep the text between them
        clean_text = response_text.replace("<think>", "").replace("</think>", "").strip()
        codepart_for_fmt = clean_text  # For format rubric: full cleaned text

        # Extract ``` code blocks for judge scoring
        code_blocks = re.findall(r'```(?:\w+)?\s*\n(.*?)```', clean_text, re.DOTALL)
        if code_blocks:
            extracted_code = "\n\n".join(code_blocks).strip()
        else:
            extracted_code = clean_text
        # Truncate for judge — prevents JSON parse failures on 8K+ char responses
        MAX_CODE_LEN = 3000
        if len(extracted_code) > MAX_CODE_LEN:
            extracted_code = extracted_code[:MAX_CODE_LEN]

        results.append({
            "prompt": prompt,
            "full_response": response_text,
            "think_part": think_part,
            "code_part": codepart_for_fmt,
            "extracted_code": extracted_code,
            "has_code_blocks": len(code_blocks) > 0,
            "error": None,
        })
        t1 = time.time()
        print(f'  [{i+1:2d}/{len(prompts)}] {t1-t0:.1f}s | think={len(think_part)} code_blocks={len(code_blocks)} extracted={len(extracted_code)} | {prompt[:40]}...')
    except Exception as e:
        results.append({"prompt": prompt, "full_response": "", "extracted_code": "", "error": str(e)})
        print(f'  [{i+1:2d}/{len(prompts)}] ERROR: {e}')

gen_time = time.time() - t_start
n_ok = sum(1 for r in results if not r['error'])
print(f'Generation done: {gen_time:.1f}s ({gen_time/len(prompts):.1f}s/prompt), {n_ok}/{len(prompts)} OK')

# ---- Step 2: Score with individual scoring ----
print(f'\n{"="*60}')
print(f'Step 2: Individual scoring ({len(active)} code rubrics + 1 format rubric)')
print(f'{"="*60}')

sys.path.insert(0, '/data/wangye/trainable-openclaw')
from trainable_openclaw.evaluation.judge import JudgeExecutor
from trainable_openclaw.evaluation.rubric import Rubric

code_rubrics = [Rubric.from_dict(r) for r in active]

# Add format-compliance rubric
format_rubric = Rubric(
    id="format_check",
    名称="回答格式合规性",
    评分提示词="""请评估以下回答的格式合规性（满分10分）：
- 9-10分：直接提供可执行代码，无冗余解释，格式简洁专业
- 7-8分：主要提供代码，附带简短说明或注释，不影响代码提取
- 4-6分：代码与解释混杂，解释占比大，需要人工提取代码
- 0-3分：主要是解释/教程/思路描述，几乎无可直接使用的代码

输出JSON: {"分数": <0-10整数>, "扣分项": ["具体问题"], "总结": "一句话"}

待评内容：{content}""",
    来源模式="manual_format_check",
)

judge = JudgeExecutor(
    api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL, model=JUDGE_MODEL,
    enable_thinking=False, use_merged=False,
)

all_scores = []
all_details = []
t_start = time.time()
total_calls = 0

for i, r in enumerate(results):
    if r['error']:
        all_scores.append(0.0)
        all_details.append({'error': r['error']})
        continue

    t0 = time.time()
    try:
        # Score code rubrics on extracted code blocks (pure code)
        code_for_judge = r['extracted_code'] if r['extracted_code'].strip() else r['code_part']
        code_results = judge.score_answers_sync(
            prompt=r['prompt'],
            answers=[code_for_judge],
            rubrics=code_rubrics,
        )
        # Score format rubric on post-think full response
        format_results = judge.score_answers_sync(
            prompt=r['prompt'],
            answers=[r['code_part']],
            rubrics=[format_rubric],
        )

        # Merge scores
        all_rubric_scores = code_results[0].get("评分", []) + format_results[0].get("评分", [])
        all_score_vector = code_results[0].get("分数向量", []) + format_results[0].get("分数向量", [])
        score = sum(all_score_vector) / len(all_score_vector) if all_score_vector else 0

        all_scores.append(score)
        detail = {
            'prompt': r['prompt'][:60],
            'mean_score': score,
            'code_scores': code_results[0].get("分数向量", []),
            'format_score': format_results[0].get("分数向量", [0])[0],
            'has_code_blocks': r['has_code_blocks'],
            'extracted_len': len(r['extracted_code']),
            'rubric_scores': [
                {
                    'name': rs.get("rubric名称", ""),
                    'score': rs.get("分数", 0),
                    'deductions': rs.get("扣分项", [])[:2],
                }
                for rs in all_rubric_scores
            ],
        }
        all_details.append(detail)

        t1 = time.time()
        calls_for_this = len(code_rubrics) + 1
        total_calls += calls_for_this
        print(f'  [{i+1:2d}/{len(results)}] {t1-t0:.1f}s | mean={score:.2f} code_scores={detail["code_scores"]} fmt={detail["format_score"]:.0f} | {r["prompt"][:40]}...')
    except Exception as e:
        all_scores.append(0.0)
        all_details.append({'error': str(e)})
        print(f'  [{i+1:2d}/{len(results)}] ERROR: {e}')

score_time = time.time() - t_start

# ---- Summary ----
print(f'\n{"="*60}')
print('EVALUATION RESULTS')
print(f'{"="*60}')
print(f'Code rubrics: {len(active)} + 1 format rubric')
print(f'Test prompts: {len(prompts)}')
print(f'Valid responses: {n_ok}')
print(f'Total API calls: {total_calls}')
print(f'Mean score: {sum(all_scores)/len(all_scores):.4f}')
print(f'Score range: {min(all_scores):.4f} ~ {max(all_scores):.4f}')
above_5 = sum(1 for s in all_scores if s >= 0.5)
print(f'Score >= 0.5: {above_5}/{len(prompts)}')
print(f'Scores: {[f"{s:.2f}" for s in all_scores]}')
print(f'Gen time: {gen_time:.1f}s')
print(f'Judge time: {score_time:.1f}s')
print(f'Total time: {gen_time+score_time:.1f}s')

# Save
out = {
    'timestamp': time.time(),
    'config': {'rubrics': RUBRICS_PATH, 'code_rubric_count': len(active),
               'scoring_mode': 'individual+format', 'model': JUDGE_MODEL},
    'results': [{
        'prompt': r['prompt'],
        'full_response': r['full_response'],
        'think_len': len(r['think_part']),
        'code_len': len(r['code_part']),
        'extracted_len': len(r['extracted_code']),
        'has_code_blocks': r['has_code_blocks'],
        'score': all_scores[i] if i < len(all_scores) else 0,
        'detail': all_details[i] if i < len(all_details) else {},
    } for i, r in enumerate(results)],
    'summary': {
        'mean_score': sum(all_scores)/len(all_scores) if all_scores else 0,
        'min_score': min(all_scores) if all_scores else 0,
        'max_score': max(all_scores) if all_scores else 0,
        'above_0.5': above_5,
        'gen_time_s': gen_time, 'judge_time_s': score_time,
    },
}
os.makedirs('data/coding', exist_ok=True)
with open('data/coding/testset_eval.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'Saved to data/coding/testset_eval.json')
