"""Generate coding-specific rubrics from coding trajectories and training pairs."""
import json
import asyncio
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter

# Add project to path
sys.path.insert(0, '/data/wangye/trainable-openclaw')

# --- Gather coding error data ---
base = '/data/wangye/trainable-openclaw/data'
traj_file = f'{base}/trajectories_high_error.jsonl'
pairs_file = f'{base}/phase3_datasets/training_pairs.jsonl'

error_cases = []  # list of {dimensions, correction_text, prompt}

# From trajectories
with open(traj_file, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d.get('类别', '') != 'coding':
            continue
        if d.get('最终判定', '') == '直接通过':
            continue  # skip direct-passes
        # Extract correction info from conversation
        corrections = []
        for msg in d.get('对话消息', []):
            if isinstance(msg, dict) and msg.get('role') == 'user':
                content = msg.get('content', '')
                if content and len(content) > 20:
                    corrections.append(content[:300])
        error_cases.append({
            'prompt': d.get('种子提示词', '')[:200],
            'corrections': corrections[:2],
            'verdict': d.get('最终判定', ''),
            'rounds': d.get('纠错次数', 0),
        })

print(f'From trajectories: {len(error_cases)} error cases')

# From training pairs
with open(pairs_file, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        cat = d.get('类别', d.get('source_category', ''))
        if 'coding' not in str(cat).lower() or 'debugging' in str(cat).lower():
            continue
        correction = d.get('纠错意见', d.get('纠错意�', ''))
        if correction:
            error_cases.append({
                'prompt': d.get('prompt', d.get('种子提示词', ''))[:200],
                'corrections': [correction[:300]],
                'verdict': 'training_pair',
                'rounds': 1,
            })

print(f'Total error cases (traj + pairs): {len(error_cases)}')

# Collect all correction texts
all_corrections = []
for ec in error_cases:
    all_corrections.extend(ec['corrections'])

correction_sample = '\n---\n'.join(all_corrections[:15])

# --- LLM: Generate coding-specific rubrics ---
api_key = os.environ.get('DEEPSEEK_API_KEY', 'sk-906ad0dc48354e7aba594ef6d9aa5be6')
base_url = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
model = 'deepseek-v4-flash'

from openai import OpenAI

client = OpenAI(api_key=api_key, base_url=base_url)

prompt = f"""你是一个严格的代码质量评审专家。请基于以下AI代码生成错误的真实案例，设计5-6条代码质量评分Rubric。

这些Rubric将用于评估AI生成的代码质量。请确保：
1. 针对代码生成场景（coding）
2. 每条rubric聚焦一个独立的代码质量维度
3. 满分10分，严格扣分制。普通代码应在4-7分，完美的才9-10分
4. 评分标准必须具体、量化、可机械执行（禁止"较好""较差"等模糊词）
5. 输出格式统一为JSON: {{"分数": <0-10>, "扣分项": ["具体错误"], "总结": "一句话"}}
6. 维度间不能重叠，总数5-6条
7. 待评代码放在{{content}}占位符位置

以下是AI在代码生成中犯的真实错误案例（纠错意见）：

{correction_sample}

=== 输出格式（严格JSON数组）===
[
  {{
    "名称": "rubric名称（如：代码语法与API正确性）",
    "评分提示词": "完整的评分prompt（含评分标准+扣分细则+JSON输出格式+{{content}}占位符）",
    "适用类别": ["coding"]
  }},
  ...
]
只输出JSON数组，不要有其他文字。"""

print('\n=== Generating coding rubrics via LLM ===')
print(f'Correction sample length: {len(correction_sample)} chars')
print(f'Prompt length: {len(prompt)} chars')

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
    max_tokens=4000,
    extra_body={"thinking": {"type": "disabled"}},
)

raw = response.choices[0].message.content.strip()
print(f'Response length: {len(raw)} chars')
print(f'Response preview: {raw[:200]}...')

# Parse JSON
if raw.startswith('```'):
    lines = raw.split('\n')
    lines = lines[1:] if len(lines) > 1 else lines
    if lines and lines[-1].startswith('```'):
        lines = lines[:-1]
    raw = '\n'.join(lines)

try:
    rubrics = json.loads(raw)
except json.JSONDecodeError:
    print('JSON parse failed, trying to extract array...')
    # Find array boundaries
    start = raw.find('[')
    end = raw.rfind(']') + 1
    if start >= 0 and end > start:
        rubrics = json.loads(raw[start:end])
    else:
        print('Failed to parse response')
        print(raw)
        sys.exit(1)

if not isinstance(rubrics, list):
    rubrics = [rubrics]

print(f'\nGenerated {len(rubrics)} coding rubrics:')
for i, r in enumerate(rubrics):
    print(f'  {i+1}. {r.get("名称", "unnamed")} [{len(r.get("评分提示词", ""))} chars]')

# Save
import hashlib
import time

output = []
for r in rubrics:
    rubric_id = hashlib.md5(r['名称'].encode()).hexdigest()[:12]
    output.append({
        'id': rubric_id,
        '名称': r['名称'],
        '评分提示词': r['评分提示词'],
        '来源模式': 'coding_rubric_engine_v1',
        '版本': 1,
        '命中次数': 0,
        '最后命中时间': 0.0,
        '状态': '活跃',
        '创建时间': time.time(),
        '适用类别': ['coding'],
    })

out_path = f'{base}/rubrics_coding.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f'\nSaved to {out_path}')
print(f'Rubric names: {[r["名称"] for r in output]}')
