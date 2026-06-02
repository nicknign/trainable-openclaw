#!/bin/bash
# RubricEngine — 动态分析错误案例 + 生成/合并/精炼 Rubrics
# 用法: bash scripts/update_rubrics.sh [轨迹文件1] [轨迹文件2] ...
set -e

cd /data/wangye/trainable-openclaw
export LD_LIBRARY_PATH=/data/anaconda3/lib:$LD_LIBRARY_PATH

INPUT_FILES="${@:-data/trajectories_500.jsonl data/trajectories_high_error.jsonl}"
OUTPUT="${OUTPUT:-data/rubrics_dynamic.json}"
API_KEY="${DEEPSEEK_API_KEY:-sk-906ad0dc48354e7aba594ef6d9aa5be6}"
MODEL="${MODEL:-deepseek-v4-flash}"
MAX_RUBRICS="${MAX_RUBRICS:-8}"

echo "=== RubricEngine ==="
echo "Input:  $INPUT_FILES"
echo "Output: $OUTPUT"
echo "Model:  $MODEL"
echo "Max:    $MAX_RUBRICS rubrics"
echo ""

/data/anaconda3/bin/python -m trainable_openclaw.evaluation.rubric_engine \
    --input $INPUT_FILES \
    --output "$OUTPUT" \
    --api-key "$API_KEY" \
    --model "$MODEL" \
    --max-rubrics "$MAX_RUBRICS" \
    --verbose

echo ""
echo "Done. Rubrics saved to: $OUTPUT"
echo "Review with: cat $OUTPUT | python -m json.tool"
