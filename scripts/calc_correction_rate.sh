#!/bin/bash
# 计算纠错率 (Correction Rate)
# 用法:
#   bash scripts/calc_correction_rate.sh                    # 测试集
#   bash scripts/calc_correction_rate.sh train              # 训练集
#   bash scripts/calc_correction_rate.sh all                # 两者

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

case "${1:-test}" in
    test)
        $PYTHON scripts/calc_correction_rate.py data/test_trajectories.jsonl
        ;;
    train)
        $PYTHON scripts/calc_correction_rate.py data/train_trajectories.jsonl
        ;;
    all)
        $PYTHON scripts/calc_correction_rate.py --all
        ;;
    *)
        echo "用法: $0 {test|train|all}"
        exit 1
        ;;
esac
