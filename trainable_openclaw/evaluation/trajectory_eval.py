"""
S3: 轨迹评估与数据导出 (Phase 1.5)

处理 S2 产出的纠错轨迹，完成：
1. 分级 — direct_pass / corrected / partial / failed
2. 提取训练对 — (错误回答, 纠错意见, 修正后回答) 三元组
3. 聚合 Rubric 种子 — 从纠错维度中提取高频模式
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def grade_trajectory(traj: dict) -> str:
    """对一条纠错轨迹进行分级。

    返回:
        "直接通过" — 0 轮纠错，User Sim 直接满意
        "纠错后通过" — 经过纠错后最终满意
        "部分通过" — 经过纠错但部分改善，未完全满意
        "失败" — 出错或无法纠正
    """
    verdict = traj.get("最终判定", "")
    error = traj.get("错误信息", "")
    if error:
        return "失败"
    return verdict or "未知"


def extract_training_pairs(traj: dict) -> list[dict]:
    """从一条轨迹中提取 (错误回答, 纠错意见, 修正后回答) 训练对。

    规则：每轮 assistant 回答如果是被纠正过的（下一轮有 user 纠错），
    则形成 (当前回答, 下一轮纠错意见, 再下一轮回答) 三元组。
    """
    messages = traj.get("对话消息", [])
    if len(messages) < 3:
        return []

    pairs = []
    # 遍历消息找 pattern: assistant → user(correction) → assistant
    for i in range(len(messages) - 2):
        m1 = messages[i]
        m2 = messages[i + 1]
        m3 = messages[i + 2]
        if (m1["role"] == "assistant" and
                m2["role"] == "user" and
                m3["role"] == "assistant"):
            # m2 content 包含纠错意见
            correction_text = m2.get("content", "")
            if "我发现一个问题" in correction_text:
                # 提取纯纠错意见（去掉包装文字）
                correction_text = correction_text.replace(
                    "关于你之前的回答，我发现一个问题：\n", ""
                ).replace("\n请修改你的回答，解决这个问题。", "")

            pairs.append({
                "种子提示词": traj.get("种子提示词", ""),
                "类别": traj.get("类别", ""),
                "错误回答": m1["content"],
                "错误思考": m1.get("reasoning", ""),
                "纠错意见": correction_text,
                "修正回答": m3["content"],
                "修正思考": m3.get("reasoning", ""),
            })

    return pairs


def extract_direct_pass_examples(traj: dict) -> dict | None:
    """提取直接通过的示例（作为正例训练数据）。"""
    if traj.get("最终判定") != "直接通过":
        return None
    messages = traj.get("对话消息", [])
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    if not assistant_msgs:
        return None
    m = assistant_msgs[0]
    return {
        "种子提示词": traj.get("种子提示词", ""),
        "类别": traj.get("类别", ""),
        "回答": m["content"],
        "思考": m.get("reasoning", ""),
    }


def extract_rubric_seeds(trajectories: list[dict]) -> list[dict]:
    """从一批轨迹的纠错轮次中聚合 Rubric 种子。

    通过分析对话消息中的 user 纠错消息，统计纠错涉及的高频维度。
    """
    dimension_examples: dict[str, list[dict]] = defaultdict(list)

    for traj in trajectories:
        if traj.get("最终判定") not in ("直接通过", "纠错后通过"):
            continue
        messages = traj.get("对话消息", [])
        for i, m in enumerate(messages):
            if m["role"] != "user":
                continue
            content = m.get("content", "")
            if "我发现一个问题" not in content:
                continue

            # 找到对应被纠错的 assistant 回答
            bad_answer = ""
            if i > 0 and messages[i - 1]["role"] == "assistant":
                bad_answer = messages[i - 1]["content"][:300]

            # 提取纠错意见
            correction = content.replace(
                "关于你之前的回答，我发现一个问题：\n", ""
            ).replace("\n请修改你的回答，解决这个问题。", "")

            # 用简单的关键词提取来推断维度（后面 B1 会用 LLM 精炼）
            dims = _infer_dimensions(correction)

            for dim in dims:
                dimension_examples[dim].append({
                    "种子提示词": traj.get("种子提示词", "")[:150],
                    "类别": traj.get("类别", ""),
                    "错误片段": bad_answer,
                    "纠错意见": correction[:300],
                })

    # 按频次排序，输出维度种子
    seeds = []
    for dim, examples in sorted(dimension_examples.items(),
                                 key=lambda x: len(x[1]), reverse=True):
        seeds.append({
            "维度": dim,
            "频次": len(examples),
            "示例": examples[:5],  # 最多保留 5 个示例
        })

    return seeds


def _infer_dimensions(correction: str) -> list[str]:
    """从纠错文本中粗略推断涉及的维度（关键词匹配）。

    B1 反馈分析模块会用 LLM 精炼这些维度，这里只做初步标签。
    """
    dims = []
    keywords = {
        "代码正确性": ["逻辑错误", "bug", "边界", "异常", "空值", "索引"],
        "命名规范": ["变量名", "命名", "函数名", "PEP", "表意"],
        "类型注解": ["类型", "typing", "注解", "返回类型"],
        "错误处理": ["异常", "try", "catch", "错误处理", "容错"],
        "性能": ["复杂度", "性能", "效率", "O(n)", "优化"],
        "文笔流畅": ["流畅", "通顺", "节奏", "句子"],
        "意境表达": ["意境", "画面感", "用词", "优美", "表达"],
        "结构逻辑": ["结构", "段落", "组织", "论证", "逻辑"],
        "计算准确性": ["计算", "数值", "结果", "错误", "准确"],
        "步骤完整性": ["步骤", "跳步", "省略", "完整", "推导"],
        "表达严谨性": ["严谨", "定义", "定理", "引用", "假设"],
        "信息完整性": ["遗漏", "缺失", "完整", "不全面"],
        "清晰易懂": ["清晰", "理解", "初学者", "易懂"],
        "事实准确性": ["事实", "编造", "错误", "不准确"],
        "安全性": ["安全", "注入", "权限", "SQL", "XSS"],
        "可维护性": ["维护", "SOLID", "可读", "重构"],
    }
    for dim, kws in keywords.items():
        if any(kw in correction for kw in kws):
            dims.append(dim)
    if not dims:
        dims.append("其他")
    return dims


def process_trajectories(
    input_file: str | Path,
    output_dir: str | Path = "data",
) -> dict:
    """完整的 S3 处理流水线。

    Args:
        input_file: S2 产出的 trajectories JSONL 文件
        output_dir: 输出目录

    Returns:
        统计信息 dict
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载轨迹
    trajectories = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))

    logger.info(f"加载 {len(trajectories)} 条轨迹")

    # 1. 分级统计
    grades = Counter()
    for t in trajectories:
        grades[grade_trajectory(t)] += 1

    # 2. 提取训练对
    all_pairs = []
    direct_passes = []
    for t in trajectories:
        pairs = extract_training_pairs(t)
        all_pairs.extend(pairs)
        dp = extract_direct_pass_examples(t)
        if dp:
            direct_passes.append(dp)

    # 写入训练对
    pairs_file = output_dir / "training_pairs.jsonl"
    with open(pairs_file, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 写入正例
    positive_file = output_dir / "positive_examples.jsonl"
    with open(positive_file, "w", encoding="utf-8") as f:
        for p in direct_passes:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 3. 提取 Rubric 种子
    rubric_seeds = extract_rubric_seeds(trajectories)
    seeds_file = output_dir / "rubric_seeds.json"
    with open(seeds_file, "w", encoding="utf-8") as f:
        json.dump(rubric_seeds, f, ensure_ascii=False, indent=2)

    stats = {
        "总轨迹数": len(trajectories),
        "分级分布": dict(grades),
        "训练对数": len(all_pairs),
        "正例数": len(direct_passes),
        "Rubric种子维度": len(rubric_seeds),
        "输出文件": {
            "训练对": str(pairs_file),
            "正例": str(positive_file),
            "Rubric种子": str(seeds_file),
        },
    }

    logger.info(f"S3 处理完成: {stats}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse
    parser = argparse.ArgumentParser(description="S3: 轨迹评估与数据导出")
    parser.add_argument("input", help="S2 轨迹 JSONL 文件")
    parser.add_argument("--output-dir", default="data", help="输出目录")
    parser.add_argument("--stats-only", action="store_true", help="仅打印统计")
    args = parser.parse_args()

    if args.stats_only:
        trajectories = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trajectories.append(json.loads(line))
        grades = Counter()
        for t in trajectories:
            grades[grade_trajectory(t)] += 1
        print(f"总轨迹数: {len(trajectories)}")
        print(f"分级分布: {dict(grades)}")
        pairs_count = sum(len(extract_training_pairs(t)) for t in trajectories)
        print(f"训练对数: {pairs_count}")
        seeds = extract_rubric_seeds(trajectories)
        print(f"Rubric种子维度: {len(seeds)}")
        for s in seeds[:10]:
            print(f"  {s['维度']}: {s['频次']}次")
    else:
        stats = process_trajectories(args.input, args.output_dir)
        print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
