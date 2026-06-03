from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


# --- Data structures ---

@dataclass
class JudgeQualityMetrics:
    平均分: float = 0.0
    分数标准差: float = 0.0
    区分度: float = 0.0
    批内一致性: float = 0.0
    响应率: float = 1.0
    Spearman_rho: float | None = None
    n_samples: int = 0
    n_rubrics: int = 0

    def to_dict(self) -> dict:
        return {
            "平均分": self.平均分,
            "分数标准差": self.分数标准差,
            "区分度": self.区分度,
            "批内一致性": self.批内一致性,
            "响应率": self.响应率,
            "Spearman_rho": self.Spearman_rho,
            "样本数": self.n_samples,
            "rubric数": self.n_rubrics,
        }


@dataclass
class ModelImprovementMetrics:
    pre_平均分: float = 0.0
    post_平均分: float = 0.0
    Δ平均分: float = 0.0
    pre_正向率: float = 0.0
    post_正向率: float = 0.0
    Δ正向率: float = 0.0
    pre_纠错率: float = 0.0
    post_纠错率: float = 0.0
    Δ纠错率: float = 0.0
    rounds: int = 0

    def to_dict(self) -> dict:
        return {
            "pre_平均分": self.pre_平均分,
            "post_平均分": self.post_平均分,
            "Δ平均分": self.Δ平均分,
            "pre_正向率": self.pre_正向率,
            "post_正向率": self.post_正向率,
            "Δ正向率": self.Δ正向率,
            "pre_纠错率": self.pre_纠错率,
            "post_纠错率": self.post_纠错率,
            "Δ纠错率": self.Δ纠错率,
            "轮次": self.rounds,
        }


@dataclass
class RubricQualityMetrics:
    覆盖率: float = 0.0
    区分度: float = 0.0
    冗余度: float = 0.0
    活跃率: float = 1.0
    命中集中度: dict = field(default_factory=dict)
    未覆盖反馈数: int = 0
    n_total: int = 0

    def to_dict(self) -> dict:
        d = {
            "覆盖率": self.覆盖率,
            "区分度": self.区分度,
            "冗余度": self.冗余度,
            "活跃率": self.活跃率,
            "未覆盖反馈数": self.未覆盖反馈数,
            "总数": self.n_total,
        }
        if self.命中集中度:
            d["命中集中度"] = self.命中集中度
        return d


@dataclass
class ConvergenceMetrics:
    达到目标的轮次: int = 0
    累积Δ纠错率: list[float] = field(default_factory=list)
    每轮提升: list[float] = field(default_factory=list)
    收敛方向: str = ""
    total_rounds: int = 0

    def to_dict(self) -> dict:
        return {
            "达到目标的轮次": self.达到目标的轮次,
            "累计Δ纠错率": self.累积Δ纠错率,
            "每轮提升": self.每轮提升,
            "收敛方向": self.收敛方向,
            "总轮次": self.total_rounds,
        }


# --- Statistical helpers ---

def _rankdata(values: list[float]) -> list[float]:
    """Pure-Python rankdata (average ranks for ties)."""
    indexed = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[indexed[k][1]] = avg_rank
        i = j
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    std_x = (sum((v - mean_x) ** 2 for v in x) ** 0.5)
    std_y = (sum((v - mean_y) ** 2 for v in y) ** 0.5)
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def spearman_correlation(predicted: list[float], actual: list[float]) -> float:
    """Compute Spearman rank correlation (pure Python, no scipy needed)."""
    if len(predicted) < 2:
        return 0.0
    return _pearson(_rankdata(predicted), _rankdata(actual))


def accuracy_at_k(rankings: list[list[float]], ground_truth: list[int], k: int = 1) -> float:
    """Accuracy@K: fraction where top-K predicted includes ground-truth best.

    Args:
        rankings: Per-item score vectors. Higher = better.
        ground_truth: Index of the best item for each group.
        k: Top-K threshold.
    """
    if not rankings or not ground_truth:
        return 0.0
    correct = 0
    for scores, gt_idx in zip(rankings, ground_truth):
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        if gt_idx in top_k:
            correct += 1
    return correct / len(rankings)


# --- Core metric functions ---

def compute_judge_quality(
    score_results: list[dict],
    ground_truth: list[dict] | None = None,
) -> JudgeQualityMetrics:
    """Compute judge reliability metrics from scoring results.

    Args:
        score_results: Output from JudgeExecutor.score_answers_sync().
            Each dict has '平均分', optionally '分数向量' or '评分' keys.
        ground_truth: Optional list of {prompt, best_answer_index} for Spearman/Accuracy.

    Returns:
        JudgeQualityMetrics.
    """
    metrics = JudgeQualityMetrics()
    if not score_results:
        return metrics

    means = [r.get("平均分", 0) for r in score_results]
    n = len(means)
    metrics.n_samples = n
    metrics.平均分 = sum(means) / n if n else 0.0
    metrics.分数标准差 = (sum((m - metrics.平均分) ** 2 for m in means) / n) ** 0.5 if n > 1 else 0.0
    metrics.区分度 = max(means) - min(means) if n else 0.0

    # Response rate: count non-zero scores
    nonzero = sum(1 for m in means if m > 0)
    metrics.响应率 = nonzero / n if n else 1.0

    # Intra-batch consistency: pairwise correlation between rubric dims
    score_vectors = [r.get("分数向量", []) for r in score_results if r.get("分数向量")]
    if len(score_vectors) > 1 and score_vectors[0]:
        metrics.n_rubrics = len(score_vectors[0])
        if metrics.n_rubrics > 1:
            dim_scores = list(zip(*score_vectors))
            cors = []
            for a in range(metrics.n_rubrics):
                for b in range(a + 1, metrics.n_rubrics):
                    cors.append(spearman_correlation(list(dim_scores[a]), list(dim_scores[b])))
            metrics.批内一致性 = sum(cors) / len(cors) if cors else 0.0

    # Ground truth metrics
    if ground_truth:
        gt_best_indices = [g.get("best_answer_index", 0) for g in ground_truth]
        # Group score results by prompt (assume they come in groups)
        group_scores = []
        group_size = len(score_results) // len(ground_truth) if ground_truth else 0
        if group_size > 0:
            for g_idx in range(len(ground_truth)):
                start = g_idx * group_size
                group = means[start : start + group_size]
                group_scores.append(group)
            metrics.Spearman_rho = spearman_correlation(
                [max(g) if g else 0 for g in group_scores],
                [float(gt_best_indices[i]) for i in range(len(group_scores))],
            )

    return metrics


def compute_improvement(
    pre_scores: list[dict] | None = None,
    post_scores: list[dict] | None = None,
    pre_correction: dict | None = None,
    post_correction: dict | None = None,
) -> ModelImprovementMetrics:
    """Compute model improvement between pre and post training evaluations.

    Args:
        pre_scores: Judge score results before training.
        post_scores: Judge score results after training.
        pre_correction: CorrectionRateResult.to_dict() before training.
        post_correction: CorrectionRateResult.to_dict() after training.

    Returns:
        ModelImprovementMetrics with deltas.
    """
    m = ModelImprovementMetrics()

    if pre_scores:
        pre_means = [r.get("平均分", 0) for r in pre_scores]
        m.pre_平均分 = sum(pre_means) / len(pre_means) if pre_means else 0.0
        m.pre_正向率 = sum(1 for v in pre_means if v > 0.5) / len(pre_means) if pre_means else 0.0

    if post_scores:
        post_means = [r.get("平均分", 0) for r in post_scores]
        m.post_平均分 = sum(post_means) / len(post_means) if post_means else 0.0
        m.post_正向率 = sum(1 for v in post_means if v > 0.5) / len(post_means) if post_means else 0.0

    m.Δ平均分 = m.post_平均分 - m.pre_平均分
    m.Δ正向率 = m.post_正向率 - m.pre_正向率

    if pre_correction:
        m.pre_纠错率 = pre_correction.get("纠错率", 0)
    if post_correction:
        m.post_纠错率 = post_correction.get("纠错率", 0)
    m.Δ纠错率 = m.post_纠错率 - m.pre_纠错率

    return m


def _extract_bigrams(text: str) -> set[str]:
    """Extract character bigrams from Chinese text for fuzzy matching."""
    chars = [c for c in text if c.strip()]
    if len(chars) < 2:
        return {text}
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def compute_coverage(
    feedback_texts: list[str],
    rubric_names: list[str],
    rubric_patterns: list[str] | None = None,
) -> tuple[float, list[str]]:
    """Simple coverage: fraction of feedback items matching at least one rubric.

    Uses character bigram overlap for Chinese text — a feedback is "covered"
    if it shares at least one bigram with any rubric name or source pattern.

    Returns (coverage_rate, uncovered_feedback).
    """
    if not feedback_texts:
        return 1.0, []

    patterns = rubric_patterns or []
    all_bigrams = set()
    for name in rubric_names:
        all_bigrams.update(_extract_bigrams(name))
    for pat in patterns:
        all_bigrams.update(_extract_bigrams(pat))

    uncovered = []
    for fb in feedback_texts:
        fb_bigrams = _extract_bigrams(fb)
        if not fb_bigrams & all_bigrams:
            uncovered.append(fb)

    coverage = 1.0 - (len(uncovered) / len(feedback_texts))
    return coverage, uncovered


def rubrics_variance_test(
    score_results: list[dict],
    threshold: float = 0.1,
) -> tuple[bool, dict[str, float]]:
    """Test if rubrics produce sufficient score variance across answers.

    A rubric dimension with low variance has poor discrimination.

    Returns (passed, {rubric_dim_name: std_dev}).
    """
    if not score_results:
        return False, {}

    score_vectors = [r.get("分数向量", []) for r in score_results if r.get("分数向量")]
    if not score_vectors:
        means = [r.get("平均分", 0) for r in score_results]
        n = len(means)
        std = (sum((m - sum(means) / n) ** 2 for m in means) / n) ** 0.5 if n > 1 else 0.0
        passed = std >= threshold
        return passed, {"overall": std}

    n_dims = len(score_vectors[0])
    dim_scores = list(zip(*score_vectors))
    dim_stds = {}
    for d in range(n_dims):
        vals = list(dim_scores[d])
        n = len(vals)
        mean_v = sum(vals) / n if n else 0.0
        std_v = (sum((v - mean_v) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
        dim_stds[f"dim_{d}"] = std_v

    passed = all(s >= threshold for s in dim_stds.values())
    return passed, dim_stds


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Character-level Jaccard similarity for Chinese text."""
    set_a = set(text_a)
    set_b = set(text_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_rubric_quality(
    rubrics: list,
    feedback_samples: list[dict] | None = None,
    score_results: list[dict] | None = None,
) -> RubricQualityMetrics:
    """Compute rubric set quality metrics.

    Args:
        rubrics: List of rubric dicts (from RubricStore) or Rubric objects.
        feedback_samples: Optional list of {text, ...} for coverage test.
        score_results: Optional list of score dicts for variance test.

    Returns:
        RubricQualityMetrics.
    """
    metrics = RubricQualityMetrics()

    if not rubrics:
        metrics.活跃率 = 0.0
        return metrics

    metrics.n_total = len(rubrics)

    # Active rate
    active = 0
    rubric_names = []
    rubric_patterns = []
    hit_counts = {}
    for r in rubrics:
        if hasattr(r, "状态"):
            if r.状态 == "活跃":
                active += 1
            rubric_names.append(getattr(r, "名称", ""))
            rubric_patterns.append(getattr(r, "评分提示词", ""))
            hit_counts[getattr(r, "名称", r.id if hasattr(r, "id") else "?")] = getattr(r, "命中次数", 0)
        elif isinstance(r, dict):
            if r.get("状态") == "活跃":
                active += 1
            rubric_names.append(r.get("名称", ""))
            rubric_patterns.append(r.get("评分提示词", ""))
            hit_counts[r.get("名称", r.get("id", "?"))] = r.get("命中次数", 0)

    metrics.活跃率 = active / len(rubrics) if rubrics else 1.0
    metrics.命中集中度 = hit_counts

    # Coverage
    if feedback_samples:
        texts = [fb.get("text", fb.get("纠错意见", "")) for fb in feedback_samples]
        texts = [t for t in texts if t]
        cov, uncovered = compute_coverage(texts, rubric_names, rubric_patterns)
        metrics.覆盖率 = cov
        metrics.未覆盖反馈数 = len(uncovered)

    # Discrimination via variance test
    if score_results:
        passed, stds = rubrics_variance_test(score_results)
        metrics.区分度 = sum(stds.values()) / len(stds) if stds else 0.0

    # Redundancy: pairwise Jaccard similarity of rubric prompts
    n = len(rubric_patterns)
    if n > 1:
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(_jaccard_similarity(rubric_patterns[i], rubric_patterns[j]))
        metrics.冗余度 = sum(sims) / len(sims) if sims else 0.0

    return metrics


def compute_convergence(
    round_history: list[dict],
    target_delta: float = -0.05,
) -> ConvergenceMetrics:
    """Analyze convergence from round history.

    Args:
        round_history: List of per-round results with 'Δ纠错率' or 'delta' keys.
        target_delta: Target improvement (negative = fewer corrections = better).

    Returns:
        ConvergenceMetrics with rounds-to-target, per-round gains, trend.
    """
    m = ConvergenceMetrics()
    if not round_history:
        return m

    m.total_rounds = len(round_history)

    deltas = []
    for r in round_history:
        d = r.get("Δ纠错率", r.get("delta", 0))
        if isinstance(d, (int, float)):
            deltas.append(d)

    m.累积Δ纠错率 = []
    running = 0.0
    for d in deltas:
        running += d
        m.累积Δ纠错率.append(round(running, 4))

    per_round = []
    for i in range(len(deltas)):
        if i == 0:
            per_round.append(deltas[i])
        else:
            per_round.append(deltas[i] - deltas[i - 1])
    m.每轮提升 = per_round

    # Find first round reaching target
    m.达到目标的轮次 = 0
    for i, d in enumerate(m.累积Δ纠错率):
        if d <= target_delta:
            m.达到目标的轮次 = i + 1
            break

    # Trend direction
    if len(deltas) >= 3:
        improving = sum(1 for i in range(1, len(deltas)) if deltas[i] < deltas[i - 1])
        worsening = sum(1 for i in range(1, len(deltas)) if deltas[i] > deltas[i - 1])
        if improving > worsening and improving >= len(deltas) // 2:
            m.收敛方向 = "持续改善"
        elif worsening > improving and worsening >= len(deltas) // 2:
            m.收敛方向 = "退化"
        else:
            m.收敛方向 = "波动"
    elif len(deltas) == 2:
        if deltas[1] < deltas[0]:
            m.收敛方向 = "改善中"
        elif deltas[1] > deltas[0]:
            m.收敛方向 = "退化中"
        else:
            m.收敛方向 = "持平"
    else:
        m.收敛方向 = "数据不足"

    return m
