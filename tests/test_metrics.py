"""Tests for trainable_openclaw.evaluation.metrics."""

import pytest
import tempfile
import json
import os

from trainable_openclaw.evaluation.metrics import (
    JudgeQualityMetrics,
    ModelImprovementMetrics,
    RubricQualityMetrics,
    ConvergenceMetrics,
    spearman_correlation,
    accuracy_at_k,
    compute_judge_quality,
    compute_improvement,
    compute_rubric_quality,
    compute_convergence,
    compute_coverage,
    rubrics_variance_test,
)


class TestSpearmanCorrelation:
    def test_perfect_positive(self):
        assert spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert spearman_correlation([4, 3, 2, 1], [10, 20, 30, 40]) == pytest.approx(-1.0)

    def test_uncorrelated(self):
        rho = spearman_correlation([1, 2, 3, 4], [10, 30, 20, 40])
        assert -1.0 <= rho <= 1.0

    def test_single_element(self):
        assert spearman_correlation([1], [2]) == 0.0

    def test_empty(self):
        assert spearman_correlation([], []) == 0.0

    def test_constant_value(self):
        rho = spearman_correlation([1, 1, 1], [1, 2, 3])
        assert rho == 0.0


class TestAccuracyAtK:
    def test_all_correct_k1(self):
        rankings = [[0.9, 0.5, 0.3], [0.2, 0.9, 0.1]]
        gt = [0, 1]
        assert accuracy_at_k(rankings, gt, k=1) == 1.0

    def test_half_correct(self):
        rankings = [[0.5, 0.9, 0.3], [0.9, 0.5, 0.3]]
        gt = [1, 0]
        assert accuracy_at_k(rankings, gt, k=1) == 1.0

    def test_k2(self):
        rankings = [[0.9, 0.8, 0.1]]
        gt = [2]
        assert accuracy_at_k(rankings, gt, k=2) == 0.0

    def test_empty(self):
        assert accuracy_at_k([], []) == 0.0


class TestComputeJudgeQuality:
    def test_basic(self):
        results = [
            {"平均分": 0.5, "分数向量": [0.4, 0.6]},
            {"平均分": 0.7, "分数向量": [0.6, 0.8]},
            {"平均分": 0.3, "分数向量": [0.2, 0.4]},
        ]
        q = compute_judge_quality(results)
        assert q.n_samples == 3
        assert q.区分度 == pytest.approx(0.4)
        assert 0.45 < q.平均分 < 0.55

    def test_empty(self):
        q = compute_judge_quality([])
        assert q.n_samples == 0
        assert q.平均分 == 0.0

    def test_response_rate(self):
        results = [
            {"平均分": 0.5},
            {"平均分": 0.0},
            {"平均分": 0.7},
        ]
        q = compute_judge_quality(results)
        assert q.响应率 == pytest.approx(2 / 3)

    def test_with_ground_truth(self):
        results = [
            {"平均分": 0.9}, {"平均分": 0.5},
            {"平均分": 0.3}, {"平均分": 0.8},
        ]
        gt = [{"best_answer_index": 0}, {"best_answer_index": 1}]
        q = compute_judge_quality(results, gt)
        # 2 prompts, 2 answers each
        assert q.Spearman_rho is not None


class TestComputeImprovement:
    def test_deltas(self):
        pre = [{"平均分": 0.3}, {"平均分": 0.4}]
        post = [{"平均分": 0.5}, {"平均分": 0.6}]
        m = compute_improvement(pre, post)
        assert m.pre_平均分 == pytest.approx(0.35)
        assert m.post_平均分 == pytest.approx(0.55)
        assert m.Δ平均分 == pytest.approx(0.2)

    def test_positive_rate(self):
        pre = [{"平均分": 0.3}, {"平均分": 0.6}]
        post = [{"平均分": 0.7}, {"平均分": 0.8}]
        m = compute_improvement(pre, post)
        assert m.pre_正向率 == pytest.approx(0.5)
        assert m.post_正向率 == pytest.approx(1.0)
        assert m.Δ正向率 == pytest.approx(0.5)

    def test_correction_rate_delta(self):
        m = compute_improvement(
            pre_correction={"纠错率": 0.4},
            post_correction={"纠错率": 0.3},
        )
        assert m.pre_纠错率 == 0.4
        assert m.post_纠错率 == 0.3
        assert m.Δ纠错率 == pytest.approx(-0.1)

    def test_none_inputs(self):
        m = compute_improvement()
        assert m.Δ平均分 == 0.0
        assert m.Δ纠错率 == 0.0


class TestComputeCoverage:
    def test_full_coverage(self):
        texts = ["变量命名不规范", "缺少类型注解"]
        names = ["变量命名检查", "类型完整性"]
        cov, uncovered = compute_coverage(texts, names)
        assert cov == 1.0
        assert uncovered == []

    def test_partial_coverage(self):
        texts = ["代码性能差", "文档缺失"]
        names = ["性能优化", "格式规范"]
        cov, uncovered = compute_coverage(texts, ["性能优化", "格式规范"])
        assert cov == 0.5

    def test_empty(self):
        cov, uncovered = compute_coverage([], ["a", "b"])
        assert cov == 1.0


class TestRubricsVarianceTest:
    def test_good_variance(self):
        results = [
            {"分数向量": [0.8, 0.1]},
            {"分数向量": [0.2, 0.9]},
            {"分数向量": [0.5, 0.5]},
        ]
        passed, stds = rubrics_variance_test(results, threshold=0.1)
        assert passed

    def test_poor_variance(self):
        results = [
            {"分数向量": [0.5, 0.5]},
            {"分数向量": [0.5, 0.5]},
        ]
        passed, stds = rubrics_variance_test(results, threshold=0.1)
        assert not passed

    def test_no_score_vectors(self):
        results = [{"平均分": 0.5}, {"平均分": 0.6}]
        passed, stds = rubrics_variance_test(results)
        assert "overall" in stds


class TestComputeRubricQuality:
    def test_active_rate(self):
        rubrics = [
            {"名称": "r1", "评分提示词": "abc", "状态": "活跃", "命中次数": 5},
            {"名称": "r2", "评分提示词": "def", "状态": "已归档", "命中次数": 1},
        ]
        q = compute_rubric_quality(rubrics)
        assert q.活跃率 == 0.5
        assert q.n_total == 2

    def test_coverage(self):
        rubrics = [
            {"名称": "代码规范", "评分提示词": "check code", "状态": "活跃", "命中次数": 3},
        ]
        feedback = [{"text": "我的代码变量命名不好"}]
        q = compute_rubric_quality(rubrics, feedback_samples=feedback)
        assert q.n_total == 1

    def test_empty(self):
        q = compute_rubric_quality([])
        assert q.n_total == 0
        assert q.活跃率 == 0.0
        assert q.覆盖率 == 0.0


class TestComputeConvergence:
    def test_improving(self):
        history = [
            {"Δ纠错率": -0.02},
            {"Δ纠错率": -0.05},
            {"Δ纠错率": -0.08},
        ]
        c = compute_convergence(history, target_delta=-0.05)
        assert c.达到目标的轮次 == 2
        assert c.收敛方向 == "持续改善"

    def test_fluctuating(self):
        history = [
            {"Δ纠错率": -0.02},
            {"Δ纠错率": -0.01},
            {"Δ纠错率": -0.04},
        ]
        c = compute_convergence(history, target_delta=-0.05)
        assert c.收敛方向 in ("波动", "改善中", "退化中", "持平")

    def test_single_round(self):
        history = [{"Δ纠错率": -0.03}]
        c = compute_convergence(history)
        assert c.total_rounds == 1
        assert c.收敛方向 == "数据不足"

    def test_empty(self):
        c = compute_convergence([])
        assert c.total_rounds == 0


class TestDataclassToDict:
    def test_judge_quality(self):
        q = JudgeQualityMetrics(平均分=0.5, 区分度=0.3, n_samples=10)
        d = q.to_dict()
        assert d["平均分"] == 0.5
        assert d["区分度"] == 0.3
        assert d["样本数"] == 10

    def test_model_improvement(self):
        m = ModelImprovementMetrics(Δ纠错率=-0.1, rounds=3)
        d = m.to_dict()
        assert d["Δ纠错率"] == -0.1
        assert d["轮次"] == 3

    def test_convergence(self):
        c = ConvergenceMetrics(收敛方向="持续改善", total_rounds=5)
        d = c.to_dict()
        assert d["收敛方向"] == "持续改善"
        assert d["总轮次"] == 5
