"""Tests for trainable_openclaw.evaluation.correction_rate — no GPU required."""

import json
import os
import tempfile

import pytest


@pytest.fixture
def temp_test_pairs():
    """Create temporary test training_pairs.jsonl."""
    pairs = [
        {
            "种子提示词": "写一个Python排序函数",
            "类别": "coding",
            "错误回答": "def sort(a): return sorted(a)",
            "纠错意见": "变量名a不够表意",
            "修正回答": "def sort_array(arr): return sorted(arr)",
        },
        {
            "种子提示词": "什么是机器学习？",
            "类别": "explanation",
            "错误回答": "机器学习就是让机器学习的学科",
            "纠错意见": "定义不够完整准确",
            "修正回答": "机器学习是人工智能的一个分支...(完整解释)",
        },
        {
            "种子提示词": "计算 15*3+7",
            "类别": "math",
            "错误回答": "15*3=45, 45+7=52",
            "纠错意见": "",
            "修正回答": "",
        },
    ]
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


class TestLoadTestPrompts:
    """Tests for load_test_prompts helper."""

    def test_loads_unique_prompts(self, temp_test_pairs):
        from trainable_openclaw.evaluation.correction_rate import load_test_prompts

        prompts = load_test_prompts(temp_test_pairs)
        assert len(prompts) == 3
        assert prompts[0]["prompt"] == "写一个Python排序函数"
        assert prompts[0]["category"] == "coding"

    def test_max_prompts_cap(self, temp_test_pairs):
        from trainable_openclaw.evaluation.correction_rate import load_test_prompts

        prompts = load_test_prompts(temp_test_pairs, max_prompts=2)
        assert len(prompts) == 2

    def test_missing_file(self):
        from trainable_openclaw.evaluation.correction_rate import load_test_prompts

        prompts = load_test_prompts("nonexistent.jsonl")
        assert prompts == []

    def test_deduplicates_seeds(self, temp_test_pairs):
        # Add a duplicate prompt
        duplicate = {
            "种子提示词": "写一个Python排序函数",
            "类别": "coding",
            "错误回答": "another bad",
            "纠错意见": "another correction",
            "修正回答": "another good",
        }
        with open(temp_test_pairs, "a", encoding="utf-8") as f:
            f.write(json.dumps(duplicate, ensure_ascii=False) + "\n")

        from trainable_openclaw.evaluation.correction_rate import load_test_prompts

        prompts = load_test_prompts(temp_test_pairs)
        # Still 3 unique (the duplicate is merged)
        assert len(prompts) == 3


class TestPromptEvalResult:
    """Tests for PromptEvalResult dataclass."""

    def test_defaults(self):
        from trainable_openclaw.evaluation.correction_rate import PromptEvalResult

        r = PromptEvalResult(prompt="test")
        assert r.prompt == "test"
        assert r.outcome == ""
        assert r.correction_rounds == 0

    def test_fields(self):
        from trainable_openclaw.evaluation.correction_rate import PromptEvalResult

        r = PromptEvalResult(
            prompt="test prompt",
            category="math",
            outcome="直接通过",
            correction_rounds=0,
        )
        assert r.outcome == "直接通过"
        assert r.category == "math"


class TestCorrectionRateResult:
    """Tests for CorrectionRateResult dataclass."""

    def test_to_dict(self):
        from trainable_openclaw.evaluation.correction_rate import CorrectionRateResult

        r = CorrectionRateResult(
            total=10,
            直接通过=7,
            纠错后通过=2,
            失败=1,
            纠错率=0.3,
            avg_rounds=1.5,
            elapsed_seconds=60.0,
            Δ纠错率=-0.05,
            pre_纠错率=0.35,
        )
        d = r.to_dict()
        assert d["总数"] == 10
        assert d["直接通过"] == 7
        assert d["纠错率"] == 0.3
        assert d["平均纠正轮次"] == 1.5


class TestCorrectionRateEvaluator:
    """Tests for CorrectionRateEvaluator construction and properties."""

    def test_construction(self):
        from trainable_openclaw.evaluation.correction_rate import CorrectionRateEvaluator

        ev = CorrectionRateEvaluator(
            model_server_url="http://localhost:8000/v1",
            api_key="sk-test",
            max_concurrent=3,
        )
        assert ev.model_server_url == "http://localhost:8000/v1"
        assert ev.api_key == "sk-test"
        assert ev.max_concurrent == 3

    def test_env_api_key(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-key")
        from trainable_openclaw.evaluation.correction_rate import CorrectionRateEvaluator

        ev = CorrectionRateEvaluator()
        assert ev.api_key == "sk-env-key"
