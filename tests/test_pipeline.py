"""Tests for trainable_openclaw.pipeline."""

import pytest
import tempfile
import json
import os

from trainable_openclaw.pipeline import (
    PipelineConfig,
    RoundResult,
    Pipeline,
    load_training_data,
    load_test_prompts,
)


class TestPipelineConfig:
    def test_defaults(self):
        c = PipelineConfig()
        assert c.train_data_path == "data/phase3_datasets/train_prompts.jsonl"
        assert c.rubrics_path == "data/rubrics_category.json"
        assert c.reward_mode == "mean"
        assert c.max_rubrics == 8
        assert c.run_pre_eval is True
        assert c.run_post_eval is True
        assert c.run_rubric_evolution is False

    def test_custom(self):
        c = PipelineConfig(
            train_data_path="/tmp/train.jsonl",
            rubrics_path="/tmp/rubrics.json",
            reward_mode="total",
            max_rubrics=4,
        )
        assert c.train_data_path == "/tmp/train.jsonl"
        assert c.rubrics_path == "/tmp/rubrics.json"
        assert c.reward_mode == "total"
        assert c.max_rubrics == 4

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        c = PipelineConfig()
        assert c.api_key == "test-key"

    def test_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
        c = PipelineConfig(api_key="explicit-key")
        assert c.api_key == "explicit-key"


class TestRoundResult:
    def test_defaults(self):
        r = RoundResult()
        assert r.round_number == 1
        assert r.pre_纠错率 is None
        assert r.Δ纠错率 is None
        assert r.training_completed is False
        assert r.errors == []

    def test_to_dict(self):
        r = RoundResult(
            round_number=2,
            pre_纠错率=0.4,
            post_纠错率=0.3,
            Δ纠错率=-0.1,
            elapsed_seconds=120.5,
        )
        d = r.to_dict()
        assert d["轮次"] == 2
        assert d["pre_纠错率"] == 0.4
        assert d["Δ纠错率"] == -0.1
        assert d["耗时秒"] == 120.5

    def test_with_errors(self):
        r = RoundResult(errors=["Connection failed", "Timeout"])
        d = r.to_dict()
        assert len(d["错误"]) == 2


class TestLoadTrainingData:
    def test_load(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"种子提示词": "hello", "类别": "coding"}) + "\n")
            f.write(json.dumps({"种子提示词": "world", "类别": "writing"}) + "\n")

        items = load_training_data(path)
        assert len(items) == 2
        assert items[0]["prompt"] == "hello"
        assert items[0]["category"] == "coding"
        assert items[1]["prompt"] == "world"

        os.unlink(path)

    def test_empty_file(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write("")

        items = load_training_data(path)
        assert items == []
        os.unlink(path)

    def test_missing_file(self):
        items = load_training_data("/nonexistent/path.jsonl")
        assert items == []

    def test_prompt_field_fallback(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "test", "category": "math"}) + "\n")

        items = load_training_data(path)
        assert len(items) == 1
        assert items[0]["prompt"] == "test"
        os.unlink(path)


class TestLoadTestPrompts:
    def test_load_and_dedup(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"种子提示词": "prompt A", "类别": "coding"}) + "\n")
            f.write(json.dumps({"种子提示词": "prompt A", "类别": "coding"}) + "\n")
            f.write(json.dumps({"种子提示词": "prompt B", "类别": "writing"}) + "\n")

        items = load_test_prompts(path)
        assert len(items) == 2  # deduplicated
        os.unlink(path)

    def test_max_prompts(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({"种子提示词": f"prompt {i}"}) + "\n")

        items = load_test_prompts(path, max_prompts=3)
        assert len(items) == 3
        os.unlink(path)

    def test_missing_file(self):
        items = load_test_prompts("/nonexistent/path.jsonl")
        assert items == []


class TestPipelineInit:
    def test_default_config(self):
        p = Pipeline()
        assert p.config.train_data_path == "data/phase3_datasets/train_prompts.jsonl"

    def test_custom_config(self):
        c = PipelineConfig(train_data_path="/tmp/custom.jsonl")
        p = Pipeline(c)
        assert p.config.train_data_path == "/tmp/custom.jsonl"


class TestGenerateTrainingConfig:
    def test_basic(self):
        p = Pipeline(PipelineConfig(api_key="sk-test"))
        cfg = p.generate_training_config()
        assert "+trainer.trajectory.enabled=true" in cfg
        assert "+trainer.trajectory.api_key=sk-test" in cfg
        assert "data/phase3_datasets/train_prompts.jsonl" in cfg
        assert "data/rubrics_category.json" in cfg

    def test_output_file(self):
        p = Pipeline(PipelineConfig(api_key="sk-test"))
        fd, path = tempfile.mkstemp()
        os.close(fd)
        cfg = p.generate_training_config(path)
        with open(path) as f:
            content = f.read().strip()
        assert content == cfg
        os.unlink(path)


class TestExportResults:
    def test_export(self):
        p = Pipeline()
        results = [
            RoundResult(round_number=1, pre_纠错率=0.4, Δ纠错率=-0.05),
            RoundResult(round_number=2, pre_纠错率=0.35, Δ纠错率=-0.03),
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        p.export_results(results, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["轮次结果"]) == 2
        assert data["轮次结果"][0]["pre_纠错率"] == 0.4
        os.unlink(path)
