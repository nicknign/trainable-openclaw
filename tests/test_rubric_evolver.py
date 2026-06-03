"""Tests for trainable_openclaw.evaluation.rubric_evolver."""

import pytest
import tempfile
import os
import json
import sqlite3
from unittest.mock import MagicMock, patch, PropertyMock

from trainable_openclaw.evaluation.rubric_evolver import (
    EvolutionTrigger,
    EvolutionResult,
    RubricEvolver,
    extract_dimensions_from_text,
    extract_dimensions_from_telemetry,
)


class TestEvolutionTrigger:
    def test_defaults(self):
        t = EvolutionTrigger()
        assert t.reason == ""
        assert t.low_score_count == 0
        assert t.low_score_threshold == 0.3
        assert t.min_low_samples == 10
        assert t.min_new_corrections == 5

    def test_has_reason(self):
        t = EvolutionTrigger(
            reason="低分样本过多",
            low_score_count=15,
            new_correction_count=8,
        )
        assert "低分样本过多" in t.reason
        assert t.low_score_count == 15


class TestEvolutionResult:
    def test_defaults(self):
        r = EvolutionResult()
        assert r.triggered is False
        assert r.new_rubrics == 0
        assert r.details == []

    def test_triggered(self):
        r = EvolutionResult(
            triggered=True,
            trigger_reason="test",
            new_rubrics=3,
            archived_rubrics=1,
        )
        assert r.triggered
        assert r.new_rubrics == 3
        assert r.archived_rubrics == 1


class TestExtractDimensionsFromText:
    def test_code_dimension(self):
        dims = extract_dimensions_from_text("变量名a和b不够表意")
        assert "命名规范" in dims

    def test_math_dimension(self):
        dims = extract_dimensions_from_text("计算步骤跳了一步，结果不对")
        assert any(d in dims for d in ["计算准确性", "步骤完整性"])

    def test_fact_dimension(self):
        dims = extract_dimensions_from_text("这个事实是编造的，不准确")
        assert "事实准确性" in dims

    def test_multiple_dimensions(self):
        dims = extract_dimensions_from_text(
            "代码逻辑有bug，而且变量命名不规范，也没加类型注解"
        )
        assert "代码正确性" in dims
        assert "命名规范" in dims
        assert "类型注解" in dims

    def test_no_match(self):
        dims = extract_dimensions_from_text("xyz 无意义的文本 123")
        assert dims == ["其他"]

    def test_empty(self):
        dims = extract_dimensions_from_text("")
        assert dims == ["其他"]


class TestExtractDimensionsFromTelemetry:
    def test_basic(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE telemetry_events ("
            "session_id TEXT, trace_id TEXT, event_type TEXT, "
            "rating REAL, correction TEXT, created_at TEXT, metadata TEXT)"
        )
        conn.execute(
            "INSERT INTO telemetry_events (correction) VALUES (?)",
            ("变量命名不规范",),
        )
        conn.execute(
            "INSERT INTO telemetry_events (correction) VALUES (?)",
            ("代码有逻辑bug",),
        )
        conn.execute(
            "INSERT INTO telemetry_events (correction) VALUES (?)",
            ("计算步骤不完整",),
        )
        conn.commit()

        mock_store = MagicMock()
        mock_store.conn = conn

        result = extract_dimensions_from_telemetry(mock_store)
        assert len(result) > 0
        dims = [r["维度"] for r in result]
        assert "命名规范" in dims or "代码正确性" in dims

        conn.close()

    def test_empty_store(self):
        mock_store = MagicMock()
        mock_store.conn.execute.return_value.fetchall.return_value = []
        result = extract_dimensions_from_telemetry(mock_store)
        assert result == []

    def test_error_handling(self):
        mock_store = MagicMock()
        mock_store.conn.execute.side_effect = Exception("DB error")
        result = extract_dimensions_from_telemetry(mock_store)
        assert result == []


class TestRubricEvolverInit:
    def test_defaults(self):
        evolver = RubricEvolver()
        assert evolver.min_low_samples == 10
        assert evolver.max_rubrics == 8
        assert evolver.stale_days == 30
        assert evolver.model == "deepseek-v4-flash"

    def test_custom(self):
        evolver = RubricEvolver(
            api_key="sk-test",
            rubrics_path="test.json",
            min_low_samples=5,
            max_rubrics=12,
            stale_days=14,
        )
        assert evolver.api_key == "sk-test"
        assert evolver.rubrics_path == "test.json"
        assert evolver.min_low_samples == 5
        assert evolver.max_rubrics == 12
        assert evolver.stale_days == 14

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key-123")
        evolver = RubricEvolver()
        assert evolver.api_key == "env-key-123"


class TestRubricEvolverBuildTrigger:
    def _make_store_with_tables(self):
        """Create an in-memory SQLite store with both telemetry_events and messages."""
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE telemetry_events ("
            "session_id TEXT, trace_id TEXT, event_type TEXT, "
            "rating REAL, correction TEXT, created_at TEXT, metadata TEXT)"
        )
        conn.execute(
            "CREATE TABLE messages ("
            "id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, "
            "content TEXT, prompt TEXT, created_at TEXT, reasoning TEXT, metadata TEXT)"
        )
        mock_store = MagicMock()
        mock_store.conn = conn
        return mock_store, conn

    def test_below_thresholds(self):
        evolver = RubricEvolver(min_low_samples=100, min_new_corrections=100)
        mock_store, conn = self._make_store_with_tables()
        conn.execute(
            "INSERT INTO telemetry_events (rating, correction, created_at) "
            "VALUES (0.5, '', datetime('now'))"
        )
        conn.commit()

        trigger = evolver._build_trigger(mock_store)
        assert trigger.reason == ""

        conn.close()

    def test_low_scores_trigger(self):
        evolver = RubricEvolver(min_low_samples=2, min_new_corrections=100)
        mock_store, conn = self._make_store_with_tables()
        conn.execute(
            "INSERT INTO telemetry_events (rating, correction, created_at) "
            "VALUES (0.1, '', datetime('now'))"
        )
        conn.execute(
            "INSERT INTO telemetry_events (rating, correction, created_at) "
            "VALUES (0.2, '', datetime('now'))"
        )
        conn.commit()

        trigger = evolver._build_trigger(mock_store)
        assert "低分样本" in trigger.reason
        assert trigger.low_score_count == 2

        conn.close()


class TestRubricEvolverArchiveStale:
    def test_no_rubrics(self):
        evolver = RubricEvolver(rubrics_path="/nonexistent/path.json")
        count = evolver.archive_stale()
        assert count == 0


class TestRubricEvolverGetStats:
    def test_no_file(self):
        evolver = RubricEvolver(rubrics_path="/nonexistent/path.json")
        stats = evolver.get_rubric_stats()
        assert stats["活跃数"] == 0
        assert stats["总数"] == 0


class TestRubricEvolverTrajectoryExtraction:
    def test_extract_basic(self):
        evolver = RubricEvolver()
        corrections = [
            {
                "prompt": "写个排序函数",
                "answer": "def sort(a): ...",
                "correction": "变量名不规范",
                "rating": 2,
            }
        ]
        trajectories = evolver._extract_trajectory_like(corrections)
        assert len(trajectories) == 1
        t = trajectories[0]
        assert t["种子提示词"] == "写个排序函数"
        assert t["最终回答"] == "def sort(a): ..."
        assert len(t["对话消息"]) == 3  # user + assistant + correction

    def test_extract_empty(self):
        evolver = RubricEvolver()
        trajectories = evolver._extract_trajectory_like([])
        assert trajectories == []
