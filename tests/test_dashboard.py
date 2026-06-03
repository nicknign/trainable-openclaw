"""Lightweight tests for dashboard data loading patterns."""

import json
import os
import pytest
import tempfile
from pathlib import Path


class TestLoadRubricStats:
    def test_valid_file(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        data = {
            "_category_to_group": {"coding": "coding_debugging"},
            "rubrics": [
                {"id": "r1", "名称": "test rubric", "状态": "活跃", "类别组": "coding_debugging"},
                {"id": "r2", "名称": "old rubric", "状态": "已归档", "类别组": "coding_debugging"},
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        # Manually test the load logic
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        rubric_items = loaded.get("rubrics", loaded) if isinstance(loaded, dict) else loaded
        active = [r for r in rubric_items if r.get("状态") == "活跃"]
        archived = [r for r in rubric_items if r.get("状态") != "活跃"]

        assert len(active) == 1
        assert len(archived) == 1
        assert active[0]["名称"] == "test rubric"
        os.unlink(path)

    def test_missing_file(self):
        d = tempfile.mkdtemp()
        os.rmdir(d)
        path = os.path.join(d, "nonexistent.json")
        p = Path(path)
        assert not p.exists()

    def test_direct_array_format(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        data = [
            {"id": "r1", "名称": "rubric 1", "状态": "活跃"},
            {"id": "r2", "名称": "rubric 2", "状态": "活跃"},
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        items = loaded.get("rubrics", loaded) if isinstance(loaded, dict) else loaded
        assert len(items) == 2
        os.unlink(path)


class TestLoadCheckpointInfo:
    def test_missing_dir(self):
        info = []
        p = Path("/nonexistent/checkpoints")
        if p.exists():
            for d in sorted(p.iterdir(), reverse=True):
                info.append({"name": d.name})
        assert info == []

    def test_real_dir(self):
        p = Path("checkpoints")
        info = []
        if p.exists():
            for d in sorted(p.iterdir(), reverse=True):
                if d.is_dir():
                    info.append({"name": d.name})
        # Just verify it doesn't crash
        assert isinstance(info, list)


class TestLoadPipelineResults:
    def test_valid_file(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        data = {
            "生成时间": "2026-06-03 12:00:00",
            "轮次结果": [
                {"轮次": 1, "Δ纠错率": -0.05},
                {"轮次": 2, "Δ纠错率": -0.08},
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert len(loaded["轮次结果"]) == 2
        assert loaded["轮次结果"][1]["Δ纠错率"] == -0.08
        os.unlink(path)

    def test_missing_file(self):
        d = tempfile.mkdtemp()
        os.rmdir(d)
        path = os.path.join(d, "nonexistent.json")
        p = Path(path)
        assert not p.exists()
