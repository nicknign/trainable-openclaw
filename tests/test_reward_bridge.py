"""Tests for trainable_openclaw.training.reward_bridge — no GPU required."""

import json
import os
import tempfile

import pytest


@pytest.fixture
def temp_rubrics_file():
    """Create a temporary rubrics.json with sample rubrics."""
    rubrics = [
        {
            "id": "abc123",
            "名称": "代码命名规范",
            "评分提示词": "检查代码命名质量。满分10分。\n\n待检查内容：\n{content}",
            "来源模式": "命名规范",
            "版本": 1,
            "命中次数": 0,
            "最后命中时间": 0.0,
            "状态": "活跃",
            "创建时间": 1000.0,
        },
        {
            "id": "def456",
            "名称": "计算准确性",
            "评分提示词": "检查计算是否正确。满分10分。\n\n待检查内容：\n{content}",
            "来源模式": "计算准确性",
            "版本": 1,
            "命中次数": 2,
            "最后命中时间": 2000.0,
            "状态": "活跃",
            "创建时间": 1000.0,
        },
        {
            "id": "archived1",
            "名称": "已归档规则",
            "评分提示词": "检查xxx。\n\n待检查内容：\n{content}",
            "来源模式": "旧模式",
            "版本": 1,
            "命中次数": 0,
            "最后命中时间": 0.0,
            "状态": "归档",
            "创建时间": 500.0,
        },
    ]
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rubrics, f, ensure_ascii=False)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


class TestRewardBridge:
    """Unit tests for RewardBridge (no real API calls)."""

    def test_construction(self, temp_rubrics_file):
        from trainable_openclaw.training.reward_bridge import RewardBridge

        bridge = RewardBridge(
            rubrics_path=temp_rubrics_file,
            api_key="sk-test",
            max_rubrics=10,
            reward_mode="mean",
        )
        assert bridge.rubrics_path.name.endswith(".json")
        assert bridge.api_key == "sk-test"
        assert bridge.reward_mode == "mean"
        assert bridge.max_rubrics == 10

    def test_loads_only_active_rubrics(self, temp_rubrics_file):
        from trainable_openclaw.training.reward_bridge import RewardBridge

        bridge = RewardBridge(rubrics_path=temp_rubrics_file)
        rubrics = bridge.rubrics
        # 2 active, 1 archived → should load 2
        assert len(rubrics) == 2
        names = [r.名称 for r in rubrics]
        assert "代码命名规范" in names
        assert "计算准确性" in names
        assert "已归档规则" not in names

    def test_max_rubrics_cap(self, temp_rubrics_file):
        from trainable_openclaw.training.reward_bridge import RewardBridge

        bridge = RewardBridge(rubrics_path=temp_rubrics_file, max_rubrics=1)
        rubrics = bridge.rubrics
        assert len(rubrics) == 1

    def test_missing_file_returns_zero(self):
        from trainable_openclaw.training.reward_bridge import RewardBridge

        bridge = RewardBridge(rubrics_path="nonexistent_file.json")
        rewards = bridge.score_responses("prompt", ["response1", "response2"])
        assert rewards == [0.0, 0.0]

    def test_empty_rubrics_returns_zero(self, temp_rubrics_file):
        from trainable_openclaw.training.reward_bridge import RewardBridge

        # Use a file with no active rubrics
        bridge = RewardBridge(rubrics_path=temp_rubrics_file)
        # Manually clear cached rubrics
        bridge._rubrics = []
        rewards = bridge.score_responses("prompt", ["r1"])
        assert rewards == [0.0]

    def test_create_reward_bridge_helper(self, temp_rubrics_file):
        from trainable_openclaw.training.reward_bridge import create_reward_bridge

        bridge = create_reward_bridge(
            rubrics_path=temp_rubrics_file,
            config={"api_key": "sk-config-key", "reward_mode": "total"},
        )
        assert bridge.api_key == "sk-config-key"
        assert bridge.reward_mode == "total"

    def test_reward_result_dataclass(self):
        from trainable_openclaw.training.reward_bridge import RewardResult

        r = RewardResult(
            response="test response",
            reward=0.75,
            rubric_scores=[7.0, 8.0],
            mean_score=7.5,
            details=[{"rubric_id": "x", "分数": 7.0}],
        )
        assert r.reward == 0.75
        assert r.mean_score == 7.5
        assert len(r.rubric_scores) == 2
