"""
B1: 用户反馈收集与分析 (Phase 2)

从 S3 产出的 Rubric 种子和训练对中，使用 LLM 分析反馈模式，
将相似的纠错聚合为可操作的 feedback pattern。

输入: S3 的 rubric_seeds + training_pairs
输出: [{pattern, frequency, severity, examples, suggested_check}]

数据流:
  纠错意见 × N → LLM 分析 → 反馈模式列表 → B2 Rubric 生成器
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FeedbackPattern:
    """一个从用户反馈中识别的模式。"""
    模式名称: str
    描述: str
    频次: int
    严重程度: str  # "高" | "中" | "低"
    典型示例: list[str] = field(default_factory=list)
    建议检查项: str = ""  # LLM 生成的检查建议


class FeedbackAnalyzer:
    """分析用户反馈，识别模式并聚合。

    Usage::

        analyzer = FeedbackAnalyzer(api_key="sk-xxx")
        patterns = await analyzer.analyze(rubric_seeds, training_pairs)
        # patterns: list[FeedbackPattern]
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    async def analyze(
        self,
        rubric_seeds: list[dict],
        training_pairs: list[dict] | None = None,
    ) -> list[FeedbackPattern]:
        """分析反馈种子和训练对，输出反馈模式列表。

        Args:
            rubric_seeds: S3 产出的维度种子 [{"维度": ..., "频次": ..., "示例": [...]}]
            training_pairs: S3 产出的训练对（可选，用于补充上下文）

        Returns:
            反馈模式列表，按频次降序
        """
        if not rubric_seeds:
            logger.warning("无 Rubric 种子数据，返回空列表")
            return []

        # 构建分析输入
        seeds_text = self._format_seeds(rubric_seeds)

        system_prompt = """你是一个用户反馈分析专家。你会收到一批"用户对AI助手回答的纠错意见"的统计汇总。

你的任务：
1. 识别其中的反馈模式（相似的纠错意见归为一类）
2. 为每个模式生成：
   - 简洁的模式名称（如 "变量命名不规范"）
   - 详细描述（该模式的具体表现）
   - 严重程度（高/中/低，取决于该问题出现的频率和影响面）
   - 建议检查项（如果要把这个模式做成自动评分规则，应该怎么检查？）
3. 合并相似的模式，避免过度细分

输出格式（严格JSON）：
{
  "patterns": [
    {
      "模式名称": "...",
      "描述": "...",
      "严重程度": "高/中/低",
      "建议检查项": "..."
    }
  ]
}"""

        user_prompt = f"""以下是用户纠错意见的统计汇总，请分析其中的反馈模式：

{seeds_text}

请识别出所有有意义的反馈模式，合并相似的，按重要性排序。"""

        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        data = self._parse_json(raw)

        # 映射回原始频次和示例
        patterns = []
        for p in data.get("patterns", []):
            # 尝试匹配原始种子中的频次
            name = p.get("模式名称", "")
            freq = self._match_frequency(name, rubric_seeds)
            examples = self._match_examples(name, rubric_seeds)

            patterns.append(FeedbackPattern(
                模式名称=name,
                描述=p.get("描述", ""),
                频次=freq,
                严重程度=p.get("严重程度", "中"),
                典型示例=examples,
                建议检查项=p.get("建议检查项", ""),
            ))

        patterns.sort(key=lambda x: x.频次, reverse=True)
        return patterns

    def analyze_simple(
        self,
        rubric_seeds: list[dict],
    ) -> list[FeedbackPattern]:
        """同步简化版：不做 LLM 精炼，直接转换 rubric_seeds → FeedbackPattern。

        用于快速验证或无 API key 场景。
        """
        patterns = []
        for seed in rubric_seeds:
            examples = [e.get("纠错意见", "")[:200] for e in seed.get("示例", [])]
            patterns.append(FeedbackPattern(
                模式名称=seed.get("维度", "未知"),
                描述=f"用户多次指出该维度的问题，共 {seed.get('频次', 0)} 次",
                频次=seed.get("频次", 0),
                严重程度="高" if seed.get("频次", 0) >= 5 else "中",
                典型示例=examples,
                建议检查项=f"检查回答在「{seed.get('维度', '')}」方面是否达标",
            ))
        patterns.sort(key=lambda x: x.频次, reverse=True)
        return patterns

    def _format_seeds(self, seeds: list[dict]) -> str:
        """格式化 Rubric 种子为 LLM 输入文本。"""
        lines = []
        for s in seeds[:30]:  # 最多 30 个维度
            lines.append(f"维度: {s.get('维度', '?')}")
            lines.append(f"频次: {s.get('频次', 0)}")
            for i, ex in enumerate(s.get("示例", [])[:3]):
                lines.append(f"  示例{i+1}: {ex.get('纠错意见', '')[:200]}")
            lines.append("")
        return "\n".join(lines)

    def _match_frequency(self, pattern_name: str, seeds: list[dict]) -> int:
        """尝试将 LLM 生成的模式名匹配回原始种子的频次。"""
        for s in seeds:
            if s.get("维度", "") in pattern_name or pattern_name in s.get("维度", ""):
                return s.get("频次", 0)
        return 0

    def _match_examples(self, pattern_name: str, seeds: list[dict]) -> list[str]:
        """匹配示例文本。"""
        for s in seeds:
            if s.get("维度", "") in pattern_name or pattern_name in s.get("维度", ""):
                return [e.get("纠错意见", "")[:200] for e in s.get("示例", [])[:3]]
        return []

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """解析 LLM JSON 输出。"""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:] if len(lines) > 1 else lines
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"B1 JSON 解析失败: {raw[:200]}")
            return {}


def load_rubric_seeds(path: str | Path) -> list[dict]:
    """加载 S3 产出的 Rubric 种子文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_training_pairs(path: str | Path) -> list[dict]:
    """加载 S3 产出的训练对文件。"""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs
