"""
B3: Rubric 执行器 / LLM Judge (Phase 2)

接收 Rubric 列表 + 候选回答，逐条执行独立打分。
输出分数向量供 GRPO reward 计算。

核心功能:
- 单回答评分:  M 条 rubric → M 维分数向量
- 多回答评分:  N 个回答 × M 条 rubric → N×M 分数矩阵
- GRPO reward: 将分数矩阵转换为每回答的单一 reward 值
- 位置偏差消除: 随机打乱 rubric 和回答的顺序

Usage::

    judge = JudgeExecutor(api_key="sk-xxx")
    store = RubricStore()
    rubrics = store.list_active()
    scores = await judge.score_answers(
        prompt="写一个排序函数",
        answers=["def sort(a):...", "def merge_sort(seq):..."],
        rubrics=rubrics,
    )
    # scores: [{"分数": 7.5, "扣分项": [...], "总结": "..."}, ...]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RubricScore:
    """单条 Rubric 对单个回答的评分结果。"""
    rubric_id: str
    rubric_name: str
    分数: float
    扣分项: list[str] = field(default_factory=list)
    总结: str = ""
    原始输出: str = ""
    解析错误: str = ""

    @classmethod
    def from_llm_output(cls, rubric_id: str, rubric_name: str, raw: str) -> RubricScore:
        """从 LLM 输出解析评分。"""
        data = _parse_score_json(raw)
        return cls(
            rubric_id=rubric_id,
            rubric_name=rubric_name,
            分数=data.get("分数", 0),
            扣分项=data.get("扣分项", []),
            总结=data.get("总结", ""),
            原始输出=raw,
            解析错误=data.get("_parse_error", ""),
        )


@dataclass
class AnswerScores:
    """一个回答的完整评分结果（M 条 rubric 的分数）。"""
    answer: str
    rubric_scores: list[RubricScore] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        if not self.rubric_scores:
            return 0.0
        return sum(rs.分数 for rs in self.rubric_scores) / len(self.rubric_scores)

    @property
    def total_score(self) -> float:
        return sum(rs.分数 for rs in self.rubric_scores)

    @property
    def score_vector(self) -> list[float]:
        return [rs.分数 for rs in self.rubric_scores]


class JudgeExecutor:
    """LLM Judge 执行器。

    对候选回答执行 Rubric 评分，支持异步并发。

    优化策略:
    - merged 模式: 所有 rubric 合并为一次 API 调用（减少 5x 调用量）
    - thinking 默认关闭: 评分任务不需要深度推理
    - max_tokens=500: JSON 响应很短
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        enable_thinking: bool = False,
        max_concurrent_rubrics: int = 16,
        use_merged: bool = True,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model
        self.enable_thinking = enable_thinking
        self.use_merged = use_merged
        self._client = None
        self._semaphore: asyncio.Semaphore | None = None
        self._rubric_sem = asyncio.Semaphore(max_concurrent_rubrics)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _get_sync_client(self):
        if not hasattr(self, '_sync_client') or self._sync_client is None:
            from openai import OpenAI
            self._sync_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._sync_client

    async def score_one(
        self,
        rubric,
        answer: str,
        content_key: str = "{content}",
    ) -> RubricScore:
        """用一条 Rubric 对一个回答打分。

        Args:
            rubric: Rubric 对象（有 评分提示词 字段）
            answer: 待评分的回答文本
            content_key: 评分提示词中代表待检查内容的占位符
        """
        # 插入待评估内容
        judge_prompt = rubric.评分提示词.replace(content_key, answer)
        # 如果提示词中没有占位符，追加内容
        if content_key not in rubric.评分提示词:
            judge_prompt += f"\n\n待检查内容：\n{answer}"

        client = self._get_client()
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": judge_prompt}],
            "temperature": 0.0,
            "max_tokens": 500,
        }
        if self.enable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

        async with self._rubric_sem:
            response = await client.chat.completions.create(**kwargs)

        raw = response.choices[0].message.content.strip()
        return RubricScore.from_llm_output(rubric.id, rubric.名称, raw)

    async def score_merged(
        self,
        answer: str,
        rubrics: list,
    ) -> list[RubricScore]:
        """合并模式：一次 API 调用对所有 rubric 评分。

        将所有 rubric 合并为一个 prompt，一次调用返回所有分数。
        API 调用量: N → 1（减少 N 倍）。
        """
        if not rubrics:
            return []

        merged_prompt = _build_merged_prompt(rubrics, answer)

        client = self._get_client()
        # Scale max_tokens with rubric count: ~100 tokens per rubric JSON entry
        max_tok = max(800, len(rubrics) * 200)
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": merged_prompt}],
            "temperature": 0.0,
            "max_tokens": max_tok,
        }
        if self.enable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

        async with self._rubric_sem:
            response = await client.chat.completions.create(**kwargs)

        raw = response.choices[0].message.content.strip()
        return _parse_merged_response(raw, rubrics)

    def score_merged_sync(
        self,
        answer: str,
        rubrics: list,
    ) -> list[RubricScore]:
        """Sync version of score_merged for Ray actor compatibility."""
        if not rubrics:
            return []

        merged_prompt = _build_merged_prompt(rubrics, answer)

        client = self._get_sync_client()
        max_tok = max(800, len(rubrics) * 200)
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": merged_prompt}],
            "temperature": 0.0,
            "max_tokens": max_tok,
        }
        if self.enable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

        response = client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content.strip()
        return _parse_merged_response(raw, rubrics)

    def score_answer_sync(
        self,
        answer: str,
        rubrics: list,
    ) -> AnswerScores:
        """Sync version of score_answer for Ray actor compatibility."""
        if self.use_merged:
            try:
                scores = self.score_merged_sync(answer, rubrics)
            except Exception as e:
                logger.error(f"Merged rubric scoring failed: {e}")
                scores = [
                    RubricScore(rubric_id=r.id, rubric_name=r.名称, 分数=0, 解析错误=str(e))
                    for r in rubrics
                ]
            return AnswerScores(answer=answer, rubric_scores=scores)

        # Per-rubric mode
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _score_one(rubric):
            try:
                return self.score_one_sync(rubric, answer)
            except Exception as e:
                logger.error(f"Rubric [{rubric.名称}] 评分失败: {e}")
                return RubricScore(
                    rubric_id=rubric.id,
                    rubric_name=rubric.名称,
                    分数=0,
                    解析错误=str(e),
                )

        with ThreadPoolExecutor(max_workers=min(8, len(rubrics))) as pool:
            futures = [pool.submit(_score_one, r) for r in rubrics]
            scores = [f.result() for f in futures]

        return AnswerScores(answer=answer, rubric_scores=scores)

    def score_one_sync(self, rubric, answer: str) -> RubricScore:
        """Sync version of score_one."""
        judge_prompt = rubric.评分提示词.replace("{content}", answer)
        if "{content}" not in rubric.评分提示词:
            judge_prompt += f"\n\n待检查内容：\n{answer}"

        client = self._get_sync_client()
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": judge_prompt}],
            "temperature": 0.0,
            "max_tokens": 500,
        }
        if self.enable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

        response = client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content.strip()
        return RubricScore.from_llm_output(rubric.id, rubric.名称, raw)

    async def score_answer(
        self,
        answer: str,
        rubrics: list,
        shuffle_rubrics: bool = True,
    ) -> AnswerScores:
        """对一个回答执行所有 Rubric 评分。

        Args:
            answer: 待评分回答
            rubrics: Rubric 对象列表
            shuffle_rubrics: 是否打乱 Rubric 顺序（减少位置偏差）
        """
        if self.use_merged:
            try:
                scores = await self.score_merged(answer, rubrics)
            except Exception as e:
                logger.error(f"Merged rubric scoring failed: {e}")
                scores = [
                    RubricScore(rubric_id=r.id, rubric_name=r.名称, 分数=0, 解析错误=str(e))
                    for r in rubrics
                ]
            return AnswerScores(answer=answer, rubric_scores=scores)

        ordered = list(rubrics)
        if shuffle_rubrics:
            ordered = list(ordered)
            random.shuffle(ordered)

        # 并发评分所有 rubric（受 _rubric_sem 控制并发上限）
        async def _score_with_fallback(rubric):
            try:
                return await self.score_one(rubric, answer)
            except Exception as e:
                logger.error(f"Rubric [{rubric.名称}] 评分失败: {e}")
                return RubricScore(
                    rubric_id=rubric.id,
                    rubric_name=rubric.名称,
                    分数=0,
                    解析错误=str(e),
                )

        tasks = [_score_with_fallback(r) for r in ordered]
        scores = await asyncio.gather(*tasks)

        return AnswerScores(answer=answer, rubric_scores=list(scores))

    async def score_answers(
        self,
        prompt: str,
        answers: list[str],
        rubrics: list,
        shuffle_answers: bool = True,
    ) -> list[dict]:
        """对多个候选回答（GRPO N 个采样）执行评分。

        Args:
            prompt: 原始用户提示词
            answers: GRPO 生成的 N 个候选回答
            rubrics: 活跃的 Rubric 列表
            max_concurrent: 最大并发回答数
            shuffle_answers: 是否打乱回答顺序

        Returns:
            [{answer, scores: [{rubric_id, rubric_name, score, deductions, summary}], mean_score}]
        """
        # 可选打乱
        ordered_answers = list(answers)
        if shuffle_answers:
            random.shuffle(ordered_answers)

        tasks = [self.score_answer(ans, rubrics) for ans in ordered_answers]
        results: list[AnswerScores] = await asyncio.gather(*tasks)

        output = []
        for result in results:
            output.append({
                "回答": result.answer,
                "评分": [
                    {
                        "rubric_id": rs.rubric_id,
                        "rubric名称": rs.rubric_name,
                        "分数": rs.分数,
                        "扣分项": rs.扣分项,
                        "总结": rs.总结,
                        "解析错误": rs.解析错误,
                    }
                    for rs in result.rubric_scores
                ],
                "平均分": result.mean_score,
                "总分": result.total_score,
                "分数向量": result.score_vector,
            })

        # 恢复原始顺序
        if shuffle_answers:
            output.sort(key=lambda x: answers.index(x["回答"]))

        return output

    def score_answers_sync(
        self,
        prompt: str,
        answers: list[str],
        rubrics: list,
    ) -> list[dict]:
        """Sync version of score_answers — uses ThreadPoolExecutor for concurrency.
        Designed for Ray actor compatibility (no asyncio.run() needed).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[AnswerScores] = []

        def _score_one(ans):
            return self.score_answer_sync(ans, rubrics)

        with ThreadPoolExecutor(max_workers=min(8, len(answers))) as pool:
            future_to_idx = {pool.submit(_score_one, a): i for i, a in enumerate(answers)}
            results_by_idx = {}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results_by_idx[idx] = future.result()

        results = [results_by_idx[i] for i in range(len(answers))]

        output = []
        for result in results:
            output.append({
                "回答": result.answer,
                "评分": [
                    {
                        "rubric_id": rs.rubric_id,
                        "rubric名称": rs.rubric_name,
                        "分数": rs.分数,
                        "扣分项": rs.扣分项,
                        "总结": rs.总结,
                        "解析错误": rs.解析错误,
                    }
                    for rs in result.rubric_scores
                ],
                "平均分": result.mean_score,
                "总分": result.total_score,
                "分数向量": result.score_vector,
            })

        return output

    def compute_grpo_rewards(
        self,
        score_results: list[dict],
        reward_mode: str = "mean",
        weights: list[float] | None = None,
    ) -> list[float]:
        """从评分结果计算 GRPO reward 值。

        Args:
            score_results: score_answers 的输出
            reward_mode:
                - "mean": 所有 rubric 分数的加权均值 (0-10 → 0-1)
                - "total": 所有 rubric 分数的加权总和
                - "pass_fail": 平均分 > 6 得 1，否则 0
            weights: rubric 权重列表，与 rubric 顺序对应。
                     未提供时等权。

        Returns:
            每个回答的 reward 值列表
        """
        rewards = []
        for r in score_results:
            score_vec = r.get("分数向量", [])
            if not score_vec:
                rewards.append(0.0)
                continue

            n = len(score_vec)
            w = weights if weights and len(weights) == n else [1.0] * n
            total_w = sum(w) or 1.0

            if reward_mode == "mean":
                reward = sum(s * wi for s, wi in zip(score_vec, w)) / total_w / 10.0
            elif reward_mode == "total":
                reward = sum(s * wi for s, wi in zip(score_vec, w)) / (total_w * 10.0)
            elif reward_mode == "pass_fail":
                avg = sum(s * wi for s, wi in zip(score_vec, w)) / total_w
                reward = 1.0 if avg >= 6.0 else 0.0
            else:
                reward = 0.0

            rewards.append(reward)

        return rewards


def _build_merged_prompt(rubrics: list, answer: str) -> str:
    """构建合并评分 prompt：所有 rubric 合并为一次 API 调用。"""
    parts = []
    for i, r in enumerate(rubrics):
        # 去掉原始 prompt 中的 {content} 占位符和目标前缀，合并 prompt 会统一放置
        prompt = r.评分提示词.replace("{content}", "").replace("\n待检查内容：\n", "\n").strip()
        parts.append(f"## 维度{i + 1}: {r.名称}\n{prompt}\n")

    rubric_list = "\n".join(f"  {i+1}. {r.名称}" for i, r in enumerate(rubrics))

    return (
        f"你是一个多维度AI回答质量评估员。请对以下回答从 {len(rubrics)} 个维度分别评分。\n\n"
        f"评分维度：\n{rubric_list}\n\n"
        f"--- 各维度评分标准 ---\n\n"
        f"{''.join(parts)}\n"
        f"=== 待评估内容 ===\n{answer}\n\n"
        f"=== 输出格式（严格JSON数组）===\n"
        f'[\n  {{"维度": "{rubrics[0].名称}", "分数": <0-10>, "扣分项": [], "总结": ""}},\n'
        f'  ...\n'
        f']\n'
        f"请按维度顺序输出，每个维度一个JSON对象。只输出JSON数组，不要有其他文字。"
    )


def _parse_merged_response(raw: str, rubrics: list) -> list[RubricScore]:
    """解析合并评分的 JSON 数组响应。"""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if len(lines) > 1 else lines
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Merged JSON parse failed, raw: {text[:200]}")
        return [
            RubricScore(rubric_id=r.id, rubric_name=r.名称, 分数=0, 解析错误="merged parse failed")
            for r in rubrics
        ]

    if not isinstance(data, list):
        data = [data]

    scores = []
    for i, r in enumerate(rubrics):
        item = data[i] if i < len(data) else {}
        scores.append(RubricScore(
            rubric_id=r.id,
            rubric_name=r.名称,
            分数=float(item.get("分数", 0)),
            扣分项=item.get("扣分项", []),
            总结=item.get("总结", ""),
            原始输出=json.dumps(item, ensure_ascii=False),
        ))
    return scores


def _parse_score_json(raw: str) -> dict:
    """解析 Judge 返回的评分 JSON。"""
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
        # 尝试提取分数
        import re
        score_match = re.search(r'"分数"\s*:\s*([\d.]+)', raw)
        deductions = re.findall(r'"扣分项"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
        if score_match:
            return {
                "分数": float(score_match.group(1)),
                "扣分项": [],  # 简化处理
                "总结": "解析部分成功",
                "_parse_error": f"完整 JSON 解析失败，仅提取分数: {raw[:100]}",
            }
        logger.warning(f"Judge 评分 JSON 解析失败: {raw[:200]}")
        return {"分数": 0, "扣分项": [], "总结": "解析失败", "_parse_error": raw[:200]}


def load_rubrics_for_judge(path: str | Path = "data/rubrics.json") -> list:
    """从 RubricStore 文件加载 Rubric 对象列表供 Judge 使用。"""
    from trainable_openclaw.evaluation.rubric import Rubric  # noqa
    rubrics = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            r = Rubric.from_dict(item)
            if r.状态 == "活跃":
                rubrics.append(r)
    return rubrics
