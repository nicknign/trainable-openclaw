"""
Dynamic Rubric Engine — 自主分析错误案例，生成/合并/精炼高通用性 Rubrics.

Pipeline:
  1. extract_errors()   — 从轨迹 JSONL 提取错误案例，按类别分组
  2. analyze_errors()   — LLM 分析每类错误的共性模式
  3. generate_rubrics() — 为每类生成 1-2 条高覆盖 Rubric
  4. merge_rubrics()    — 跨类别合并相似 Rubric
  5. refine_rubrics()   — 精炼最终版，确保数量少、覆盖全

Usage as script:
  python -m trainable_openclaw.evaluation.rubric_engine \
    --input data/trajectories_high_error.jsonl \
    --output data/rubrics_dynamic.json \
    --api-key sk-xxx \
    --max-rubrics 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ErrorCase:
    """单个错误案例."""
    category: str
    persona: str
    verdict: str
    correction_rounds: int
    error_dimensions: list[str] = field(default_factory=list)
    correction_opinion: str = ""
    original_prompt: str = ""
    final_answer: str = ""


@dataclass
class CategoryErrors:
    """某个类别的所有错误案例汇总."""
    category: str
    errors: list[ErrorCase] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.errors)

    @property
    def common_dimensions(self) -> list[tuple[str, int]]:
        dims: dict[str, int] = {}
        for e in self.errors:
            for d in e.error_dimensions:
                dims[d] = dims.get(d, 0) + 1
        return sorted(dims.items(), key=lambda x: -x[1])


@dataclass
class CategoryRubric:
    """某类别专属的 Rubric."""
    name: str
    prompt: str  # 评分提示词
    categories: list[str]  # 适用类别
    score: float = 0.0  # 质量评分 (LLM 自评)


# ---------------------------------------------------------------------------
# RubricEngine
# ---------------------------------------------------------------------------

class RubricEngine:
    """动态 Rubric 管理引擎."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        max_rubrics: int = 8,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model
        self.max_rubrics = max_rubrics
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    # ------------------------------------------------------------------
    # Step 1: Extract errors from trajectory files
    # ------------------------------------------------------------------

    def extract_errors(
        self,
        trajectory_files: list[str],
        min_correction_rounds: int = 1,
    ) -> list[CategoryErrors]:
        """从轨迹文件中提取错误案例，按类别分组.

        Args:
            trajectory_files: JSONL 轨迹文件路径列表
            min_correction_rounds: 最少纠错轮次阈值 (0=全部, 1=只取有纠错的)

        Returns:
            按类别分组的错误案例列表，按案例数降序
        """
        all_errors: dict[str, list[ErrorCase]] = defaultdict(list)

        for fpath in trajectory_files:
            p = Path(fpath)
            if not p.exists():
                logger.warning("Trajectory file not found: %s", fpath)
                continue

            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    verdict = d.get("最终判定", "")
                    rounds = d.get("纠错次数", 0)

                    if min_correction_rounds > 0 and verdict == "直接通过":
                        continue
                    if rounds < min_correction_rounds:
                        continue

                    # Extract error dimensions from conversation messages
                    dims = self._extract_dimensions(d)

                    cat = d.get("类别", "未分类")
                    all_errors[cat].append(ErrorCase(
                        category=cat,
                        persona=d.get("用户画像", ""),
                        verdict=verdict,
                        correction_rounds=rounds,
                        error_dimensions=dims,
                        correction_opinion=self._extract_corrections(d),
                        original_prompt=d.get("种子提示词", ""),
                        final_answer=d.get("最终回答", ""),
                    ))

        result = [CategoryErrors(cat, errs) for cat, errs in all_errors.items()]
        result.sort(key=lambda x: -x.count)
        logger.info(
            "Extracted %d error cases across %d categories",
            sum(r.count for r in result), len(result),
        )
        return result

    def _extract_dimensions(self, trajectory: dict) -> list[str]:
        """从对话消息中提取错误维度."""
        dims = []
        for msg in trajectory.get("对话消息", []):
            if isinstance(msg, dict):
                dim_text = msg.get("维度", "") or msg.get("error_dimension", "")
                if dim_text:
                    dims.append(dim_text.strip())
        return dims if dims else ["未标注"]

    def _extract_corrections(self, trajectory: dict) -> str:
        """从对话消息中提取纠错意见汇总."""
        corrections = []
        for msg in trajectory.get("对话消息", []):
            if isinstance(msg, dict):
                text = msg.get("纠错意见", "") or msg.get("content", "")
                role = msg.get("role", "")
                if role == "user" and text and len(text) > 20:
                    corrections.append(text[:300])
        return "\n---\n".join(corrections[:3])

    # ------------------------------------------------------------------
    # Step 2: LLM analyzes error patterns per category
    # ------------------------------------------------------------------

    async def analyze_errors(
        self,
        all_errors: list[CategoryErrors],
    ) -> dict[str, str]:
        """LLM 分析每类错误的共性模式.

        Returns:
            {category: error_pattern_summary}
        """
        # Build a single merged prompt for all categories
        parts = []
        for i, ce in enumerate(all_errors):
            dims_str = ", ".join(
                f"{d}({c}次)" for d, c in ce.common_dimensions[:6]
            )
            samples = []
            for e in ce.errors[:3]:
                if e.correction_opinion:
                    samples.append(e.correction_opinion[:200])
            samples_str = "\n---\n".join(samples) if samples else "无示例"

            parts.append(
                f"## 类别 {i+1}: {ce.category}\n"
                f"错误案例数: {ce.count}\n"
                f"常见错误维度: {dims_str}\n"
                f"纠错示例:\n{samples_str}\n"
            )

        prompt = (
            "你是一个AI质量分析专家。请分析以下各提示词类别的AI回答常见错误模式。\n\n"
            "对每个类别，提炼出:\n"
            "1. 核心弱点（1-2句话概括该类提示词下AI最容易犯的错误类型）\n"
            "2. 最关键的检查维度（2-3个，按重要性排序）\n\n"
            + "\n".join(parts) +
            "\n\n=== 输出格式（严格JSON对象）===\n"
            '{\n  "类别1名称": {\n'
            '    "核心弱点": "...",\n'
            '    "关键维度": ["维度1", "维度2", "维度3"]\n'
            "  }, ...\n"
            "}\n"
            "只输出JSON对象，不要有其他文字。"
        )

        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
            extra_body={"thinking": {"type": "disabled"}},
        )

        raw = response.choices[0].message.content.strip()
        analysis = self._parse_json(raw)
        return analysis

    # ------------------------------------------------------------------
    # Step 3: Generate rubrics per category
    # ------------------------------------------------------------------

    async def generate_rubrics(
        self,
        all_errors: list[CategoryErrors],
        analysis: dict[str, dict],
    ) -> list[CategoryRubric]:
        """为每类生成 1-2 条高覆盖 Rubric."""
        # Build merged prompt
        parts = []
        for ce in all_errors:
            cat_analysis = analysis.get(ce.category, {})
            weakness = cat_analysis.get("核心弱点", "未知")
            dims = cat_analysis.get("关键维度", [])

            # Show error samples
            samples = []
            for e in ce.errors[:2]:
                if e.correction_opinion:
                    samples.append(e.correction_opinion[:200])
            samples_str = "\n".join(samples) if samples else "无"

            parts.append(
                f"## {ce.category}\n"
                f"核心弱点: {weakness}\n"
                f"关键维度: {', '.join(dims)}\n"
                f"案例数: {ce.count}\n"
                f"错误示例:\n{samples_str}\n"
            )

        prompt = (
            "你是一个评分标准设计专家。请为以下各类提示词设计严格量化的评分 Rubric。\n\n"
            "设计要求:\n"
            "1. 每条 rubric 覆盖 1-3 个相关类别（不要一条覆盖全部）\n"
            "2. 满分10分，扣分制，每项扣分理由明确量化\n"
            "3. 评分结果必须为 JSON: {\"分数\": <0-10>, \"扣分项\": [], \"总结\": \"\"}\n"
            "4. 总数量控制在 5-8 条，优先覆盖错误率高的类别\n"
            "5. 通用维度（如事实准确性）可跨类别共用\n\n"
            + "\n".join(parts) +
            f"\n\n=== 输出格式（严格JSON数组）===\n"
            '[\n  {{\n'
            '    "名称": "rubric名称",\n'
            '    "评分提示词": "完整评分prompt（含评分标准+JSON输出格式+{{content}}占位符）",\n'
            '    "适用类别": ["类别1", "类别2"]\n'
            '  }}, ...\n'
            ']\n'
            "只输出JSON数组，不要有其他文字。"
        )

        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000,
            extra_body={"thinking": {"type": "disabled"}},
        )

        raw = response.choices[0].message.content.strip()
        data = self._parse_json(raw)
        if not isinstance(data, list):
            data = [data] if data else []

        rubrics = []
        for item in data:
            if not isinstance(item, dict):
                continue
            rubrics.append(CategoryRubric(
                name=item.get("名称", "未命名"),
                prompt=item.get("评分提示词", ""),
                categories=item.get("适用类别", []),
            ))

        logger.info("Generated %d rubrics for %d categories", len(rubrics), len(all_errors))
        return rubrics

    # ------------------------------------------------------------------
    # Step 4: Merge overlapping rubrics
    # ------------------------------------------------------------------

    async def merge_rubrics(
        self,
        rubrics: list[CategoryRubric],
    ) -> list[CategoryRubric]:
        """LLM 合并内容重叠的 Rubric."""
        if len(rubrics) <= self.max_rubrics:
            return rubrics

        parts = []
        for i, r in enumerate(rubrics):
            parts.append(
                f"### Rubric {i+1}: {r.name}\n"
                f"适用类别: {', '.join(r.categories)}\n"
                f"评分内容:\n{r.prompt[:400]}\n"
            )

        prompt = (
            f"当前有 {len(rubrics)} 条 Rubric，需要合并到最多 {self.max_rubrics} 条。\n\n"
            "合并原则:\n"
            "1. 内容重叠 >50% 的两条合并为一条\n"
            "2. 合并后适用类别取并集\n"
            "3. 保留更详细的评分标准\n"
            "4. 不要创造新维度，只做合并去重\n\n"
            + "\n".join(parts) +
            f"\n\n=== 输出格式（严格JSON数组，{self.max_rubrics}条以内）===\n"
            '[\n  {{\n'
            '    "名称": "rubric名称",\n'
            '    "评分提示词": "合并后的完整评分prompt",\n'
            '    "适用类别": ["类别1", "类别2"],\n'
            '    "合并来源": ["Rubric X", "Rubric Y"]\n'
            '  }}, ...\n'
            ']\n'
            "只输出JSON数组，不要有其他文字。"
        )

        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4000,
            extra_body={"thinking": {"type": "disabled"}},
        )

        raw = response.choices[0].message.content.strip()
        data = self._parse_json(raw)
        if not isinstance(data, list):
            data = [data] if data else []

        merged = []
        for item in data:
            if not isinstance(item, dict):
                continue
            merged.append(CategoryRubric(
                name=item.get("名称", "未命名"),
                prompt=item.get("评分提示词", ""),
                categories=item.get("适用类别", []),
            ))

        logger.info("Merged %d → %d rubrics", len(rubrics), len(merged))
        return merged

    # ------------------------------------------------------------------
    # Step 5: Refine final rubrics
    # ------------------------------------------------------------------

    async def refine_rubrics(
        self,
        rubrics: list[CategoryRubric],
    ) -> list[CategoryRubric]:
        """LLM 精炼 Rubric 确保最大覆盖、最少数量."""
        parts = []
        for i, r in enumerate(rubrics):
            parts.append(
                f"### {i+1}. {r.name}\n"
                f"类别: {', '.join(r.categories)}\n"
                f"内容:\n{r.prompt}\n"
            )

        prompt = (
            "你是一个评分标准精炼专家。请精炼以下 Rubric 集合。\n\n"
            "精炼原则:\n"
            "1. 每条 rubric 的评分标准必须量化到可机械执行（具体扣几分）\n"
            "2. 消除模糊表述（如'较好''较差'改为具体数值门槛）\n"
            "3. 确保 JSON 输出格式指令统一且明确\n"
            "4. 适用类别标注准确\n"
            "5. 保持或减少当前数量\n\n"
            + "\n".join(parts) +
            "\n\n=== 输出格式（严格JSON数组）===\n"
            '[\n  {{\n'
            '    "名称": "rubric名称",\n'
            '    "评分提示词": "精炼后的完整评分prompt",\n'
            '    "适用类别": ["类别1", "类别2"]\n'
            '  }}, ...\n'
            ']\n'
            "只输出JSON数组。"
        )

        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4000,
            extra_body={"thinking": {"type": "disabled"}},
        )

        raw = response.choices[0].message.content.strip()
        data = self._parse_json(raw)
        if not isinstance(data, list):
            data = [data] if data else []

        refined = []
        for item in data:
            if not isinstance(item, dict):
                continue
            refined.append(CategoryRubric(
                name=item.get("名称", "未命名"),
                prompt=item.get("评分提示词", ""),
                categories=item.get("适用类别", []),
            ))

        logger.info("Refined %d rubrics", len(refined))
        return refined

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def run(
        self,
        trajectory_files: list[str],
        existing_rubrics_path: str = "",
    ) -> list[CategoryRubric]:
        """运行完整流水线: 提取 → 分析 → 生成 → 合并 → 精炼.

        Returns:
            最终 Rubric 列表
        """
        logger.info("=== RubricEngine pipeline start ===")

        # Step 1: Extract
        all_errors = self.extract_errors(trajectory_files)
        if not all_errors:
            logger.warning("No error cases found — aborting")
            return []

        logger.info(
            "Categories: %s",
            ", ".join(f"{ce.category}({ce.count})" for ce in all_errors[:10]),
        )

        # Step 2: Analyze
        logger.info("Step 2: Analyzing error patterns...")
        analysis = await self.analyze_errors(all_errors)
        logger.info("Analyzed %d categories", len(analysis))

        # Step 3: Generate
        logger.info("Step 3: Generating rubrics...")
        rubrics = await self.generate_rubrics(all_errors, analysis)
        logger.info(
            "Generated: %s",
            ", ".join(f"{r.name}[{','.join(r.categories[:2])}]" for r in rubrics),
        )

        # Step 4: Merge (if needed)
        if len(rubrics) > self.max_rubrics:
            logger.info("Step 4: Merging %d → ≤%d rubrics...", len(rubrics), self.max_rubrics)
            rubrics = await self.merge_rubrics(rubrics)

        # Step 5: Refine
        logger.info("Step 5: Refining rubrics...")
        rubrics = await self.refine_rubrics(rubrics)

        logger.info("=== Pipeline complete: %d rubrics ===", len(rubrics))
        for r in rubrics:
            logger.info("  - %s → [%s]", r.name, ", ".join(r.categories))

        return rubrics

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, rubrics: list[CategoryRubric], path: str):
        """保存 Rubric 到 JSON 文件，兼容 RubricStore 格式."""
        import hashlib
        import time

        data = []
        for r in rubrics:
            rubric_id = hashlib.md5(r.name.encode()).hexdigest()[:12]
            data.append({
                "id": rubric_id,
                "名称": r.name,
                "评分提示词": r.prompt,
                "来源模式": f"rubric_engine (categories: {', '.join(r.categories)})",
                "版本": 1,
                "命中次数": 0,
                "最后命中时间": 0.0,
                "状态": "活跃",
                "创建时间": time.time(),
                "适用类别": r.categories,
            })

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d rubrics to %s", len(data), path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> dict | list:
        """Parse JSON from LLM response, with markdown fence stripping."""
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
            logger.warning("JSON parse failed for: %s", raw[:200])
            return {} if text.startswith("{") else []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RubricEngine — 动态分析错误案例，生成高通用性 Rubrics",
    )
    parser.add_argument("--input", nargs="+", required=True,
                        help="轨迹 JSONL 文件路径 (可多个)")
    parser.add_argument("--output", default="data/rubrics_dynamic.json",
                        help="输出 Rubric JSON 文件路径")
    parser.add_argument("--api-key", default="",
                        help="DeepSeek API key (默认从环境变量 DEEPSEEK_API_KEY 读取)")
    parser.add_argument("--base-url", default="",
                        help="DeepSeek API base URL")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--max-rubrics", type=int, default=8,
                        help="最大 Rubric 数量 (默认 8)")
    parser.add_argument("--min-correction-rounds", type=int, default=1,
                        help="最少纠错轮次阈值 (0=全部轨迹)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="详细日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engine = RubricEngine(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_rubrics=args.max_rubrics,
    )

    rubrics = asyncio.run(engine.run(
        trajectory_files=args.input,
    ))

    engine.save(rubrics, args.output)
    print(f"\nDone. {len(rubrics)} rubrics saved to {args.output}")

    if args.verbose:
        for i, r in enumerate(rubrics):
            print(f"\n--- Rubric {i+1}: {r.name} ---")
            print(f"Categories: {r.categories}")
            print(r.prompt[:300] + "...")


if __name__ == "__main__":
    main()
