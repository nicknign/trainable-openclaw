"""
B2: LLM 自主生成 Rubrics (Phase 2)

核心模块：从 B1 反馈模式中，让 LLM 自主生成严格量化的评分 Rubric。
每条 Rubric 是一个可独立执行打分的 prompt，Judge (B3) 严格按 Rubric 执行。

Rubric 生命周期:
  新建 → 活跃 → (多次命中) → 更新版本
                    → (长期未命中) → 归档

数据流:
  B1 FeedbackPattern → RubricGenerator → Rubric → RubricStore (持久化)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Rubric:
    """一条评分 Rubric。"""
    id: str
    名称: str
    评分提示词: str  # 给 Judge (B3) 执行的严格量化评分 prompt
    来源模式: str  # 来自哪个 B1 反馈模式
    版本: int = 1
    命中次数: int = 0
    最后命中时间: float = 0.0
    状态: str = "活跃"  # "活跃" | "归档"
    创建时间: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "名称": self.名称,
            "评分提示词": self.评分提示词,
            "来源模式": self.来源模式,
            "版本": self.版本,
            "命中次数": self.命中次数,
            "最后命中时间": self.最后命中时间,
            "状态": self.状态,
            "创建时间": self.创建时间,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Rubric:
        from dataclasses import fields as dc_fields
        known = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class RubricStore:
    """Rubric 持久化存储，支持版本管理和查询。"""

    def __init__(self, path: str | Path = "data/rubrics.json"):
        self.path = Path(path)
        self.rubrics: dict[str, Rubric] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    r = Rubric.from_dict(item)
                    self.rubrics[r.id] = r
            logger.info(f"加载 {len(self.rubrics)} 条 Rubric")

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.rubrics.values()],
                      f, ensure_ascii=False, indent=2)

    def add(self, rubric: Rubric) -> Rubric:
        self.rubrics[rubric.id] = rubric
        self.save()
        return rubric

    def update(self, rubric: Rubric):
        rubric.版本 += 1
        rubric.最后命中时间 = time.time()
        rubric.命中次数 += 1
        self.rubrics[rubric.id] = rubric
        self.save()

    def get(self, rubric_id: str) -> Optional[Rubric]:
        return self.rubrics.get(rubric_id)

    def list_active(self) -> list[Rubric]:
        return [r for r in self.rubrics.values() if r.状态 == "活跃"]

    def match(self, pattern_name: str) -> Optional[Rubric]:
        """检查是否已有匹配某个模式的 Rubric。"""
        for r in self.rubrics.values():
            if r.来源模式 == pattern_name and r.状态 == "活跃":
                return r
            # 名称模糊匹配
            if pattern_name in r.名称 or r.名称 in pattern_name:
                return r
        return None

    def archive_stale(self, days: int = 30):
        """归档长期未使用的 Rubric。"""
        now = time.time()
        for r in self.rubrics.values():
            if (now - r.最后命中时间) > days * 86400 and r.状态 == "活跃":
                r.状态 = "归档"
                logger.info(f"归档 Rubric: {r.名称}")
        self.save()


class RubricGenerator:
    """LLM 驱动的 Rubric 生成器。

    从 B1 反馈模式 → LLM 生成严格评分任务 → Rubric 对象。

    Usage::

        gen = RubricGenerator(api_key="sk-xxx")
        rubric = await gen.generate(pattern)
        store = RubricStore()
        store.add(rubric)
    """

    SYSTEM_PROMPT = """你是一个评分标准设计专家。你的任务是将"用户反馈模式"转化为严格的、可量化的评分规则（Rubric）。

设计要求：
1. **严格量化**：评分标准必须有明确的数值门槛，避免主观判断
2. **可执行**：评分者（另一个 LLM）只需照做，不需要额外推理
3. **扣分制**：满分 10 分，明确列出什么情况扣几分
4. **JSON 输出**：评分结果必须是 JSON 格式，便于程序解析
5. **边界清晰**：考虑边界情况（空、不完整、格式错误等）

示例：
  反馈模式: "变量命名不规范"

  生成的评分提示词:
  ```
  你是一个代码质量检查员。请检查以下代码中的变量命名质量。

  评分标准（满分 10 分）：
  - 所有变量名使用有意义的英文单词（非 a/b/c 等单字母）: +4 分
    - 每出现 1 个无意义单字母变量名（循环变量 i/j 除外）: -1 分
  - 变量名遵循 snake_case (Python) 或 camelCase (JS/Java): +2 分
    - 每出现 1 个不符合命名风格: -0.5 分
  - 变量名能准确表达其含义和用途: +2 分
    - 含义模糊的变量名（如 data, info, tmp 且无上下文限定）: -0.5 分/个
  - 常量使用全大写: +1 分
  - 无拼音或中英混合变量名: +1 分
    - 每出现 1 个拼音变量名: -2 分

  输出格式（严格 JSON）：
  {"分数": <0-10的数值>, "扣分项": ["具体扣分原因"], "总结": "一句话评价"}

  待检查代码：
  {code}
  ```

  请按照以上标准，为下面的反馈模式生成评分提示词。"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        store: RubricStore | None = None,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model
        self.store = store or RubricStore()
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    async def generate(
        self,
        pattern,
        extra_context: str = "",
    ) -> Rubric:
        """从反馈模式生成一条 Rubric。

        Args:
            pattern: B1 FeedbackPattern 对象
            extra_context: 额外上下文（如示例纠错文本）

        Returns:
            生成的 Rubric 对象
        """
        # 检查是否已有匹配的 Rubric
        existing = self.store.match(pattern.模式名称)
        if existing:
            logger.info(f"复用已有 Rubric: {existing.名称} (v{existing.版本})")
            self.store.update(existing)
            return existing

        # 构建生成 prompt
        context = f"反馈模式: {pattern.模式名称}\n"
        context += f"描述: {pattern.描述}\n"
        context += f"严重程度: {pattern.严重程度}\n"
        context += f"建议检查项: {pattern.建议检查项}\n"
        if pattern.典型示例:
            context += "\n典型纠错示例:\n"
            for i, ex in enumerate(pattern.典型示例[:3]):
                context += f"  {i+1}. {ex}\n"
        if extra_context:
            context += f"\n补充上下文:\n{extra_context}"

        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"请为以下反馈模式生成评分 Rubric：\n\n{context}"},
            ],
            temperature=0.4,
            max_tokens=1500,
        )

        prompt_text = response.choices[0].message.content.strip()

        if not prompt_text or len(prompt_text) < 50:
            raise ValueError(f"LLM 返回内容过短或为空 ({len(prompt_text)} chars): {prompt_text[:100]}")

        # 生成唯一 ID
        rubric_id = hashlib.md5(
            f"{pattern.模式名称}:{prompt_text[:100]}".encode()
        ).hexdigest()[:12]

        rubric = Rubric(
            id=rubric_id,
            名称=pattern.模式名称,
            评分提示词=prompt_text,
            来源模式=pattern.模式名称,
        )

        self.store.add(rubric)
        logger.info(f"生成新 Rubric: {rubric.名称} (id={rubric_id})")
        return rubric

    async def generate_all(
        self,
        patterns: list,
    ) -> list[Rubric]:
        """为所有反馈模式生成 Rubric。API 失败时回退到简单模板。"""
        rubrics = []
        for p in patterns:
            try:
                r = await self.generate(p)
                rubrics.append(r)
            except Exception as e:
                logger.warning(f"LLM 生成 Rubric 失败 [{p.模式名称}]: {e}，回退到简单模板")
                r = self._generate_fallback(p)
                rubrics.append(r)
        return rubrics

    def _generate_fallback(self, pattern) -> Rubric:
        """API 失败时的简单模板回退。"""
        existing = self.store.match(pattern.模式名称)
        if existing:
            return existing

        prompt = f"""你是一个质量检查员。请检查以下内容在「{pattern.模式名称}」方面的质量。

评分标准（满分 10 分）：
- 完全满足该维度要求: +5 分
- 存在轻微不足: +3 分（扣 2 分）
- 存在明显缺陷: +1 分（扣 4 分）
- 完全不符合要求: 0 分

{pattern.描述}

{pattern.建议检查项}

输出格式（严格 JSON）：
{{"分数": <0-10的数值>, "扣分项": ["具体扣分原因"], "总结": "一句话评价"}}

待检查内容：
{{content}}"""

        rubric_id = hashlib.md5(pattern.模式名称.encode()).hexdigest()[:12]
        rubric = Rubric(
            id=rubric_id,
            名称=pattern.模式名称,
            评分提示词=prompt,
            来源模式=pattern.模式名称,
        )
        self.store.add(rubric)
        logger.info(f"回退模板生成 Rubric: {rubric.名称}")
        return rubric

    def generate_simple(
        self,
        patterns: list,
    ) -> list[Rubric]:
        """同步简化版：不调用 LLM，用模板生成基础 Rubric。

        用于快速验证逻辑。
        """
        rubrics = []
        for p in patterns:
            existing = self.store.match(p.模式名称)
            if existing:
                rubrics.append(existing)
                continue

            prompt = f"""你是一个质量检查员。请检查以下内容在「{p.模式名称}」方面的质量。

评分标准（满分 10 分）：
- 完全满足该维度要求: +5 分
- 存在轻微不足: +3 分（扣 2 分）
- 存在明显缺陷: +1 分（扣 4 分）
- 完全不符合要求: 0 分

{p.描述}

{p.建议检查项}

输出格式（严格 JSON）：
{{"分数": <0-10的数值>, "扣分项": ["具体扣分原因"], "总结": "一句话评价"}}

待检查内容：
{{content}}"""

            rubric_id = hashlib.md5(p.模式名称.encode()).hexdigest()[:12]
            rubric = Rubric(
                id=rubric_id,
                名称=p.模式名称,
                评分提示词=prompt,
                来源模式=p.模式名称,
            )
            self.store.add(rubric)
            rubrics.append(rubric)

        logger.info(f"简单模式生成 {len(rubrics)} 条 Rubric")
        return rubrics
