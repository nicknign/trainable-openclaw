"""
User Sim Agent (Phase 1.5 S2).

A strong model (DeepSeek-v4-flash) plays a demanding user persona,
reviews Qwen3-4B responses, and gives specific corrections when
errors are found. Used to generate correction-dialogue training data.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

PERSONAS: dict[str, dict] = {
    "coder_zhang": {
        "name": "张工",
        "role": "资深后端工程师，10年Python经验",
        "focus": [
            "代码正确性（逻辑是否正确，边界条件是否处理）",
            "命名规范（变量名是否表意，是否遵循PEP 8）",
            "类型注解（是否使用类型标注，泛型是否正确）",
            "错误处理（异常是否被捕获和处理）",
            "性能（时间复杂度是否合理，是否有不必要的开销）",
        ],
        "style": "技术向、精确、会指出具体行和问题，不泛泛而谈",
        "strictness": "high",
    },
    "writer_li": {
        "name": "李编辑",
        "role": "资深文字编辑，曾获文学奖",
        "focus": [
            "文笔流畅度（句子是否通顺，节奏是否自然）",
            "意境与表达（用词是否精准优美，是否有画面感）",
            "结构逻辑（段落组织是否合理，论证是否完整）",
            "读者意识（是否考虑目标读者，信息密度是否恰当）",
        ],
        "style": "感性但有见地，关注整体感受胜过细节，会引用具体句子说明",
        "strictness": "medium",
    },
    "math_student": {
        "name": "王同学",
        "role": "数学系研究生，正在准备博士资格考试",
        "focus": [
            "计算准确性（数值是否正确，推导是否有误）",
            "步骤完整性（每一步是否清晰，跳步是否合理）",
            "表达严谨性（定义是否明确，定理引用是否正确）",
            "逻辑严密性（假设是否说明，结论是否有依据）",
        ],
        "style": "严格、要求精确，不接受'大概'或'显而易见'的说法",
        "strictness": "high",
    },
    "qa_tester": {
        "name": "陈测试",
        "role": "资深QA工程师，关注信息质量和用户体验",
        "focus": [
            "回答是否完整（是否有遗漏信息）",
            "是否清晰易懂（初学者能否理解）",
            "是否有误导性内容（是否有事实错误）",
            "是否诚实（不知道时是否承认，而不是编造）",
        ],
        "style": "关注信息的完整性和准确性，会追问细节",
        "strictness": "medium",
    },
    "dev_alex": {
        "name": "Alex",
        "role": "全栈工程师，关注生产级代码质量",
        "focus": [
            "安全性（是否有注入风险、权限问题）",
            "可维护性（代码是否易读、是否遵循 SOLID）",
            "可测试性（是否容易写单元测试）",
            "依赖管理（是否有不必要的重依赖）",
        ],
        "style": "工程向，关注生产级质量，会用代码审查的方式指出问题",
        "strictness": "high",
    },
}


@dataclass
class ReviewResult:
    """Result of User Sim reviewing one model response."""

    verdict: str  # "pass" | "correct"
    correction: str  # empty if pass, else specific correction text
    dimension: str  # which check dimension this relates to
    severity: str  # "critical" | "major" | "minor"


@dataclass
class TurnRecord:
    """One turn in a correction dialogue."""

    turn: int
    speaker: str  # "user_sim" | "model"
    content: str
    correction_type: str = ""  # "task" | "correction" | "response" | "final_pass"


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


def build_user_sim_system_prompt(persona_id: str) -> str:
    """Build the system prompt for a User Sim playing a specific persona."""
    p = PERSONAS.get(persona_id)
    if p is None:
        raise ValueError(
            f"Unknown persona '{persona_id}'. Available: {list(PERSONAS.keys())}"
        )

    focus_lines = "\n".join(f"  - {f}" for f in p["focus"])

    return f"""你是一个模拟用户，你的角色是「{p['name']}」——{p['role']}。

你正在测试一个AI助手的回答质量。你的任务：
1. 阅读AI助手的回答
2. 从你的专业角度检查是否有问题
3. 如果有问题，给出具体、可操作的纠正意见
4. 如果没有问题，确认通过

你关注的维度：
{focus_lines}

你的反馈风格：{p['style']}
严格程度：{p['strictness']}

重要规则：
- 纠错必须基于AI助手的实际输出，具体指出哪里有问题，不能泛泛说"不够好"
- 每个纠正意见必须是 AI 助手可以据此修改的具体建议
- 同一维度的问题不在多轮中重复提出（如果AI没改好，可以强调一次）
- 如果3轮纠正后AI仍改不好，标记为"失败"而非继续纠
- 如果AI的回答完全没问题，直接通过，不要为了纠错而纠错

输出格式（严格JSON）：
{{"verdict": "pass" 或 "correct", "correction": "你的具体纠错意见（pass时为空字符串）", "dimension": "涉及的检查维度", "severity": "critical/major/minor"}}"""


def build_final_check_prompt(persona_id: str) -> str:
    """Build the prompt for final satisfaction check."""
    p = PERSONAS.get(persona_id)
    return f"""作为「{p['name']}」，请最终确认：AI助手的最新回答是否满足你的要求？

回顾整个对话，判断：
- 之前指出的问题是否都已修复？
- 回答质量是否达到了可接受的水平？

输出格式（严格JSON）：
{{"satisfied": true/false, "summary": "简短总结为什么满意或不满意"}}"""


# ---------------------------------------------------------------------------
# User Sim Agent
# ---------------------------------------------------------------------------


class UserSimAgent:
    """A simulated user that reviews model responses and gives corrections.

    Uses a strong model (DeepSeek-v4-flash by default) to play a specific
    user persona and generate realistic, actionable corrections.

    Usage::

        sim = UserSimAgent(persona="coder_zhang")
        result = await sim.review(
            task="写一个排序函数",
            model_response="def sort(a, b): ...",
            history=[],
        )
        if result.verdict == "correct":
            print(f"需要纠正: {result.correction}")
    """

    def __init__(
        self,
        persona: str = "qa_tester",
        model: str = "deepseek-v4-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.persona = persona
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        self.system_prompt = build_user_sim_system_prompt(persona)
        self._client = None

    def _get_client(self):
        """Lazy-init OpenAI-compatible client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package required. Install: pip install openai"
                )
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    async def review(
        self,
        task: str,
        model_response: str,
        history: list[dict],
    ) -> ReviewResult:
        """Review the model's latest response.

        Args:
            task: The original user task/prompt.
            model_response: Qwen3-4B's latest response.
            history: Previous turns [{"speaker": ..., "content": ...}, ...].

        Returns:
            ReviewResult with verdict (pass/correct) and optional correction.
        """
        client = self._get_client()

        # Build messages for the review
        messages = [{"role": "system", "content": self.system_prompt}]

        # Provide context: the task and conversation so far
        context = f"原始任务:\n{task}\n\n"
        if history:
            context += "对话历史:\n"
            for h in history:
                context += f"[{h['speaker']}]: {h['content']}\n\n"
        context += f"[AI助手最新回答]:\n{model_response}"

        messages.append({"role": "user", "content": context})

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,  # Low temp for consistent review
            max_tokens=1000,
        )

        raw = response.choices[0].message.content.strip()

        # Parse JSON from response (handle markdown code fences)
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:]) if len(lines) > 1 else raw
            if raw.endswith("```"):
                raw = raw[:-3]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse User Sim JSON: {raw[:200]}")
            # Fallback: treat as pass to avoid false corrections
            return ReviewResult(
                verdict="pass",
                correction="",
                dimension="parse_error",
                severity="minor",
            )

        return ReviewResult(
            verdict=data.get("verdict", "pass"),
            correction=data.get("correction", ""),
            dimension=data.get("dimension", ""),
            severity=data.get("severity", "minor"),
        )

    async def final_check(
        self,
        task: str,
        final_response: str,
        history: list[dict],
    ) -> tuple[bool, str]:
        """Final satisfaction check after corrections.

        Returns:
            (satisfied: bool, summary: str)
        """
        client = self._get_client()

        check_prompt = build_final_check_prompt(self.persona)
        messages = [{"role": "system", "content": check_prompt}]

        context = f"原始任务:\n{task}\n\n"
        if history:
            context += "完整对话历史:\n"
            for h in history:
                context += f"[{h['speaker']}]: {h['content']}\n\n"
        context += f"\nAI助手最终回答:\n{final_response}"

        messages.append({"role": "user", "content": context})

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:]) if len(lines) > 1 else raw
            if raw.endswith("```"):
                raw = raw[:-3]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse final check JSON: {raw[:200]}")
            return False, "JSON parse error"

        return data.get("satisfied", False), data.get("summary", "")


# ---------------------------------------------------------------------------
# Persona selector
# ---------------------------------------------------------------------------


def select_persona(category: str) -> str:
    """Select a suitable persona for a given prompt category.

    Covers all 32 LMSYS categories with balanced persona distribution:
      coder_zhang: code/engineering/debugging tasks
      writer_li:   writing/editing/creative/language tasks
      math_student: math/logic/reasoning tasks
      dev_alex:    engineering quality/security tasks
      qa_tester:   general QA, explanation, translation (fallback)
    """
    mapping = {
        # 张工 — 代码/教程/文本转换
        "coding": "coder_zhang",
        "debugging": "coder_zhang",
        "tutorial": "coder_zhang",
        "text manipulation": "coder_zhang",
        # 李编辑 — 写作/语言/聊天
        "creative writing": "writer_li",
        "copywriting": "writer_li",
        "proofreading": "writer_li",
        "paraphrasing": "writer_li",
        "specific format writing": "writer_li",
        "roleplaying": "writer_li",
        "text completion": "writer_li",
        "translation": "writer_li",
        "free-form chat": "writer_li",
        # 王同学 — 数学/逻辑/推理/对比/总结
        "math": "math_student",
        "science": "math_student",
        "logical reasoning": "math_student",
        "reasoning": "math_student",
        "spatial reasoning": "math_student",
        "pattern recognition": "math_student",
        "debating": "math_student",
        "ethical reasoning": "math_student",
        "text comparison": "math_student",
        "summarization": "math_student",
        # Alex — 工程/规划/头脑风暴/解释/信息提取
        "engineering": "dev_alex",
        "instruction following": "dev_alex",
        "planning and scheduling": "dev_alex",
        "brainstorming": "dev_alex",
        "explanation": "dev_alex",
        "information extraction": "dev_alex",
        # 陈测试 — QA/分类/常识（默认兜底）
        "question answering": "qa_tester",
        "question generation": "qa_tester",
        "sentiment analysis": "qa_tester",
        "text classification": "qa_tester",
        "trivia": "qa_tester",
        "unknown": "qa_tester",
    }
    return mapping.get(category.lower(), "qa_tester")
