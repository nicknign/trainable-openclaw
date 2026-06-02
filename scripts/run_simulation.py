#!/usr/bin/env python3
"""
Phase 1.5 S2：用户模拟纠错对话流水线
=====================================

读取种子提示词（S1产出），运行 用户模拟Agent ↔ Qwen3-4B 多轮纠错对话。
每轮用户纠错产出训练数据：(错误回答, 纠错意见, 修正后回答)。

对话以 messages 格式记录，Qwen 的 <think> 思考内容提取到 reasoning 字段。

用法：
    # 真实模式（需要 GPU 服务器运行中）
    python scripts/run_simulation.py --no-mock --max-prompts 100

    # 模拟模式（DeepSeek 同时模拟双方，无需 GPU）
    python scripts/run_simulation.py --mock --max-prompts 20

    # 空跑验证（不调用 API）
    python scripts/run_simulation.py --dry-run

    # 查看已有结果的统计
    python scripts/run_simulation.py --stats-only --output data/trajectories_live.jsonl

依赖：
    pip install openai tqdm

环境变量：
    DEEPSEEK_API_KEY  -- DeepSeek API 密钥（或在 .env 文件中设置）
    DEEPSEEK_BASE_URL -- DeepSeek API 地址（默认 https://api.deepseek.com）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# ANSI 颜色
# ---------------------------------------------------------------------------

C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
}

P = {
    "user_sim": f"{C['magenta']}[用户模拟]{C['reset']}",
    "model": f"{C['cyan']}[Qwen3-4B]{C['reset']}",
    "通过": f"{C['green']}[通过]{C['reset']}",
    "纠错": f"{C['yellow']}[纠错]{C['reset']}",
    "失败": f"{C['red']}[失败]{C['reset']}",
    "信息": f"{C['dim']}[信息]{C['reset']}",
}


def trunc(s: str, n: int = 120) -> str:
    """截断过长文本用于显示。"""
    s = s.replace("\n", " ").strip()
    return s[:n] + "..." if len(s) > n else s


def split_think(text: str) -> tuple[str, str]:
    """从 Qwen 回答中提取 <think>...</think>，返回 (思考内容, 正文)。"""
    m = re.match(r"<think>\s*(.*?)\s*</think>\s*(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text


# ---------------------------------------------------------------------------
# 用户画像定义
# ---------------------------------------------------------------------------

PERSONAS: dict[str, dict] = {
    "coder_zhang": {
        "名称": "张工",
        "角色": "资深后端工程师",
        "关注维度": [
            "代码正确性（逻辑、边界条件）",
            "命名规范（变量名表意、PEP 8）",
            "类型注解完整性",
            "错误处理",
            "性能（时间复杂度）",
        ],
        "反馈风格": "技术向、精确，指出具体行和问题",
        "严格程度": "高",
    },
    "writer_li": {
        "名称": "李编辑",
        "角色": "资深文字编辑",
        "关注维度": [
            "文笔流畅度",
            "意境与表达精准度",
            "结构逻辑",
            "读者意识",
        ],
        "反馈风格": "感性但有见地，会引用具体句子",
        "严格程度": "中",
    },
    "math_student": {
        "名称": "王同学",
        "角色": "数学系研究生",
        "关注维度": [
            "计算准确性",
            "步骤完整性",
            "表达严谨性",
            "逻辑严密性",
        ],
        "反馈风格": "严格、要求精确",
        "严格程度": "高",
    },
    "qa_tester": {
        "名称": "陈测试",
        "角色": "资深QA工程师",
        "关注维度": [
            "回答完整性",
            "清晰易懂",
            "是否有误导信息",
            "是否诚实（不编造）",
        ],
        "反馈风格": "关注信息质量，会追问细节",
        "严格程度": "中",
    },
    "dev_alex": {
        "名称": "Alex",
        "角色": "全栈工程师",
        "关注维度": [
            "安全性",
            "可维护性",
            "可测试性",
            "依赖管理",
        ],
        "反馈风格": "工程向，关注生产级质量",
        "严格程度": "高",
    },
}

PERSONA_CATEGORY_MAP = {
    # 张工 — 代码相关
    "coding": "coder_zhang",
    "debugging": "coder_zhang",
    "tutorial": "coder_zhang",
    "text manipulation": "coder_zhang",
    # 李编辑 — 写作/语言相关
    "creative writing": "writer_li",
    "copywriting": "writer_li",
    "proofreading": "writer_li",
    "paraphrasing": "writer_li",
    "specific format writing": "writer_li",
    "roleplaying": "writer_li",
    "text completion": "writer_li",
    "translation": "writer_li",
    "free-form chat": "writer_li",
    # 王同学 — 数学/逻辑/推理
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
    # Alex — 工程/规划/信息处理
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


def select_persona(category: str) -> str:
    return PERSONA_CATEGORY_MAP.get(category.lower(), "qa_tester")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class ReviewResult:
    判定: str  # "通过" | "纠错"
    纠错意见: str
    维度: str
    严重程度: str  # "严重" | "重要" | "轻微"


@dataclass
class Trajectory:
    种子提示词: str
    类别: str
    用户画像: str
    消息列表: list[dict] = field(default_factory=list)
    最终判定: str = ""
    纠错次数: int = 0
    错误信息: str = ""

    def to_dict(self) -> dict:
        最终回答 = ""
        for m in reversed(self.消息列表):
            if m["role"] == "assistant":
                最终回答 = m["content"]
                break

        return {
            "种子提示词": self.种子提示词,
            "类别": self.类别,
            "用户画像": self.用户画像,
            "最终判定": self.最终判定,
            "纠错次数": self.纠错次数,
            "总轮数": len([m for m in self.消息列表 if m["role"] == "assistant"]),
            "对话消息": self.消息列表,
            "最终回答": 最终回答,
            "错误信息": self.错误信息,
        }


# ---------------------------------------------------------------------------
# LLM 客户端（OpenAI 兼容，异步）
# ---------------------------------------------------------------------------


class LLMClient:
    """异步 OpenAI 兼容 API 客户端。"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    async def chat(self, messages: list[dict], temperature: float = 0.3,
                   max_tokens: int = 2000) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 用户模拟 Agent
# ---------------------------------------------------------------------------


class UserSimAgent:
    """模拟用户，审查模型回答。使用 DeepSeek-v4-flash。"""

    def __init__(self, persona_id: str, llm: LLMClient):
        p = PERSONAS[persona_id]
        self.persona_id = persona_id
        self.用户名称 = p["名称"]
        self.llm = llm

        focus_lines = "\n".join(f"  - {f}" for f in p["关注维度"])
        self.system_prompt = f"""你是一个模拟用户，你的角色是「{p['名称']}」——{p['角色']}。

你正在测试一个AI助手的回答质量。你的任务：
1. 阅读AI助手的回答
2. 从你的专业角度检查是否有问题
3. 如果有问题，给出具体、可操作的纠正意见
4. 如果没有问题，确认通过

你关注的维度：
{focus_lines}

你的反馈风格：{p['反馈风格']}
严格程度：{p['严格程度']}

重要规则：
- 纠错必须基于AI助手的实际输出，具体指出哪里有问题，不能泛泛说"不够好"
- 每个纠正意见必须是AI助手可以据此修改的具体建议
- 同一维度的问题不在多轮中重复提出
- 如果3轮纠正后AI仍改不好，标记为"失败"而非继续纠
- 如果AI的回答完全没问题，直接通过

输出格式（严格JSON）：
{{"判定": "通过" 或 "纠错", "纠错意见": "具体的纠错意见（通过时为空）", "维度": "涉及的检查维度", "严重程度": "严重/重要/轻微"}}"""

    async def review(self, task: str, model_response: str,
                     history: list[dict]) -> ReviewResult:
        """审查模型回答，返回通过或纠错意见。"""
        # 给 User Sim 看正文（不含思考内容）
        _, content = split_think(model_response)

        context = f"原始任务:\n{task}\n\n"
        if history:
            context += "对话历史:\n"
            for h in history:
                context += f"[{h['发言者']}]: {h['内容']}\n\n"
        context += f"[AI助手最新回答]:\n{content}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context},
        ]

        raw = await self.llm.chat(messages, temperature=0.3, max_tokens=1000)
        data = self._parse_json(raw)

        return ReviewResult(
            判定=data.get("判定", "通过"),
            纠错意见=data.get("纠错意见", ""),
            维度=data.get("维度", ""),
            严重程度=data.get("严重程度", "轻微"),
        )

    async def final_check(self, task: str, response: str,
                          history: list[dict]) -> tuple[bool, str]:
        """最终满意度检查。"""
        p = PERSONAS[self.persona_id]
        _, content = split_think(response)

        check_prompt = f"""作为「{p['名称']}」，请最终确认：AI助手的最新回答是否满足你的要求？
回顾整个对话，判断之前指出的问题是否都已修复，回答质量是否达到了可接受的水平。

输出格式（严格JSON）：
{{"满意": true/false, "总结": "简短总结为什么满意或不满意"}}"""

        context = f"原始任务:\n{task}\n\n"
        if history:
            context += "完整对话历史:\n"
            for h in history:
                context += f"[{h['发言者']}]: {h['内容']}\n\n"
        context += f"\nAI助手最终回答:\n{content}"

        messages = [
            {"role": "system", "content": check_prompt},
            {"role": "user", "content": context},
        ]
        raw = await self.llm.chat(messages, temperature=0.3, max_tokens=500)
        data = self._parse_json(raw)
        return data.get("满意", False), data.get("总结", "")

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """解析 LLM 输出的 JSON，处理 markdown 代码块包裹。"""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试匹配 {"判定": ...} 格式
            m = re.search(r'\{[^{}]*"判定"[^{}]*\}', text, re.DOTALL)
            if not m:
                m = re.search(r'\{[^{}]*"(?:verdict|satisfied)"[^{}]*\}', text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            print(f"  {P['信息']} JSON 解析失败，原始输出: {raw[:150]}")
            return {}


# ---------------------------------------------------------------------------
# 单条对话引擎
# ---------------------------------------------------------------------------


async def run_one_prompt(
    idx: int,
    total: int,
    seed: dict,
    deepseek: LLMClient,
    qwen: LLMClient,
    max_turns: int,
    max_corrections: int,
    verbose: bool = True,
) -> Trajectory:
    """对一条种子提示词运行完整纠错对话。"""
    prompt = seed["prompt"]
    category = seed.get("category", "general")
    persona_id = select_persona(category)
    persona_name = PERSONAS[persona_id]["名称"]

    user_sim = UserSimAgent(persona_id, deepseek)
    traj = Trajectory(种子提示词=prompt, 类别=category, 用户画像=persona_id)

    # 对话消息列表（最终输出格式）
    all_messages: list[dict] = [{"role": "user", "content": prompt, "reasoning": ""}]

    # 模型 API 用历史（不含 reasoning，API 不区分思考和正文）
    model_history = [{"role": "user", "content": prompt}]

    # User Sim 审查历史
    review_history: list[dict] = []
    correction_count = 0
    current_response = ""

    # 抬头
    if verbose:
        print(f"\n{C['bold']}{'─'*70}{C['reset']}")
        print(f"  [{idx}/{total}] {C['blue']}{category}{C['reset']} | "
              f"用户画像: {C['magenta']}{persona_name}{C['reset']}")
        print(f"  {C['dim']}提示词:{C['reset']} {trunc(prompt, 200)}")

    try:
        # --- 第1轮：初始生成 ---
        t_start = time.time()
        current_response = await qwen.chat(model_history, temperature=0.8, max_tokens=2048)
        t_gen = time.time() - t_start
        model_history.append({"role": "assistant", "content": current_response})

        # 提取思考内容
        thinking, answer = split_think(current_response)
        all_messages.append({"role": "assistant", "content": answer, "reasoning": thinking})

        if verbose:
            print(f"\n  {P['model']} {C['dim']}(生成耗时 {t_gen:.1f}秒){C['reset']}")
            if thinking:
                print(f"  {C['dim']}思考: {trunc(thinking, 100)}{C['reset']}")
            print(f"  {C['dim']}{'─'*60}{C['reset']}")
            for line in answer.split("\n")[:10]:
                print(f"  {C['dim']}│{C['reset']} {line[:100]}")
            if answer.count("\n") > 10:
                print(f"  {C['dim']}│{C['reset']} ... (正文共 {len(answer)} 字符)")

        # --- 审查循环 ---
        for turn_idx in range(1, max_turns + 1):
            t_rev = time.time()
            result = await user_sim.review(
                task=prompt,
                model_response=current_response,
                history=review_history,
            )
            t_rev = time.time() - t_rev

            if result.判定 == "通过":
                if verbose:
                    print(f"\n  {P['通过']} {C['dim']}(审查耗时 {t_rev:.1f}秒){C['reset']}")
                    if correction_count > 0:
                        print(f"  {C['green']}✓ 经 {correction_count} 轮纠错后通过{C['reset']}")
                    else:
                        print(f"  {C['green']}✓ 直接通过 —— 无需纠错{C['reset']}")
                traj.最终判定 = "直接通过" if correction_count == 0 else "纠错后通过"
                break

            # 需要纠错
            correction_count += 1

            if verbose:
                print(f"\n  {P['纠错']} {C['yellow']}第 {turn_idx} 轮纠错{C['reset']} "
                      f"{C['dim']}(审查耗时 {t_rev:.1f}秒){C['reset']}")
                print(f"  {C['yellow']}维度:{C['reset']} {result.维度}")
                print(f"  {C['yellow']}严重程度:{C['reset']} {result.严重程度}")
                print(f"  {C['yellow']}纠错意见:{C['reset']}")
                for line in result.纠错意见.split("\n")[:6]:
                    print(f"    {line[:120]}")
                if len(result.纠错意见) > 500:
                    print(f"    ... (共 {len(result.纠错意见)} 字符)")

            if correction_count > max_corrections:
                satisfied, summary = await user_sim.final_check(
                    task=prompt,
                    response=current_response,
                    history=review_history,
                )
                traj.最终判定 = "纠错后通过" if satisfied else "部分通过"
                if verbose:
                    icon = P['通过'] if satisfied else P['失败']
                    print(f"\n  {icon} 最终判定: {summary[:150]}")
                break

            # 构造纠错消息
            correction_msg = (
                f"关于你之前的回答，我发现一个问题：\n{result.纠错意见}\n"
                f"请修改你的回答，解决这个问题。"
            )
            review_history.append({
                "发言者": "用户模拟",
                "内容": result.纠错意见,
            })

            # 记录纠错消息
            all_messages.append({
                "role": "user",
                "content": correction_msg,
                "reasoning": "",
            })
            model_history.append({"role": "user", "content": correction_msg})

            # 重新生成
            t_gen = time.time()
            current_response = await qwen.chat(model_history, temperature=0.8, max_tokens=2048)
            t_gen = time.time() - t_gen
            model_history.append({"role": "assistant", "content": current_response})

            # 提取思考内容
            thinking, answer = split_think(current_response)
            all_messages.append({
                "role": "assistant",
                "content": answer,
                "reasoning": thinking,
            })

            if verbose:
                print(f"\n  {P['model']} {C['dim']}(重新生成耗时 {t_gen:.1f}秒){C['reset']}")
                if thinking:
                    print(f"  {C['dim']}思考: {trunc(thinking, 100)}{C['reset']}")
                for line in answer.split("\n")[:5]:
                    print(f"    {line[:120]}")
                if answer.count("\n") > 5:
                    print(f"    ... (正文共 {len(answer)} 字符)")

        else:
            # 达到最大轮数
            satisfied, summary = await user_sim.final_check(
                task=prompt,
                response=current_response,
                history=review_history,
            )
            traj.最终判定 = "纠错后通过" if satisfied else "部分通过"
            if verbose:
                print(f"\n  {P['信息']} 已达最大轮数: {traj.最终判定}")

    except Exception as e:
        traj.错误信息 = str(e)
        traj.最终判定 = "失败"
        if verbose:
            print(f"\n  {P['失败']} 出错: {e}")

    traj.消息列表 = all_messages
    traj.纠错次数 = correction_count
    return traj


# ---------------------------------------------------------------------------
# 批量串行执行
# ---------------------------------------------------------------------------


async def run_batch_sequential(
    seeds: list[dict],
    output_file: str,
    deepseek: LLMClient,
    qwen: LLMClient,
    max_turns: int = 5,
    max_corrections: int = 3,
    verbose: bool = True,
) -> dict:
    """串行处理所有种子提示词（一条一条来），带进度条。"""
    from tqdm import tqdm

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    out_f = open(output_file, "w", encoding="utf-8")

    stats = {"总数": 0, "直接通过": 0, "纠错后通过": 0, "部分通过": 0, "失败": 0,
             "总纠错次数": 0}
    start_time = time.time()

    total = len(seeds)
    pbar = tqdm(total=total, unit="条", desc="仿真进度",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]")

    for i, seed in enumerate(seeds):
        traj = await run_one_prompt(
            idx=i + 1,
            total=total,
            seed=seed,
            deepseek=deepseek,
            qwen=qwen,
            max_turns=max_turns,
            max_corrections=max_corrections,
            verbose=verbose,
        )

        # 写入文件
        out_f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
        out_f.flush()

        # 更新统计
        stats["总数"] += 1
        stats[traj.最终判定] = stats.get(traj.最终判定, 0) + 1
        stats["总纠错次数"] += traj.纠错次数

        # 更新进度条
        通过数 = stats.get("直接通过", 0) + stats.get("纠错后通过", 0)
        pbar.set_postfix_str(f"通过率 {通过数/stats['总数']*100:.0f}%")
        pbar.update(1)

    pbar.close()
    out_f.close()

    elapsed = time.time() - start_time
    n = stats["总数"]
    if n > 0 and verbose:
        print(f"\n{C['bold']}{'═'*70}{C['reset']}")
        print(f"{C['bold']}  仿真完成{C['reset']}")
        print(f"{'═'*70}")
        print(f"  总数:             {n}")
        print(f"  总耗时:           {elapsed/60:.1f} 分钟 ({elapsed/n:.1f} 秒/条)")
        print(f"  直接通过:         {stats.get('直接通过',0):>4d} ({stats.get('直接通过',0)/n*100:5.1f}%)")
        print(f"  纠错后通过:       {stats.get('纠错后通过',0):>4d} ({stats.get('纠错后通过',0)/n*100:5.1f}%)")
        print(f"  部分通过:         {stats.get('部分通过',0):>4d} ({stats.get('部分通过',0)/n*100:5.1f}%)")
        print(f"  失败:             {stats.get('失败',0):>4d} ({stats.get('失败',0)/n*100:5.1f}%)")
        闭环数 = stats.get("直接通过", 0) + stats.get("纠错后通过", 0)
        print(f"  闭环率:           {闭环数/n*100:.1f}%")
        print(f"  平均纠错次数:     {stats['总纠错次数']/n:.2f}")
        print(f"  输出文件:         {output_file}")
        print(f"{'═'*70}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_env() -> None:
    """从项目根目录或当前目录加载 .env 文件。"""
    for base in [
        Path(__file__).resolve().parent.parent / ".env",
        Path(".env"),
    ]:
        if base.exists():
            with open(base) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser(
        description="Phase 1.5 S2：用户模拟纠错对话流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed-file", default="data/seed_prompts.jsonl",
                        help="种子提示词文件路径")
    parser.add_argument("--output", default="data/correction_trajectories.jsonl",
                        help="输出轨迹文件路径")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""),
                        help="DeepSeek API 密钥")
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                        help="DeepSeek API 地址")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                        help="用户模拟使用的模型")
    parser.add_argument("--mock", action="store_true", default=False,
                        help="使用 DeepSeek 模拟 Qwen3-4B（无需 GPU）")
    parser.add_argument("--no-mock", dest="mock", action="store_false",
                        help="使用真实 Qwen3-4B（需要 GPU 服务运行中）")
    parser.add_argument("--qwen-url", default="http://localhost:8000/v1",
                        help="Qwen3-4B API 地址")
    parser.add_argument("--max-prompts", type=int, default=0,
                        help="最多处理条数（0=全部）")
    parser.add_argument("--max-turns", type=int, default=5,
                        help="每条最大纠错轮数")
    parser.add_argument("--max-corrections", type=int, default=3,
                        help="每条最大纠错次数")
    parser.add_argument("--quiet", action="store_true",
                        help="只显示进度条，不显示每轮对话详情")
    parser.add_argument("--dry-run", action="store_true",
                        help="空跑验证（不调用 API）")
    parser.add_argument("--stats-only", action="store_true",
                        help="仅查看已有输出文件的统计")
    args = parser.parse_args()

    # 仅查看统计
    if args.stats_only:
        print_stats(args.output if args.output != "data/correction_trajectories.jsonl"
                    else args.output)
        return

    # 验证种子文件
    if not os.path.exists(args.seed_file):
        print(f"错误：找不到种子文件: {args.seed_file}")
        print("  请先运行: python scripts/extract_seed_prompts.py")
        sys.exit(1)

    # 加载种子
    seeds = []
    with open(args.seed_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seeds.append(json.loads(line))
    if args.max_prompts > 0:
        seeds = seeds[:args.max_prompts]

    # 空跑验证
    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"  空跑验证")
        print(f"{'='*60}")
        print(f"  种子文件:   {args.seed_file}")
        print(f"  种子数:     {len(seeds)}")
        cats = Counter(s.get("category", "unknown") for s in seeds)
        print(f"  类别数:     {len(cats)}")
        for cat, cnt in cats.most_common():
            persona = PERSONAS[select_persona(cat)]["名称"]
            print(f"    {cat:<30s} {cnt:>4d} → {persona}")
        personas = Counter(select_persona(s.get("category", "general")) for s in seeds)
        print(f"\n  用户画像分布:")
        for p, cnt in personas.most_common():
            print(f"    {PERSONAS[p]['名称']:<10s} ({p:<16s}) {cnt:>4d}")
        lengths = [s.get("char_count", len(s["prompt"])) for s in seeds]
        print(f"\n  提示词长度: 最短={min(lengths)} 最长={max(lengths)} 平均={sum(lengths)/len(lengths):.0f}")
        mode_text = "模拟模式（DeepSeek 模拟 Qwen3-4B）" if args.mock else "真实模式（使用 Qwen3-4B GPU 服务）"
        print(f"\n  运行模式: {mode_text}")
        print()
        return

    # API 密钥检查
    if not args.api_key:
        print("错误：未设置 API 密钥。请在 .env 中设置 DEEPSEEK_API_KEY 或通过 --api-key 传入")
        sys.exit(1)

    verbose = not args.quiet

    # 模式
    if args.mock:
        mode_text = "模拟模式（DeepSeek 模拟双方）"
    else:
        mode_text = f"真实模式 → {args.qwen_url}"

    # 抬头
    print(f"\n{C['bold']}{'═'*70}{C['reset']}")
    print(f"{C['bold']}  Phase 1.5 S2：用户模拟纠错对话流水线{C['reset']}")
    print(f"{'═'*70}")
    print(f"  种子数:       {len(seeds)}")
    print(f"  运行模式:     {mode_text}")
    print(f"  最大轮数:     {args.max_turns}")
    print(f"  最大纠错:     {args.max_corrections}")
    print(f"  输出文件:     {args.output}")
    print(f"  详细输出:     {'是' if verbose else '否（--quiet）'}")
    est_per_prompt = 25 if args.mock else 35
    est_total = len(seeds) * est_per_prompt / 60
    print(f"  预估耗时:     ~{est_total:.0f} 分钟 ({est_per_prompt}秒/条 × {len(seeds)} 条)")
    print(f"{'═'*70}")

    # 创建客户端
    deepseek = LLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    if args.mock:
        qwen = deepseek
    else:
        qwen = LLMClient(api_key="", base_url=args.qwen_url, model="default")

    # 串行运行
    asyncio.run(run_batch_sequential(
        seeds=seeds,
        output_file=args.output,
        deepseek=deepseek,
        qwen=qwen,
        max_turns=args.max_turns,
        max_corrections=args.max_corrections,
        verbose=verbose,
    ))


def print_stats(output_file: str) -> None:
    """从已有轨迹文件输出统计。"""
    if not os.path.exists(output_file):
        print(f"文件不存在: {output_file}")
        return

    trajectories = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))

    n = len(trajectories)
    if n == 0:
        print("没有找到轨迹数据。")
        return

    verdicts = Counter(t.get("最终判定", "未知") for t in trajectories)
    personas = Counter(t.get("用户画像", "未知") for t in trajectories)

    avg_corrections = sum(t.get("纠错次数", 0) for t in trajectories) / n
    # 兼容新旧格式：新格式用"对话消息"，旧格式用"turns"
    if "对话消息" in trajectories[0]:
        avg_turns = sum(len([m for m in t.get("对话消息", []) if m["role"] == "assistant"]) for t in trajectories) / n
    else:
        avg_turns = sum(t.get("总轮数", t.get("total_turns", 0)) for t in trajectories) / n

    print(f"\n{'='*60}")
    print(f"  仿真结果统计 —— {output_file}")
    print(f"{'='*60}")
    print(f"  总轨迹数:           {n}")
    print(f"  平均纠错次数:       {avg_corrections:.2f}")
    print(f"  平均轮数:           {avg_turns:.2f}")
    直接通过 = verdicts.get("直接通过", 0)
    纠错后通过 = verdicts.get("纠错后通过", 0)
    print(f"  直接通过:           {直接通过} ({直接通过/n*100:.1f}%)")
    print(f"  纠错后通过:         {纠错后通过} ({纠错后通过/n*100:.1f}%)")
    close_rate = (直接通过 + 纠错后通过) / n * 100
    print(f"  闭环率:             {close_rate:.1f}%")

    print(f"\n  判定分布:")
    for v, c in verdicts.most_common():
        bar = "█" * int(c / n * 30)
        print(f"    {v:<12s} {c:>4d}  {bar}")

    print(f"\n  用户画像分布:")
    persona_names = {k: v["名称"] for k, v in PERSONAS.items()}
    for p, c in personas.most_common():
        label = persona_names.get(p, p)
        print(f"    {label:<20s} {c:>4d}")

    # 显示一条样例的对话消息结构
    print(f"\n  对话消息样例 (第1条):")
    t = trajectories[0]
    messages = t.get("对话消息", [])
    if messages:
        for i, m in enumerate(messages[:6]):
            role_label = {"user": "用户", "assistant": "助手"}.get(m["role"], m["role"])
            content_preview = m["content"][:100]
            has_reasoning = "✓" if m.get("reasoning") else "—"
            print(f"    [{i}] {role_label} | 思考:{has_reasoning} | {content_preview}...")
        if len(messages) > 6:
            print(f"    ... 共 {len(messages)} 条消息")
    else:
        # 旧格式
        print(f"    (旧格式，无对话消息字段)")


if __name__ == "__main__":
    main()
