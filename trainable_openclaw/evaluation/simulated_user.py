"""
LLM-driven simulated user for interactive agent evaluation.

Plays the role of a tau-bench customer: observes agent responses and tool
execution results, provides natural language feedback, and determines task
completion.  Designed so evaluation logic mirrors production self-evolution
— the only difference is simulated vs. real user as feedback source.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UserResponse:
    message: str
    status: str        # "continue" | "complete" | "give_up"
    satisfaction: float  # 0.0 - 1.0


_SYSTEM_PROMPT = """You are simulating a real customer interacting with a customer service agent. Stay in character throughout the conversation.

YOUR PERSONA AND TASK:
{persona}

SUCCESS CRITERIA (what needs to happen for you to be satisfied):
{success_criteria}

RULES:
1. You are a regular customer — speak naturally, use everyday language
2. If the agent's response correctly answers ALL your questions (verify against TOOL EXECUTION RESULTS) → thank them, status: complete, satisfaction: 1.0
3. Be precise when checking answers — if the agent says $60 and the tool results show gift_card_balance: 60.0, those MATCH. Don't invent discrepancies.
4. If the agent did something partially right → acknowledge what's correct, clarify what still needs fixing (status: continue)
5. If the agent did something wrong → express confusion, re-explain what you want (status: continue)
6. If you've gone back and forth 3+ times with no progress → express frustration (status: give_up)
7. Never break character — do not say "as a simulated user" or mention the test

Respond ONLY with a JSON object (no markdown fences):
{{"message": "<your natural response to the agent>", "status": "<continue|complete|give_up>", "satisfaction": <0.0-1.0>}}"""


class SimulatedUser:
    """LLM-backed simulated customer for evaluating tau-bench agents."""

    def __init__(
        self,
        task: dict,
        model: str = "deepseek-chat",
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
    ):
        self._task = task
        self._model = model
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self._client = None
        self.history: list[dict] = []  # {"role": "agent"|"user", "content": str}
        self.round_count = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def initial_message(self) -> str:
        """Derive the opening customer message from the task prompt."""
        prompt = self._task.get("prompt", "")
        # The tau-bench prompt format is:
        #   You are <Name>. Your user id is <id>.
        #   You want to <task description>.
        lines = [l.strip() for l in prompt.split("\n") if l.strip()]
        want_lines = [l for l in lines if "want to" in l.lower() or "need to" in l.lower() or "wish to" in l.lower()]
        if want_lines:
            return f"Hi there! {want_lines[0]}"
        # Fallback: use the third line onward as the request
        task_lines = [l for l in lines if not l.lower().startswith("you are") and "user id" not in l.lower()]
        if task_lines:
            return f"Hi there! {' '.join(task_lines)}"
        return "Hi, I need some help with my order."

    def respond(self, agent_message: str, tool_results: list[dict] | None = None) -> UserResponse:
        """Generate the customer's response to the agent's latest message.

        Args:
            agent_message: The text the agent sent to the user.
            tool_results: Optional list of tool execution results the agent
                produced since the last user message.  Each dict has ``name``
                and ``content`` keys.

        Returns:
            UserResponse with the customer's message, conversation status, and
            a satisfaction score.
        """
        self.round_count += 1
        self.history.append({"role": "agent", "content": agent_message})

        prompt = self._build_prompt(agent_message, tool_results)
        raw = self._call_llm(prompt)
        data = self._parse_json(raw)

        result = UserResponse(
            message=data.get("message", "I'm not sure about that."),
            status=data.get("status", "continue"),
            satisfaction=float(data.get("satisfaction", 0.5)),
        )
        self.history.append({"role": "user", "content": result.message})
        return result

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    @staticmethod
    def _format_tool_result(name: str, result: dict) -> str:
        """Format a single tool result as readable key-value pairs for LLM consumption."""
        if isinstance(result, dict):
            lines = []
            for k, v in result.items():
                if k == 'status':
                    continue
                if isinstance(v, dict):
                    nested = ', '.join(
                        f'{nk}={nv}' for nk, nv in v.items()
                        if not isinstance(nv, (dict, list))
                    )
                    lines.append(f'  {k}: {nested}')
                elif isinstance(v, list) and len(v) <= 5:
                    items = ', '.join(str(item)[:80] for item in v)
                    lines.append(f'  {k}: [{items}]')
                else:
                    lines.append(f'  {k}: {v}')
            return f'[{name}]:\n' + '\n'.join(lines) if lines else f'[{name}]: (empty)'
        return f'[{name}]: {str(result)[:300]}'

    def _build_prompt(self, agent_message: str, tool_results: list[dict] | None) -> str:
        parts = []
        parts.append(self._format_history())
        parts.append("")
        parts.append(f"AGENT'S LATEST RESPONSE: {agent_message[:800]}")
        if tool_results:
            parts.append("")
            parts.append("TOOL EXECUTION RESULTS:")
            for tr in tool_results:
                name = tr.get("name", tr.get("function", {}).get("name", "unknown"))
                content = tr.get("content", tr.get("result", ""))
                parts.append("  " + self._format_tool_result(name, content))
        parts.append("")
        parts.append(f"(Round {self.round_count}) Respond as the customer with a JSON object.")
        return "\n".join(parts)

    def _format_history(self) -> str:
        if not self.history:
            return "CONVERSATION: (just started)"
        lines = ["CONVERSATION SO FAR:"]
        for m in self.history[-6:]:
            label = "Customer" if m["role"] == "user" else "Agent"
            lines.append(f"  [{label}]: {m['content'][:300]}")
        return "\n".join(lines)

    def _call_llm(self, prompt: str) -> str:
        client = self._get_client()
        system = _SYSTEM_PROMPT.format(
            persona=self._task.get("prompt", ""),
            success_criteria=self._format_success_criteria(),
        )
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()

    def _format_success_criteria(self) -> str:
        ev = self._task.get("evaluation", {})
        if isinstance(ev, str):
            try:
                ev = ast.literal_eval(ev)
            except (ValueError, SyntaxError):
                ev = {"purpose": str(ev)[:300]}
        parts = []
        purpose = ev.get("purpose") if isinstance(ev, dict) else getattr(ev, "purpose", None)
        assertions = ev.get("nl_assertions") if isinstance(ev, dict) else getattr(ev, "nl_assertions", None)
        if purpose:
            parts.append(f"Goal: {purpose}")
        if assertions:
            for a in assertions:
                parts.append(f"- {a}")
        return "\n".join(parts) if parts else "Complete the customer's request correctly and efficiently."

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to find a JSON object in the text
        m = re.search(r'\{"message"\s*:\s*".+?",\s*"status"\s*:\s*"(?:continue|complete|give_up)"\s*,\s*"satisfaction"\s*:\s*[\d.]+\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        logger.warning("SimulatedUser: could not parse JSON from: %s", raw[:200])
        return {"message": text[:200], "status": "continue", "satisfaction": 0.5}
