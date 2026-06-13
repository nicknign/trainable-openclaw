"""Qwen3-4B agent runner with text-based tool calling.

Qwen3-4B does not support native OpenAI ``tool_calls``. This runner embeds
tool descriptions in the system prompt and parses ``<tool_call>`` blocks
from the model's text output.

Handles Qwen3-4B thinking mode: strips ``<think>...</think>`` blocks from
user-facing responses.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _format_tools_as_text(tools: list[dict]) -> str:
    """Convert OpenAI tool schemas to a compact text description."""
    lines = []
    for t in tools:
        fn = t.get("function", t)
        name = fn["name"]
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        param_strs = []
        for pname, pinfo in params.items():
            pdesc = pinfo.get("description", "")
            ptype = pinfo.get("type", "string")
            req_mark = "*" if pname in required else ""
            param_strs.append(f"  {pname}{req_mark} ({ptype}): {pdesc}")
        param_block = "\n".join(param_strs) if param_strs else "  (no parameters)"
        lines.append(f"- {name}: {desc}\n{param_block}")
    return "\n".join(lines)


_SYSTEM_PROMPT_TEMPLATE = """You are a customer service agent helping a customer with their request.
Use the available tools below to look up information and take actions.

AVAILABLE TOOLS:
{tools_text}

INSTRUCTIONS:
1. The customer's first message contains their request and identity — do NOT ask for info they already provided.
2. When you need to use a tool, output EXACTLY this format:
<tool_call>
{{"name": "<tool_name>", "arguments": {{"param1": "value1"}}}}
</tool_call>
3. After the tool result is provided, continue helping or give your answer.
4. Be concise, accurate, and direct.
5. When you have completed the customer's request, give a clear summary.
6. If a tool returns an error, try an alternative approach.

IMPORTANT: Only output ONE <tool_call> block at a time. Wait for the result before the next one."""


class QwenAgentRunner:
    """Agent runner that uses text-based tool calling for Qwen3-4B.

    Same API as ``AgentRunner`` so it's a drop-in replacement in
    ``InteractiveEvaluator``.
    """

    def __init__(
        self,
        tools: list[dict],
        tool_executor: Callable[[str, dict], dict],
        model: str = "qwen3-4b",
        api_key: str = "no-key",
        base_url: str = "http://localhost:8000/v1",
    ):
        self._tools = tools
        self._execute_tool = tool_executor
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = None
        self._conversation_messages: list[dict] = []
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            tools_text=_format_tools_as_text(tools)
        )

    def reset_conversation(self):
        """Reset conversation state for a new task."""
        self._conversation_messages = [
            {"role": "system", "content": self._system_prompt}
        ]

    def run(self, user_message: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
        """Run one turn: receive user message, call tools if needed, return response.

        Returns:
            (text_response_to_user, list_of_tool_results)
        """
        if not self._conversation_messages:
            self.reset_conversation()

        # Append user message
        self._conversation_messages.append({"role": "user", "content": user_message})

        tool_results: list[dict] = []
        client = self._get_client()

        for _ in range(10):  # max 10 internal tool-call iterations per turn
            response = client.chat.completions.create(
                model=self._model,
                messages=self._conversation_messages,
                temperature=0.3,
                max_tokens=2000,
            )
            raw_text = response.choices[0].message.content or ""

            # Parse for tool calls
            tc = _parse_tool_call(raw_text)
            if tc:
                name = tc["name"]
                args = tc.get("arguments", {})
                result = self._execute_tool(name, args)
                tool_results.append({"name": name, "content": result})

                # Feed tool result back into conversation
                tool_feedback = f"<tool_result>\n{json.dumps(result, ensure_ascii=False)}\n</tool_result>"
                self._conversation_messages.append({"role": "assistant", "content": raw_text})
                self._conversation_messages.append({"role": "user", "content": tool_feedback})
                continue

            # No tool call — this is the final response
            clean = _strip_thinking(raw_text)
            self._conversation_messages.append({"role": "assistant", "content": raw_text})
            return (clean, tool_results)

        fallback = "I wasn't able to complete that request."
        self._conversation_messages.append({"role": "assistant", "content": fallback})
        return (fallback, tool_results)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client


def _parse_tool_call(text: str) -> dict | None:
    """Extract a ``<tool_call>{...}</tool_call>`` block from the text."""
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        logger.warning("Failed to parse tool_call JSON: %s", m.group(1)[:200])
        return None


def _strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from Qwen3-4B output.

    Returns the text after the last ``</think>`` tag, or the original text
    if no think blocks are present.
    """
    # Remove all <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove any remaining unclosed <think> tag and content after it
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()
