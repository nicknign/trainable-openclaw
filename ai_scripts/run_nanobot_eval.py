"""Run interactive evaluation with Qwen3-4B (nanobot) agent + DeepSeek simulated user.

Usage:
    python ai_scripts/run_nanobot_eval.py                  # first task
    python ai_scripts/run_nanobot_eval.py --task 3         # task index 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainable_openclaw.evaluation.interactive_eval import (
    InteractiveEvaluator, TaskResult, EvalReport,
)
from trainable_openclaw.evaluation.simulated_user import SimulatedUser

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NANOBOT_BASE_URL = "http://localhost:8900/v1"
AGENT_MODEL = "qwen3-4b"
AGENT_API_KEY = "no-key"

# Load DeepSeek config from .env
def _load_dotenv(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_env = _load_dotenv(PROJECT_ROOT / ".env")
DEEPSEEK_API_KEY = _env.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = _env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"

PROMPTS_PATH = PROJECT_ROOT / "data" / "tau_bench" / "test_prompts_augmented.jsonl"

# ---------------------------------------------------------------------------
# Prompted Agent Runner (for models without native tool calling)
# ---------------------------------------------------------------------------

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

AGENT_SYSTEM_TEMPLATE = """You are a customer service agent. Use tools to help the customer.

TOOLS (call with <tool_call> JSON blocks):
{tool_descriptions}

FORMAT: <tool_call>
{{"name": "X", "arguments": {{...}}}}
</tool_call>

RULES:
- Extract customer identity from their message; use tools immediately.
- Do NOT re-ask for info the customer already provided.
- Be concise.
- If a tool fails, explain and try alternatives."""


class PromptedAgentRunner:
    """Agent using prompted tool calls via nanobot's single-message API.

    nanobot's OpenAI-compatible API only accepts a single user message per
    request (no multi-turn conversation).  We pack the full context (system
    prompt, conversation history, current instruction) into one user message
    per call and manage multi-turn tool execution locally.
    """

    def __init__(self, tools: list[dict], tool_executor, model: str, api_key: str, base_url: str):
        self._raw_tools = tools
        self._execute = tool_executor
        self._model = model
        self._client = None
        self._api_key = api_key
        self._base_url = base_url
        self._system_prompt = ""
        self._conversation_lines: list[str] = []

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def reset_conversation(self, customer_info: str = ""):
        descs = []
        for t in self._raw_tools:
            fn = t.get("function", t)
            name = fn["name"]
            desc = fn.get("description", "")
            props = fn.get("parameters", {}).get("properties", {})
            required = fn.get("parameters", {}).get("required", [])
            param_lines = []
            for pname, pinfo in props.items():
                req_mark = " (required)" if pname in required else ""
                param_lines.append(f"    {pname}: {pinfo.get('description', '')}{req_mark}")
            # Compact format: name(params): description
            param_str = ", ".join(f"{pname}{'*' if pname in required else ''}" for pname in props)
            descs.append(f"{name}({param_str}): {desc}")
        system = AGENT_SYSTEM_TEMPLATE.format(tool_descriptions="\n\n".join(descs))
        if customer_info:
            system += f"\n\nCUSTOMER INFORMATION (from account):\n{customer_info}"
        self._system_prompt = system
        self._conversation_lines = []

    def _build_single_message(self, instruction: str) -> str:
        """Pack system prompt + history + instruction into one message."""
        parts = [self._system_prompt]
        if self._conversation_lines:
            parts.append("--- HISTORY ---")
            parts.extend(self._conversation_lines[-10:])  # keep last 10 lines
        parts.append("---")
        parts.append(instruction)
        return "\n".join(parts)

    def _call_llm(self, instruction: str) -> str:
        msg = self._build_single_message(instruction)
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": msg}],
            temperature=0.3,
            max_tokens=2000,
        )
        return response.choices[0].message.content or ""

    def run(self, user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
        if not self._system_prompt:
            self.reset_conversation()

        # Record the user message in local history
        self._conversation_lines.append(f"[Customer]: {user_message.strip()}")

        tool_results: list[dict] = []

        # Internal tool-calling loop: call LLM, execute tools, feed back
        current_instruction = user_message
        for _ in range(10):
            content = self._call_llm(current_instruction)

            matches = TOOL_CALL_RE.findall(content)
            if matches:
                self._conversation_lines.append(f"[Agent (tool calls)]: {len(matches)} tool(s)")
                tool_feedback_parts = []
                for match in matches:
                    try:
                        tc = json.loads(match)
                        name = tc["name"]
                        args = tc.get("arguments", {})
                    except (json.JSONDecodeError, KeyError) as e:
                        tool_results.append({"name": "parse_error", "content": {"error": str(e), "raw": match[:200]}})
                        tool_feedback_parts.append(f"Tool call parse error: {e}. Raw: {match[:200]}")
                        continue
                    result = self._execute(name, args)
                    tool_results.append({"name": name, "content": result})
                    result_str = json.dumps(result)
                    self._conversation_lines.append(f"[Tool {name}]: {result_str[:300]}")
                    tool_feedback_parts.append(f"Tool '{name}' returned: {result_str}")

                # Build next instruction with tool results
                current_instruction = (
                    "TOOL RESULTS RECEIVED. Now either call more tools or respond to the customer.\n\n"
                    + "\n".join(tool_feedback_parts)
                )
            else:
                self._conversation_lines.append(f"[Agent]: {content.strip()[:500]}")
                return (content, tool_results)

        fallback = "I wasn't able to complete that request. Let me transfer you to a human agent."
        self._conversation_lines.append(f"[Agent]: {fallback}")
        return (fallback, tool_results)


# ---------------------------------------------------------------------------
# Tool executor (same as run_single_eval.py)
# ---------------------------------------------------------------------------

def build_tool_executor(scenario: str):
    from trainable_openclaw.agent.tau_bench_tools.registry import register_tau_bench_tools
    from trainable_openclaw.agent.tau_bench_tools.mock_db import MockDatabase

    db = MockDatabase(scenario)
    tools = register_tau_bench_tools(scenario)

    def execute(name: str, args: dict) -> dict:
        for t in tools:
            if t.name == name:
                try:
                    return t.execute(args, db.state)
                except Exception as e:
                    return {"error": str(e)}
        return {"error": f"Unknown tool: {name}"}

    tool_schemas = [
        {"type": "function", "function": t.to_schema()["function"]} for t in tools
    ]
    return execute, tool_schemas


def load_tasks() -> list[dict]:
    tasks = []
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


# ---------------------------------------------------------------------------
# Custom evaluator that uses PromptedAgentRunner for agent and
# DeepSeek-backend SimulatedUser for the customer
# ---------------------------------------------------------------------------

class HybridEvaluator:
    """Evaluator with separate backends: nanobot for agent, DeepSeek for user."""

    def __init__(self, agent: PromptedAgentRunner, max_rounds: int = 10):
        self._agent = agent
        self._max_rounds = max_rounds

    def evaluate(self, tasks: list[dict]) -> EvalReport:
        results: list[TaskResult] = []
        for i, task in enumerate(tasks):
            tid = task.get("id", "?")
            print(f"\n{'='*60}")
            print(f"Task {i+1}/{len(tasks)}: {tid} ({task.get('domain', '')})")
            print(f"Prompt: {task['prompt'][:150]}...")
            print(f"{'='*60}")
            tr = self._evaluate_one(task)
            results.append(tr)
            print(f"Result: completed={tr.completed} rounds={tr.rounds} satisfaction={tr.satisfaction} status={tr.status}")
        return self._build_report(results)

    def _evaluate_one(self, task: dict) -> TaskResult:
        # Extract customer identity from the task prompt (name, email, user ID)
        prompt = task.get("prompt", "")
        identity_lines = []
        for line in prompt.split("\n"):
            line = line.strip()
            if not line:
                break  # stop at first blank line (identity is first paragraph)
            if "you are" in line.lower() or "user id" in line.lower() or "email" in line.lower() or "your" in line.lower():
                identity_lines.append(line)
        customer_info = " ".join(identity_lines) if identity_lines else prompt.split("\n")[0].strip()

        sim_user = SimulatedUser(
            task,
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        if hasattr(self._agent, "reset_conversation"):
            self._agent.reset_conversation(customer_info=customer_info)

        conversation: list[dict] = []
        user_msg = sim_user.initial_message

        for r in range(1, self._max_rounds + 1):
            print(f"\n--- Round {r} ---")
            print(f"[SimUser] {user_msg[:200]}")
            agent_msg, tool_results = self._agent.run(user_msg, sim_user.history)
            conversation.append({"role": "agent", "content": agent_msg, "tool_results": tool_results})
            print(f"[Agent] {agent_msg[:300]}")
            if tool_results:
                for tr in tool_results:
                    tc = str(tr.get("content", ""))[:150]
                    print(f"  [Tool: {tr['name']}] {tc}")

            feedback = sim_user.respond(agent_msg, tool_results)
            conversation.append({"role": "user", "content": feedback.message})
            print(f"[SimUser] {feedback.message[:200]} (status={feedback.status}, sat={feedback.satisfaction})")

            if feedback.status == "complete":
                return TaskResult(
                    task_id=task.get("id", "?"), domain=task.get("domain", ""),
                    completed=True, rounds=r, satisfaction=feedback.satisfaction,
                    status="complete", conversation=conversation,
                )
            elif feedback.status == "give_up":
                return TaskResult(
                    task_id=task.get("id", "?"), domain=task.get("domain", ""),
                    completed=False, rounds=r, satisfaction=feedback.satisfaction,
                    status="give_up", conversation=conversation,
                )
            user_msg = feedback.message

        return TaskResult(
            task_id=task.get("id", "?"), domain=task.get("domain", ""),
            completed=False, rounds=self._max_rounds, satisfaction=0.0,
            status="timeout", conversation=conversation,
        )

    def _build_report(self, results: list[TaskResult]) -> EvalReport:
        total = len(results)
        if total == 0:
            return EvalReport(results=[], total_tasks=0, completed_tasks=0,
                              avg_rounds=0.0, first_try_rate=0.0,
                              recovery_rate=0.0, abandonment_rate=0.0)
        completed = [r for r in results if r.completed]
        n_completed = len(completed)
        avg_rounds = sum(r.rounds for r in results) / total
        first_try = sum(1 for r in results if r.first_try_success) / total
        recovery = sum(1 for r in completed if r.rounds > 1) / n_completed if n_completed else 0.0
        abandoned = sum(1 for r in results if r.status == "give_up") / total
        return EvalReport(results=results, total_tasks=total,
                          completed_tasks=n_completed, avg_rounds=avg_rounds,
                          first_try_rate=first_try, recovery_rate=recovery,
                          abandonment_rate=abandoned)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run eval with nanobot agent + DeepSeek user")
    parser.add_argument("--task", type=int, default=0, help="Task index (0-based)")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max rounds")
    args = parser.parse_args()

    tasks = load_tasks()
    task = tasks[args.task]
    tid = task["id"]
    domain = task["domain"]

    print(f"Task: {tid} ({domain})")
    print(f"Agent model: {AGENT_MODEL} @ {NANOBOT_BASE_URL}")
    print(f"SimUser model: {DEEPSEEK_MODEL} @ {DEEPSEEK_BASE_URL}")
    print(f"Max rounds: {args.max_rounds}\n")

    executor, schemas = build_tool_executor(domain)
    print(f"Loaded {len(schemas)} tools for {domain}")

    agent = PromptedAgentRunner(
        tools=schemas, tool_executor=executor,
        model=AGENT_MODEL, api_key=AGENT_API_KEY, base_url=NANOBOT_BASE_URL,
    )
    evaluator = HybridEvaluator(agent=agent, max_rounds=args.max_rounds)

    t0 = time.time()
    report = evaluator.evaluate([task])
    elapsed = time.time() - t0

    result = report.results[0]
    print("\n" + "=" * 60)
    print("  EVALUATION RESULT")
    print("=" * 60)
    print(f"  Task:        {result.task_id} ({result.domain})")
    print(f"  Completed:   {result.completed}")
    print(f"  Rounds:      {result.rounds}")
    print(f"  Satisfaction: {result.satisfaction}")
    print(f"  Status:      {result.status}")
    print(f"  Time:        {elapsed:.0f}s")
    print("=" * 60)
    print("\n--- Full Conversation ---")
    for i, turn in enumerate(result.conversation):
        role = turn["role"].upper()
        content = turn.get("content", "")
        tools = turn.get("tool_results", [])
        print(f"\n[{role}] Round {i//2 + 1}")
        print(content[:600])
        if tools:
            for t in tools:
                tc = str(t.get("content", ""))[:300]
                print(f"  [Tool: {t['name']}] {tc}")


if __name__ == "__main__":
    main()
