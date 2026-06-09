"""
T2: Nanobot-based rollout generator for training data production.

When serve_ppo enters training mode (vLLM sleeping), this module uses
nanobot subagents backed by an external LLM (DeepSeek) to generate
coding responses for training prompts. The nanobot agent has access to
filesystem + shell tools, producing higher-quality, self-verified code.

Usage inside serve_ppo's training loop::

    gen = NanobotRolloutGenerator(
        api_key="sk-...",
        model="deepseek-v4-flash",
    )
    responses = await gen.generate_rollouts(
        prompts=["写一个快速排序函数", "实现二分查找"],
        n=4,  # rollouts per prompt
    )
    # responses: [[r1, r2, r3, r4], [r1, r2, r3, r4]]
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_NANOBOT_PROMPT_TEMPLATE = """You are an expert programmer. Write a complete, correct solution to the following task.

Task: {task}

Requirements:
1. Write clean, well-documented code
2. Include error handling where appropriate
3. Test your solution by writing and running a test case (use shell tool)
4. If the test fails, fix the code and re-test
5. Return ONLY the final working code in a ```python (or appropriate language) block

Do NOT include explanations or markdown outside the code block. Just the working code."""


class NanobotRolloutGenerator:
    """Generates training rollouts using nanobot subagents + external LLM.

    Each prompt gets ``n`` independent nanobot agents spawned to solve
    the task. Each agent has filesystem + shell tools for self-testing.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        max_concurrent: int = 4,
        agent_timeout: float = 120.0,
        workspace: str | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.agent_timeout = agent_timeout
        self.workspace = Path(workspace or "/tmp/nanobot_rollout").expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._nanobot_path = Path(__file__).parent.parent.parent / "nanobot-0.2.1"

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _build_config(self) -> dict[str, Any]:
        """Build a nanobot config for the rollout agent."""
        return {
            "agents": {
                "defaults": {
                    "workspace": str(self.workspace),
                    "model": f"deepseek/{self.model}",
                    "provider": "deepseek",
                    "maxTokens": 4096,
                    "contextWindowTokens": 65536,
                    "temperature": 0.8,
                    "maxToolIterations": 20,
                    "timezone": "Asia/Shanghai",
                    "botName": "rollout-agent",
                    "disabledSkills": [
                        "image-generation",
                        "long-goal",
                        "cron",
                        "summarize",
                        "skill-creator",
                    ],
                },
            },
            "providers": {
                "deepseek": {
                    "apiKey": self.api_key,
                    "apiBase": self.base_url,
                },
            },
        }

    # ------------------------------------------------------------------
    # Single rollout
    # ------------------------------------------------------------------

    def _extract_code(self, text: str) -> str:
        """Extract code block from agent output."""
        import re

        # Try ```python / ``` code blocks first
        pattern = r"```(?:python|py|javascript|js|java|go|rust|cpp|c\+\+)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "\n\n".join(m.strip() for m in matches if m.strip())

        # Fallback: return the whole text
        return text.strip()

    async def _single_rollout(self, prompt: str, idx: int) -> str:
        """Run one nanobot agent for one prompt, return extracted code."""
        import json
        import tempfile
        import uuid

        config_dir = Path(tempfile.gettempdir()) / f"nanobot_rollout_{uuid.uuid4().hex[:8]}"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
        cfg = self._build_config()
        cfg["agents"]["defaults"]["workspace"] = str(config_dir / "workspace")
        Path(cfg["agents"]["defaults"]["workspace"]).mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False)

        message = _NANOBOT_PROMPT_TEMPLATE.format(task=prompt)

        try:
            sys.path.insert(0, str(self._nanobot_path))
            from nanobot.nanobot import Nanobot

            bot = Nanobot.from_config(str(config_path))
            result = await asyncio.wait_for(
                bot.run(message, session_key=f"rollout:{uuid.uuid4().hex[:8]}"),
                timeout=self.agent_timeout,
            )
            code = self._extract_code(result.content)
            logger.debug("Rollout %d: %d chars, %d tools used", idx, len(code), len(result.tools_used))
            return code
        except asyncio.TimeoutError:
            logger.warning("Rollout %d timed out after %.0fs", idx, self.agent_timeout)
            return f"# TIMEOUT after {self.agent_timeout}s\n# Prompt: {prompt}"
        except Exception:
            logger.exception("Rollout %d failed", idx)
            return f"# ERROR generating response\n# Prompt: {prompt}"
        finally:
            sys.path.remove(str(self._nanobot_path)) if str(self._nanobot_path) in sys.path else None
            import shutil
            shutil.rmtree(config_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    async def generate_rollouts(
        self,
        prompts: list[str],
        n: int = 4,
    ) -> list[list[str]]:
        """Generate ``n`` rollouts per prompt using nanobot agents.

        Returns a list of lists: ``[[r1_for_p1, r2_for_p1, ...], [r1_for_p2, ...]]``

        Concurrency is limited by ``max_concurrent``.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _bounded(prompt: str, idx: int) -> str:
            async with semaphore:
                return await self._single_rollout(prompt, idx)

        all_responses: list[list[str]] = []
        for pi, prompt in enumerate(prompts):
            tasks = [_bounded(prompt, pi * n + ri) for ri in range(n)]
            responses = await asyncio.gather(*tasks)
            all_responses.append(list(responses))
            logger.info(
                "Rollout batch %d/%d: prompt='%s...', %d responses",
                pi + 1, len(prompts), prompt[:60], len(responses),
            )

        return all_responses

    # ------------------------------------------------------------------
    # Simple generation (no agent tools)
    # ------------------------------------------------------------------

    async def generate_simple(
        self,
        prompts: list[str],
        n: int = 4,
        temperature: float = 0.8,
    ) -> list[list[str]]:
        """Generate rollouts using plain LLM calls (no agent tools).

        Faster than full agent rollout — suitable for quick data augmentation.
        """
        import httpx

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _one(prompt: str) -> str:
            async with semaphore:
                try:
                    async with httpx.AsyncClient(timeout=self.agent_timeout) as client:
                        resp = await client.post(
                            url,
                            headers=headers,
                            json={
                                "model": self.model,
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": "You are an expert programmer. Write complete, correct code. "
                                        "Return ONLY the code in a ```python block, no explanations.",
                                    },
                                    {"role": "user", "content": prompt},
                                ],
                                "temperature": temperature,
                                "max_tokens": 4096,
                            },
                        )
                        data = resp.json()
                        content = data["choices"][0]["message"]["content"]
                        return self._extract_code(content)
                except Exception:
                    logger.exception("Simple rollout failed for prompt")
                    return f"# ERROR\n# Prompt: {prompt[:100]}"

        all_responses: list[list[str]] = []
        for pi, prompt in enumerate(prompts):
            tasks = [_one(prompt) for _ in range(n)]
            responses = await asyncio.gather(*tasks)
            all_responses.append(list(responses))

        return all_responses

    # ------------------------------------------------------------------
    # Integration with serve_ppo training pool
    # ------------------------------------------------------------------

    def make_training_pool(
        self,
        prompts: list[str],
        responses: list[list[str]],
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert rollout outputs to the training pool format expected by serve_ppo.

        Each entry in the returned list can be appended to ``_training_pool``.
        """
        import json

        pool = []
        for pi, (prompt, variants) in enumerate(zip(prompts, responses)):
            prompt_ids = [ord(c) for c in prompt]  # placeholder token IDs

            for vi, response in enumerate(variants):
                entry = {
                    "prompt_ids": prompt_ids,
                    "prompt_text": prompt,
                    "ground_truth": None,
                    "train_count": 0,
                    "source": "nanobot_rollout",
                    "类别": categories[pi] if categories else "",
                    "metadata": {
                        "prompt_text": prompt,
                        "response_text": response,
                        "variant": vi,
                        "generator": "nanobot",
                        "model": self.model,
                    },
                }
                pool.append(entry)

        logger.info(
            "Built training pool: %d prompts × %d variants = %d entries",
            len(prompts), len(responses[0]) if responses else 0, len(pool),
        )
        return pool


# ---------------------------------------------------------------------------
# Convenience: create from environment
# ---------------------------------------------------------------------------

def create_rollout_generator(
    api_key: str = "",
    model: str = "deepseek-v4-flash",
    base_url: str = "https://api.deepseek.com",
    **kwargs: Any,
) -> NanobotRolloutGenerator:
    import os

    return NanobotRolloutGenerator(
        api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
        model=model,
        base_url=base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        **kwargs,
    )
