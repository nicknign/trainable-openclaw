"""
T1: Nanobot integration — configure and launch nanobot backed by serve_ppo.

Usage::

    adapter = NanobotAdapter(api_base="http://localhost:8000/v1", model="qwen3-4b")
    config_path = adapter.write_config()
    # Then start nanobot: `nanobot gateway --config {config_path}`
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_TEMPLATE: dict[str, Any] = {
    "agents": {
        "defaults": {
            "workspace": "~/.nanobot/workspace",
            "model": "serve_ppo/qwen3-4b",
            "provider": "custom",
            "maxTokens": 4096,
            "contextWindowTokens": 8192,
            "temperature": 0.7,
            "maxToolIterations": 50,
            "timezone": "Asia/Shanghai",
            "botName": "trainable-claw",
            "sessionTtlMinutes": 60,
            "disabledSkills": ["image-generation"],
        },
    },
    "providers": {
        "custom": {
            "apiBase": "http://localhost:8000/v1",
            "apiKey": "no-key",
        },
    },
    "gateway": {
        "host": "127.0.0.1",
        "port": 18790,
    },
    "api": {
        "host": "127.0.0.1",
        "port": 8900,
    },
}


class NanobotAdapter:
    """Bridge between trainable-openclaw and nanobot.

    Generates a nanobot config file pointing the ``custom`` provider at
    our serve_ppo endpoint, and provides helpers to start nanobot
    programmatically or validate the connection.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model: str = "qwen3-4b",
        workspace: str | None = None,
        config_dir: str | None = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.workspace = Path(workspace or "~/.nanobot/workspace").expanduser()
        self.config_dir = Path(config_dir or "~/.nanobot").expanduser()
        self.config_path = self.config_dir / "config.json"

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    def build_config(self) -> dict[str, Any]:
        """Return the full nanobot config dict."""
        cfg = json.loads(json.dumps(_DEFAULT_CONFIG_TEMPLATE))
        cfg["agents"]["defaults"]["model"] = self.model
        cfg["agents"]["defaults"]["workspace"] = str(self.workspace)
        cfg["providers"]["custom"]["apiBase"] = self.api_base
        return cfg

    def write_config(self) -> Path:
        """Write the nanobot config file and return its path."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.build_config()
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        logger.info("nanobot config written to %s", self.config_path)
        return self.config_path

    # ------------------------------------------------------------------
    # Programmatic launch
    # ------------------------------------------------------------------

    async def create_bot(self) -> Any:
        """Create a Nanobot instance connected to our serve_ppo backend.

        Returns a ``Nanobot`` facade that can be used with ``bot.run(message)``.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nanobot-0.2.1"))

        from nanobot.nanobot import Nanobot

        self.write_config()
        bot = Nanobot.from_config(str(self.config_path))
        logger.info("nanobot instance created — model=%s, api=%s", self.model, self.api_base)
        return bot

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    async def test_connection(self) -> bool:
        """Send a ping message through nanobot and check the response."""
        try:
            bot = await self.create_bot()
            result = await bot.run("Hello! Reply with just 'OK' and nothing else.")
            success = "OK" in result.content
            logger.info("nanobot connection test: %s", "PASS" if success else "FAIL")
            return success
        except Exception:
            logger.exception("nanobot connection test failed")
            return False

    # ------------------------------------------------------------------
    # serve_ppo health check
    # ------------------------------------------------------------------

    async def check_serve_ppo(self) -> bool:
        """Verify serve_ppo is reachable at api_base."""
        import httpx

        health_url = f"{self.api_base}/health"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(health_url)
                ok = resp.status_code == 200
                logger.info("serve_ppo health at %s: %s", health_url, "OK" if ok else f"HTTP {resp.status_code}")
                return ok
        except Exception:
            logger.warning("serve_ppo not reachable at %s", health_url)
            return False
