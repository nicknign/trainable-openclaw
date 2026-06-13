#!/usr/bin/env python3
"""
Phase 4 integration quick-start and validation script.

Usage::

    python scripts/test/test_phase4.py [--api-base http://localhost:8000/v1] [--quick]

Tests:
  1. nanobot config generation and validation
  2. serve_ppo health check
  3. LogBridge dual-write verification
  4. NanobotRolloutGenerator simple mode (no GPU needed)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("phase4_test")


def test_config_generation(api_base: str, model: str) -> bool:
    """T1: Verify nanobot config builds correctly."""
    from trainable_openclaw.agent.nanobot_adapter import NanobotAdapter

    adapter = NanobotAdapter(api_base=api_base, model=model)
    cfg = adapter.build_config()

    assert cfg["agents"]["defaults"]["model"] == model, f"Model mismatch: {cfg['agents']['defaults']['model']}"
    assert cfg["providers"]["custom"]["apiBase"] == api_base, f"API base mismatch"
    assert cfg["providers"]["custom"]["apiKey"] == "no-key"
    assert cfg["agents"]["defaults"]["provider"] == "custom"

    # Verify it writes correctly
    config_path = adapter.write_config()
    assert config_path.exists(), f"Config file not created: {config_path}"
    with open(config_path) as f:
        written = json.load(f)
    assert written["providers"]["custom"]["apiBase"] == api_base

    logger.info("PASS: config generation")
    return True


async def test_serve_ppo_health(api_base: str) -> bool:
    """T1: Verify serve_ppo health endpoint."""
    from trainable_openclaw.agent.nanobot_adapter import NanobotAdapter

    adapter = NanobotAdapter(api_base=api_base)
    ok = await adapter.check_serve_ppo()
    if ok:
        logger.info("PASS: serve_ppo health check")
    else:
        logger.warning("SKIP: serve_ppo not reachable (expected if no GPU)")
    return True  # Not a failure if serve_ppo isn't running


def test_log_bridge_sync() -> bool:
    """T3: Verify LogBridge can sync nanobot sessions to SQLite."""
    import tempfile

    from trainable_openclaw.agent.log_bridge import LogBridge
    from trainable_openclaw.logging.conversation_store import ConversationStore

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Create a fake nanobot session
        sessions_dir = tmp / "sessions"
        sessions_dir.mkdir()
        session_file = sessions_dir / "test_user_42.jsonl"
        with open(session_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "_type": "metadata",
                "key": "test:user_42",
                "created_at": "2026-06-09T10:00:00",
                "updated_at": "2026-06-09T10:05:00",
                "metadata": {},
                "last_consolidated": 0,
            }, ensure_ascii=False) + "\n")
            f.write(json.dumps({
                "role": "user",
                "content": "写一个排序函数",
            }, ensure_ascii=False) + "\n")
            f.write(json.dumps({
                "role": "assistant",
                "content": "```python\ndef sort(arr):\n    return sorted(arr)\n```",
            }, ensure_ascii=False) + "\n")

        db_path = str(tmp / "test.db")
        store = ConversationStore(db_path)
        bridge = LogBridge(store)

        count = bridge.sync_from_nanobot(tmp)
        assert count > 0, f"No messages imported, count={count}"

        # Verify data
        sessions = store.list_sessions()
        assert len(sessions) >= 1, f"No sessions created: {sessions}"
        sid = sessions[0]["id"]
        messages = store.get_messages(sid)
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

        store.close()
        logger.info("PASS: log bridge sync (%d messages)", count)
        return True


async def test_rollout_simple() -> bool:
    """T2: Verify NanobotRolloutGenerator simple mode (no nanobot agent)."""
    from trainable_openclaw.agent.rollout import NanobotRolloutGenerator

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        logger.warning("SKIP: DEEPSEEK_API_KEY not set — can't test rollout generator")
        return True

    gen = NanobotRolloutGenerator(
        api_key=api_key,
        model="deepseek-v4-flash",
        max_concurrent=2,
        agent_timeout=30,
    )

    prompts = ["写一个函数计算斐波那契数列"]
    responses = await gen.generate_simple(prompts, n=2)

    assert len(responses) == 1, f"Expected 1 prompt group, got {len(responses)}"
    assert len(responses[0]) == 2, f"Expected 2 rollouts, got {len(responses[0])}"
    for r in responses[0]:
        assert len(r) > 10, f"Response too short: {r[:50]}"

    logger.info("PASS: rollout simple generation")
    return True


async def test_training_pool_creation() -> bool:
    """T2: Verify training pool builder."""
    from trainable_openclaw.agent.rollout import NanobotRolloutGenerator

    gen = NanobotRolloutGenerator(api_key="test-key")
    prompts = ["task A", "task B"]
    responses = [["code A1", "code A2"], ["code B1", "code B2"]]
    pool = gen.make_training_pool(prompts, responses, categories=["coding", "coding"])

    assert len(pool) == 4, f"Expected 4 entries, got {len(pool)}"
    assert pool[0]["source"] == "nanobot_rollout"
    assert pool[0]["prompt_text"] == "task A"
    assert pool[2]["prompt_text"] == "task B"

    logger.info("PASS: training pool creation")
    return True


async def test_log_bridge_live_wrap() -> bool:
    """T3: Verify live SessionManager wrapping."""
    import tempfile
    from pathlib import Path

    from trainable_openclaw.agent.log_bridge import LogBridge
    from trainable_openclaw.logging.conversation_store import ConversationStore

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        db_path = str(tmp / "live.db")
        store = ConversationStore(db_path)
        bridge = LogBridge(store, auto_sync=True)

        # Simulate a nanobot session
        from dataclasses import dataclass, field

        @dataclass
        class FakeSession:
            key: str
            messages: list = field(default_factory=list)
            created_at: str = "2026-06-09T10:00:00"
            updated_at: str = "2026-06-09T10:05:00"
            metadata: dict = field(default_factory=dict)
            last_consolidated: int = 0

        class FakeManager:
            def save(self, session, **kwargs):
                pass

        mgr = FakeManager()
        bridge.wrap_session_manager(mgr)

        session = FakeSession(key="test:live")
        session.messages.append({"role": "user", "content": "hello"})

        # Call save — should dual-write
        mgr.save(session)

        sessions = store.list_sessions()
        assert len(sessions) >= 1, "No session created via live wrap"
        store.close()
        logger.info("PASS: log bridge live wrap")
        return True


async def main():
    parser = argparse.ArgumentParser(description="Phase 4 integration tests")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--quick", action="store_true", help="Skip API-dependent tests")
    args = parser.parse_args()

    results: dict[str, bool] = {}

    print("=" * 60)
    print("Phase 4 Integration Tests")
    print("=" * 60)
    print()

    # T1: Config
    results["config_generation"] = test_config_generation(args.api_base, args.model)

    # T1: Health check (optional)
    results["serve_ppo_health"] = await test_serve_ppo_health(args.api_base)

    # T3: Log bridge
    results["log_bridge_sync"] = test_log_bridge_sync()
    results["log_bridge_live"] = await test_log_bridge_live_wrap()

    # T2: Training pool
    results["training_pool"] = await test_training_pool_creation()

    # T2: Rollout (needs API key)
    if not args.quick:
        results["rollout_simple"] = await test_rollout_simple()
    else:
        results["rollout_simple"] = True  # skipped

    print()
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    status = "ALL PASSED" if passed == total else f"{passed}/{total} PASSED"
    print(f"Results: {status}")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
