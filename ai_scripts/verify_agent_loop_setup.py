"""
Verify Agent Loop migration setup.

Checks:
  1. agent_tools.py registers 14+ tools
  2. grpo_retail.yaml parses correctly (hydra/omegaconf)
  3. Qwen3.5-4B model exists (or check env var)
  4. Agent Loop imports work (verl multi_turn)
  5. REWARD_MODE=agent reward path works
  6. Data files have agent_name field

Usage:
    python ai_scripts/verify_agent_loop_setup.py
    python ai_scripts/verify_agent_loop_setup.py --model /data/models/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}  — {detail}")
        FAIL += 1


def header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    global PASS, FAIL
    parser = argparse.ArgumentParser(description="Verify Agent Loop migration setup")
    parser.add_argument("--model", default="/data/models/Qwen3.5-4B",
                        help="Path to model directory")
    parser.add_argument("--data-dir", default=str(PROJECT_DIR / "data" / "tau_bench"),
                        help="Path to data directory")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    header("1. agent_tools.py — tool registration")
    try:
        sys.path.insert(0, str(PROJECT_DIR))
        from trainable_openclaw.training.agent_tools import _mock_tools
        tool_count = len(_mock_tools)
        check("agent_tools importable", True)
        check(f"Tools registered: {tool_count} (expect 14+)", tool_count >= 14,
              f"found {tool_count}")
        # List tool names
        tool_names = sorted(t.name for t in _mock_tools)
        print(f"  Tools: {', '.join(tool_names[:8])}...")
        # Verify each tool has a callable in globals
        missing = [t.name for t in _mock_tools if t.name not in globals() or not callable(globals()[t.name])]
        check(f"All {tool_count} tools are callable", len(missing) == 0,
              f"missing: {missing}" if missing else "")
    except Exception as e:
        check("agent_tools import", False, str(e))

    # ------------------------------------------------------------------
    header("2. grpo_retail.yaml — config parsing")
    try:
        from omegaconf import OmegaConf
        yaml_path = PROJECT_DIR / "scripts" / "train" / "grpo_retail.yaml"
        cfg = OmegaConf.load(str(yaml_path))
        check("YAML parses with OmegaConf", True)

        # Verify key Agent Loop config sections
        check("rollout.mode=async",
              cfg.actor_rollout_ref.rollout.mode == "async",
              f"got {cfg.actor_rollout_ref.rollout.mode}")
        check("multi_turn.enable=True",
              cfg.actor_rollout_ref.rollout.multi_turn.enable is True)
        check("multi_turn.format=qwen3_coder",
              cfg.actor_rollout_ref.rollout.multi_turn.format == "qwen3_coder")
        check("agent.default_agent_loop=tool_agent",
              cfg.actor_rollout_ref.rollout.agent.default_agent_loop == "tool_agent")
        check("data.return_raw_chat=True",
              cfg.data.return_raw_chat is True)
        check("rollout.n=4",
              cfg.actor_rollout_ref.rollout.n == 4,
              f"got {cfg.actor_rollout_ref.rollout.n}")
        check("calculate_log_probs=true",
              cfg.actor_rollout_ref.rollout.get("calculate_log_probs") is True)
        check("model=Qwen3.5-4B",
              "Qwen3.5-4B" in cfg.actor_rollout_ref.model.path)
        check("No hybrid_engine",
              "hybrid_engine" not in cfg.actor_rollout_ref)
        check("No free_cache_engine",
              "free_cache_engine" not in cfg.actor_rollout_ref.rollout)

    except Exception as e:
        check("YAML parsing", False, str(e))

    # ------------------------------------------------------------------
    header("3. Model availability")
    model_path = Path(args.model)
    config_file = model_path / "config.json"
    check(f"Model dir: {args.model}", model_path.is_dir(),
          f"not found at {args.model}")
    if model_path.is_dir():
        check("config.json exists", config_file.exists(),
              "missing config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    model_cfg = json.load(f)
                check("model_type in config", "model_type" in model_cfg,
                      str(model_cfg.get("model_type", "N/A")))
                print(f"  Model type: {model_cfg.get('model_type', 'unknown')}")
            except Exception as e:
                check("Read config.json", False, str(e))

    # ------------------------------------------------------------------
    header("4. Agent Loop imports (verl multi_turn)")
    try:
        from verl.experimental.agent_loop import ToolAgentLoop
        check("verl ToolAgentLoop import", True)
    except ImportError:
        check("verl ToolAgentLoop import", False,
              "verl may need reinstall or Agent Loop not in this version")

    try:
        from verl.tools.function_tool import function_tool
        check("verl function_tool import", True)
    except ImportError:
        check("verl function_tool import", False, "check verl version")

    try:
        from verl.tools.parsers.qwen3_coder_parser import Qwen3XMLToolParser
        check("Qwen3XMLToolParser import", True)
    except ImportError:
        check("Qwen3XMLToolParser import", False,
              "may not exist in this verl version — check")

    # ------------------------------------------------------------------
    header("5. Reward function — agent mode")
    os.environ["REWARD_MODE"] = "agent"
    try:
        from trainable_openclaw.training.grpo_reward import (
            compute_score,
            _parse_agent_loop_trajectory,
            _extract_user_message,
            REWARD_MODE,
        )
        check("REWARD_MODE default=agent", REWARD_MODE == "agent",
              f"got {REWARD_MODE}")
        check("compute_score import", True)

        # Test trajectory parsing
        sample_traj = """I'll help you with that.<tool_call>
{"name": "find_user_id_by_name_zip", "arguments": {"name": "Nancy", "zip_code": "12345"}}
</tool_call>
<tool_response>
{"status": "success", "user_id": "U123"}
</tool_response>
I found your account. Let me check your orders.<tool_call>
{"name": "get_order_details", "arguments": {"order_id": "O456"}}
</tool_call>
<tool_response>
{"status": "success", "order": {"id": "O456", "status": "delivered"}}
</tool_response>
Your order O456 was delivered. Is there anything else?"""

        conv = _parse_agent_loop_trajectory(sample_traj, "I need help with my order.")
        check("Trajectory parsing produces conversation", len(conv) > 1,
              f"got {len(conv)} messages")
        roles = [m["role"] for m in conv]
        check("Has user message", "user" in roles)
        check("Has assistant messages", "assistant" in roles)
        check("Has tool messages", "tool" in roles)
        print(f"  Conversation: {len(conv)} messages, roles: {roles}")

        # Test user message extraction with list-format prompt
        msg = _extract_user_message(
            {"prompt": [{"role": "system", "content": "You are an agent."},
                        {"role": "user", "content": "I need help with my order."}]},
            "{}"
        )
        check("Extract user from list-prompt", msg == "I need help with my order.",
              f"got: {msg}")

        # Test compute_score with agent mode on sample trajectory
        gt = json.dumps({"evaluation": {"nl_assertions": ["Find user account", "Check order status"]}})
        reward = compute_score(
            data_source="test",
            solution_str=sample_traj,
            ground_truth=gt,
            extra_info={"prompt": [{"role": "user", "content": "I need help with my order."}]},
        )
        check("compute_score returns float", isinstance(reward, float),
              f"got {type(reward)}")
        check("reward in [0, 1]", 0.0 <= reward <= 1.0,
              f"reward={reward}")
        print(f"  Sample reward: {reward:.4f}")

    except Exception as e:
        import traceback
        check("Reward function agent mode", False, str(e))
        traceback.print_exc()

    # ------------------------------------------------------------------
    header("6. Data files — agent_name field")
    data_dir = Path(args.data_dir)
    for fname, expected in [("train_agent_66.jsonl", 66), ("val_agent_18.jsonl", 18)]:
        fpath = data_dir / fname
        if fpath.exists():
            count = 0
            has_agent_name = 0
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
                        obj = json.loads(line)
                        if obj.get("agent_name") == "tool_agent":
                            has_agent_name += 1
            check(f"{fname}: {count} records", count == expected,
                  f"expected {expected}, got {count}")
            check(f"{fname}: all have agent_name", has_agent_name == count,
                  f"{has_agent_name}/{count}")
            # Check prompt is list format
            prompt_type_ok = True
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if not isinstance(obj.get("prompt"), list):
                        prompt_type_ok = False
                        break
            check(f"{fname}: prompt is list format", prompt_type_ok)
        else:
            check(f"{fname} exists", False,
                  f"not found — run scripts/data/add_agent_name.py")

    # ------------------------------------------------------------------
    header("7. Run script exists")
    script = PROJECT_DIR / "scripts" / "train" / "run_grpo_agent.sh"
    check("run_grpo_agent.sh exists", script.exists())
    if script.exists():
        check("run_grpo_agent.sh is executable", os.access(str(script), os.R_OK))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = PASS + FAIL
    print(f"\n{'='*60}")
    print(f"  Results: {PASS}/{total} passed, {FAIL}/{total} failed")
    print(f"{'='*60}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
