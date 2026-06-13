#!/usr/bin/env python3
"""
Inter-agent message system for Claude Code subagents.

File-based, no daemon needed. Agents read/write JSON messages to
.claude/messages/{agent-name}/inbox/ and .../sent/.

Usage:
  python .claude/agent_message.py send --to AGENT --type TYPE --subject S [--body B] [--context JSON]
  python .claude/agent_message.py check [--agent AGENT] [--unread-only]
  python .claude/agent_message.py read MSG_ID [--agent AGENT]
  python .claude/agent_message.py reply MSG_ID --body B [--agent AGENT]
  python .claude/agent_message.py mark-read MSG_ID [--agent AGENT]
  python .claude/agent_message.py list-agents
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MSG_DIR = os.path.join(PROJECT_DIR, ".claude", "messages")

VALID_AGENTS = {
    "disciplined-coder",
    "e2e-code-tester",
    "research-scout",
    "academic-content-writer",
}

VALID_TYPES = {"task_request", "status_update", "question", "handoff", "reply"}


def _ensure_dirs(agent):
    for sub in ("inbox", "sent"):
        os.makedirs(os.path.join(MSG_DIR, agent, sub), exist_ok=True)


def _msg_path(agent, msg_id, folder="inbox"):
    return os.path.join(MSG_DIR, agent, folder, f"{msg_id}.json")


def cmd_send(args):
    if args.to not in VALID_AGENTS:
        print(f"ERROR: unknown agent '{args.to}'. Valid: {', '.join(sorted(VALID_AGENTS))}")
        return 1
    if args.type not in VALID_TYPES:
        print(f"ERROR: unknown type '{args.type}'. Valid: {', '.join(sorted(VALID_TYPES))}")
        return 1

    _ensure_dirs(args.to)

    msg_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    context = {}
    if args.context:
        try:
            context = json.loads(args.context)
        except json.JSONDecodeError as e:
            print(f"ERROR: invalid context JSON: {e}")
            return 1

    msg = {
        "id": msg_id,
        "from": args.from_agent or os.environ.get("CLAUDE_CODE_AGENT_NAME", "unknown"),
        "to": args.to,
        "type": args.type,
        "subject": args.subject,
        "body": args.body or "",
        "context": context,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "unread",
    }

    path = _msg_path(args.to, msg_id, "inbox")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(msg, f, ensure_ascii=False, indent=2)

    print(f"Message sent: {msg_id}")
    print(f"  To: {args.to} ({args.type})")
    print(f"  Subject: {args.subject}")
    return 0


def cmd_check(args):
    agent = args.agent or "disciplined-coder"
    if agent == "all":
        for a in sorted(VALID_AGENTS):
            _list_inbox(a, args.unread_only)
            print()
        return 0
    if agent not in VALID_AGENTS:
        print(f"ERROR: unknown agent '{agent}'")
        return 1
    _list_inbox(agent, args.unread_only)
    return 0


def _list_inbox(agent, unread_only):
    inbox_dir = os.path.join(MSG_DIR, agent, "inbox")
    if not os.path.isdir(inbox_dir):
        print(f"{agent}/inbox: (empty)")
        return
    files = sorted(
        [f for f in os.listdir(inbox_dir) if f.endswith(".json")],
        reverse=True,
    )
    if not files:
        print(f"{agent}/inbox: (empty)")
        return

    print(f"{agent}/inbox ({len(files)} messages):")
    for fn in files:
        path = os.path.join(inbox_dir, fn)
        try:
            with open(path, encoding="utf-8") as f:
                msg = json.load(f)
        except Exception:
            continue
        if unread_only and msg.get("status") != "unread":
            continue
        status = msg.get("status", "?")
        ts = msg.get("timestamp", "?")[:19]
        print(f"  [{status:7s}] {msg['id']}")
        print(f"             From: {msg.get('from', '?')} | Type: {msg.get('type', '?')} | {ts}")
        print(f"             Subject: {msg.get('subject', '(no subject)')}")
        body = msg.get("body", "")
        if body:
            preview = body[:120] + ("..." if len(body) > 120 else "")
            print(f"             Body: {preview}")


def cmd_read(args):
    agent = args.agent or "disciplined-coder"
    path = _msg_path(agent, args.msg_id, "inbox")
    if not os.path.exists(path):
        print(f"ERROR: message {args.msg_id} not found in {agent}/inbox")
        return 1
    with open(path, encoding="utf-8") as f:
        msg = json.load(f)
    print(json.dumps(msg, ensure_ascii=False, indent=2))
    return 0


def cmd_mark_read(args):
    agent = args.agent or "disciplined-coder"
    path = _msg_path(agent, args.msg_id, "inbox")
    if not os.path.exists(path):
        print(f"ERROR: message {args.msg_id} not found in {agent}/inbox")
        return 1
    with open(path, encoding="utf-8") as f:
        msg = json.load(f)
    msg["status"] = "read"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(msg, f, ensure_ascii=False, indent=2)
    print(f"Marked {args.msg_id} as read")
    return 0


def cmd_reply(args):
    agent = args.agent or "disciplined-coder"
    path = _msg_path(agent, args.msg_id, "inbox")
    if not os.path.exists(path):
        print(f"ERROR: original message {args.msg_id} not found in {agent}/inbox")
        return 1
    with open(path, encoding="utf-8") as f:
        original = json.load(f)

    # Mark original as read
    original["status"] = "replied"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(original, f, ensure_ascii=False, indent=2)

    # Create reply in original sender's inbox
    reply_to = original.get("from", "disciplined-coder")
    if reply_to not in VALID_AGENTS:
        print(f"ERROR: original sender '{reply_to}' is not a valid agent")
        return 1

    _ensure_dirs(reply_to)
    reply_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"

    reply = {
        "id": reply_id,
        "from": agent,
        "to": reply_to,
        "type": "reply",
        "subject": f"Re: {original.get('subject', '(no subject)')}",
        "body": args.body,
        "context": {"in_reply_to": args.msg_id},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "unread",
    }

    reply_path = _msg_path(reply_to, reply_id, "inbox")
    with open(reply_path, "w", encoding="utf-8") as f:
        json.dump(reply, f, ensure_ascii=False, indent=2)

    print(f"Reply sent: {reply_id}")
    print(f"  To: {reply_to}")
    return 0


def cmd_list_agents(_args):
    print("Available agents:")
    for a in sorted(VALID_AGENTS):
        inbox_dir = os.path.join(MSG_DIR, a, "inbox")
        unread = 0
        if os.path.isdir(inbox_dir):
            for fn in os.listdir(inbox_dir):
                if fn.endswith(".json"):
                    try:
                        with open(os.path.join(inbox_dir, fn), encoding="utf-8") as f:
                            m = json.load(f)
                        if m.get("status") == "unread":
                            unread += 1
                    except Exception:
                        pass
        print(f"  {a} ({unread} unread)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Inter-agent message system")
    sub = parser.add_subparsers(dest="command")

    p_send = sub.add_parser("send", help="Send a message to another agent")
    p_send.add_argument("--to", required=True)
    p_send.add_argument("--type", required=True, choices=sorted(VALID_TYPES))
    p_send.add_argument("--subject", required=True)
    p_send.add_argument("--body", default="")
    p_send.add_argument("--context", default="{}")
    p_send.add_argument("--from-agent", default=None)

    p_check = sub.add_parser("check", help="Check inbox for messages")
    p_check.add_argument("--agent", default="disciplined-coder")
    p_check.add_argument("--unread-only", action="store_true")

    p_read = sub.add_parser("read", help="Read a specific message")
    p_read.add_argument("msg_id")
    p_read.add_argument("--agent", default="disciplined-coder")

    p_mark = sub.add_parser("mark-read", help="Mark a message as read")
    p_mark.add_argument("msg_id")
    p_mark.add_argument("--agent", default="disciplined-coder")

    p_reply = sub.add_parser("reply", help="Reply to a message")
    p_reply.add_argument("msg_id")
    p_reply.add_argument("--body", required=True)
    p_reply.add_argument("--agent", default="disciplined-coder")

    sub.add_parser("list-agents", help="List all agents and unread counts")

    args = parser.parse_args()
    if args.command == "send":
        return cmd_send(args)
    elif args.command == "check":
        return cmd_check(args)
    elif args.command == "read":
        return cmd_read(args)
    elif args.command == "mark-read":
        return cmd_mark_read(args)
    elif args.command == "reply":
        return cmd_reply(args)
    elif args.command == "list-agents":
        return cmd_list_agents(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
