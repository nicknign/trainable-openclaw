#!/usr/bin/env python3
"""
Offline conversation log viewer.

Usage::

    python -m trainable_openclaw.logging.viewer [--db PATH] <command> [args]

Commands:
    users       List distinct users
    sessions    List recent sessions  [--user U] [--limit N]
    view ID     Show full conversation for a session
    search KW   Search message content for keyword
    stats       Print aggregate statistics

Examples::

    python -m trainable_openclaw.logging.viewer --db data/conversations.db stats
    python -m trainable_openclaw.logging.viewer sessions --user bob --limit 10
    python -m trainable_openclaw.logging.viewer view abc123def
    python -m trainable_openclaw.logging.viewer search "error message"
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import textwrap
import time


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_ROLE_COLORS = {
    "user": "\033[1;36m",       # bold cyan
    "assistant": "\033[1;32m",  # bold green
    "system": "\033[1;33m",     # bold yellow
    "tool": "\033[1;35m",       # bold magenta
}
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"


def _fmt_time(ts: float | None) -> str:
    if ts is None:
        return "N/A"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _fmt_duration(ms: float | None) -> str:
    if ms is None:
        return ""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _crop(text: str, width: int = 80) -> str:
    text = text.replace("\n", " ")
    return text[:width] + ("..." if len(text) > width else "")


def _color_role(role: str) -> str:
    c = _ROLE_COLORS.get(role, "")
    return f"{c}{role.rjust(9)}{_RESET}"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_users(db: str, _args: argparse.Namespace) -> None:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT user_id, COUNT(*) as sessions, SUM(message_count) as msgs "
        "FROM sessions GROUP BY user_id ORDER BY msgs DESC"
    ).fetchall()
    conn.close()

    if not rows:
        print("No users found.")
        return

    print(f"\n{_BOLD}{'User':<24} {'Sessions':>8} {'Messages':>8}{_RESET}")
    print("-" * 42)
    for r in rows:
        print(f"{r['user_id']:<24} {r['sessions']:>8} {r['msgs']:>8}")
    print()


def cmd_sessions(db: str, args: argparse.Namespace) -> None:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    user = getattr(args, "user", None)
    limit = getattr(args, "limit", 50)
    if user:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (user, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()

    if not rows:
        print("No sessions found.")
        return

    print(f"\n{_BOLD}{'Session ID':<34} {'User':<16} {'Msgs':>4} {'Created':>19}{_RESET}")
    print("-" * 78)
    for r in rows:
        sid = r["id"][:30]
        print(
            f"{sid:<34} {r['user_id']:<16} {r['message_count']:>4} "
            f"{_DIM}{_fmt_time(r['created_at'])}{_RESET}"
        )
    print()


def cmd_view(db: str, args: argparse.Namespace) -> None:
    sid = args.session_id
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    session = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
    if session is None:
        print(f"Session '{sid}' not found.")
        conn.close()
        return

    messages = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
        (sid,),
    ).fetchall()
    conn.close()

    # Header
    print(f"\n{_BOLD}Session: {sid}{_RESET}")
    print(f"  User:  {session['user_id']}")
    print(f"  Model: {session['model'] or 'N/A'}")
    print(f"  Time:  {_fmt_time(session['created_at'])}")
    print(f"  Msgs:  {session['message_count']}")
    print(f"\n{_BOLD}{'─' * 78}{_RESET}\n")

    # Messages
    wrapper = textwrap.TextWrapper(
        width=78, initial_indent="", subsequent_indent=" " * 11, break_long_words=False
    )
    for msg in messages:
        role_tag = _color_role(msg["role"])
        ts = _DIM + _fmt_time(msg["created_at"])[-8:] + _RESET
        extra = []
        if msg["token_count"] is not None:
            extra.append(f"tokens={msg['token_count']}")
        if msg["latency_ms"] is not None:
            extra.append(f"latency={_fmt_duration(msg['latency_ms'])}")
        extra_str = _DIM + "  " + " ".join(extra) + _RESET if extra else ""

        print(f"{role_tag} {ts} {extra_str}")
        print(wrapper.fill(msg["content"]))
        print()
    print(f"{_DIM}{'─' * 78}{_RESET}\n")


def cmd_search(db: str, args: argparse.Namespace) -> None:
    keyword = args.keyword
    limit = getattr(args, "limit", 50)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT m.*, s.user_id FROM messages m "
        "JOIN sessions s ON m.session_id = s.id "
        "WHERE m.content LIKE ? "
        "ORDER BY m.created_at DESC LIMIT ?",
        (f"%{keyword}%", limit),
    ).fetchall()
    conn.close()

    if not rows:
        print(f"No messages matching '{keyword}'.")
        return

    print(f"\n{_BOLD}{len(rows)} results for '{keyword}':{_RESET}\n")
    wrapper = textwrap.TextWrapper(width=78, initial_indent="  ", subsequent_indent="  ", break_long_words=False)
    for r in rows:
        role_tag = _color_role(r["role"])
        sid = r["session_id"][:12]
        ts = _fmt_time(r["created_at"])
        print(f"{role_tag}  {_DIM}session={sid}  user={r['user_id']}  {ts}{_RESET}")
        print(wrapper.fill(_crop(r["content"], 200)))
        print()


def cmd_stats(db: str, _args: argparse.Namespace) -> None:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    total_users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM sessions").fetchone()[0]

    role_rows = conn.execute(
        "SELECT role, COUNT(*) as cnt FROM messages GROUP BY role ORDER BY cnt DESC"
    ).fetchall()

    user_rows = conn.execute(
        "SELECT user_id, COUNT(*) as sessions, "
        "SUM(message_count) as msgs "
        "FROM sessions GROUP BY user_id ORDER BY msgs DESC LIMIT 10"
    ).fetchall()

    first = conn.execute("SELECT MIN(created_at) FROM messages").fetchone()[0]
    last = conn.execute("SELECT MAX(created_at) FROM messages").fetchone()[0]
    conn.close()

    print(f"\n{_BOLD}Conversation Store Statistics{_RESET}")
    print(f"  Database: {db}")
    print(f"  File size: {os.path.getsize(db) / 1024:.1f} KB" if os.path.exists(db) else "  (no database)")
    print(f"\n  Total sessions: {total_sessions}")
    print(f"  Total messages: {total_messages}")
    print(f"  Unique users:   {total_users}")

    print(f"\n  {_BOLD}Role Distribution:{_RESET}")
    for r in role_rows:
        print(f"    {_color_role(r['role'])}: {r['cnt']}")

    print(f"\n  {_BOLD}Top Users:{_RESET}")
    if user_rows:
        print(f"  {'User':<20} {'Sessions':>8} {'Messages':>8}")
        print(f"  {'-'*36}")
        for r in user_rows:
            print(f"  {r['user_id']:<20} {r['sessions']:>8} {r['msgs']:>8}")
    else:
        print("    (no data)")

    print(f"\n  Time range:")
    print(f"    First: {_fmt_time(first)}")
    print(f"    Last:  {_fmt_time(last)}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Offline conversation log viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__).strip(),
    )
    parser.add_argument("--db", default="data/conversations.db", help="Path to SQLite database")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("users", help="List distinct users")

    sp_sessions = sub.add_parser("sessions", help="List recent sessions")
    sp_sessions.add_argument("--user", help="Filter by user_id")
    sp_sessions.add_argument("--limit", type=int, default=50)

    sp_view = sub.add_parser("view", help="View a session's full conversation")
    sp_view.add_argument("session_id", help="Session UUID")

    sp_search = sub.add_parser("search", help="Search message content")
    sp_search.add_argument("keyword", help="Search keyword")
    sp_search.add_argument("--limit", type=int, default=50)

    sub.add_parser("stats", help="Print aggregate statistics")

    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        return

    handler = {
        "users": cmd_users,
        "sessions": cmd_sessions,
        "view": cmd_view,
        "search": cmd_search,
        "stats": cmd_stats,
    }
    handler[args.command](args.db, args)


if __name__ == "__main__":
    main()
