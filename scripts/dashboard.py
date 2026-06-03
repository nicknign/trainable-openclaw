#!/usr/bin/env python
"""
Phase 3 C2: Training Dashboard

Simple Streamlit web UI for monitoring the trainable-openclaw system.

Usage:
    streamlit run scripts/dashboard.py
    streamlit run scripts/dashboard.py -- --db data/conversations.db --rubrics data/rubrics_category.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="OpenClaw Dashboard",
    page_icon="",
    layout="wide",
)

# --- Config ---

parser = argparse.ArgumentParser()
parser.add_argument("--db", default="data/conversations.db", help="Conversation store path")
parser.add_argument("--rubrics", default="data/rubrics_category.json", help="Rubrics JSON path")
parser.add_argument("--results", default="data/pipeline_results.json", help="Pipeline results JSON")
parser.add_argument("--checkpoints", default="checkpoints", help="Checkpoint directory")
parser.add_argument("--server", default="http://localhost:8000/v1", help="Model server URL")
parser.add_argument("--refresh", type=int, default=30, help="Auto-refresh interval (seconds)")
try:
    args = parser.parse_args()
except SystemExit:
    args = parser.parse_args([])


# --- Data loading helpers ---

@st.cache_data(ttl=30)
def load_rubric_stats(rubrics_path: str) -> dict | None:
    """Load rubric file and compute stats."""
    p = Path(rubrics_path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    rubric_items = data.get("rubrics", data) if isinstance(data, dict) else data
    if not isinstance(rubric_items, list):
        return None

    active = [r for r in rubric_items if r.get("状态") == "活跃"]
    archived = [r for r in rubric_items if r.get("状态") != "活跃"]

    by_group = {}
    for r in active:
        group = r.get("类别组", "未分组")
        by_group[group] = by_group.get(group, 0) + 1

    return {
        "active_count": len(active),
        "archived_count": len(archived),
        "total_count": len(rubric_items),
        "by_group": by_group,
        "rubrics": active,
    }


@st.cache_data(ttl=30)
def load_store_stats(db_path: str) -> dict:
    """Load conversation store statistics."""
    p = Path(db_path)
    if not p.exists():
        return {"sessions": 0, "messages": 0, "users": 0}
    try:
        from trainable_openclaw.logging.conversation_store import ConversationStore
        store = ConversationStore(db_path)
        stats = store.get_statistics()
        store.close()
        return stats
    except Exception:
        return {"sessions": 0, "messages": 0, "users": 0}


@st.cache_data(ttl=30)
def load_pipeline_results(result_path: str) -> dict | None:
    """Load pipeline JSON results."""
    p = Path(result_path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_checkpoint_info(checkpoint_dir: str) -> list[dict]:
    """Scan checkpoint directory."""
    p = Path(checkpoint_dir)
    if not p.exists():
        return []
    info = []
    for d in sorted(p.iterdir(), reverse=True):
        if d.is_dir():
            info.append({
                "name": d.name,
                "mtime": d.stat().st_mtime,
            })
    return info


def get_server_status(server_url: str) -> dict | None:
    """Query /v1/health endpoint."""
    try:
        import urllib.request
        health_url = server_url.rstrip("/") + "/health"
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


# --- Render functions ---

def render_header():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("Trainable OpenClaw Dashboard")
    with col2:
        st.caption(f"Auto-refresh: {args.refresh}s")


def render_mode_status():
    st.subheader("Server Status")
    status = get_server_status(args.server)

    if status is None:
        st.error("Offline — server not reachable")
        return

    mode = status.get("mode", "unknown")
    uptime = status.get("uptime_seconds", 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        if mode == "serving":
            st.success("Serving")
        elif mode == "training":
            st.warning("Training in progress")
        else:
            st.info(mode)

    with col2:
        hours = int(uptime // 3600)
        mins = int((uptime % 3600) // 60)
        st.metric("Uptime", f"{hours}h {mins}m")

    with col3:
        gpu = status.get("gpu_count", "N/A")
        active = status.get("active_requests", 0)
        st.metric("Active Requests", active)


def render_training_progress():
    st.subheader("Training Progress")

    checkpoints = load_checkpoint_info(args.checkpoints)
    results = load_pipeline_results(args.results)

    if checkpoints:
        latest = checkpoints[0]
        st.metric("Checkpoints", len(checkpoints))
        st.caption(f"Latest: {latest['name']}")

    if results and "轮次结果" in results:
        rounds = results["轮次结果"]
        if rounds:
            # Rewards table
            st.metric("Training Rounds", len(rounds))

            # Delta trend
            deltas = [r.get("Δ纠错率", 0) for r in rounds if r.get("Δ纠错率") is not None]
            if deltas:
                st.line_chart({"Δ纠错率": deltas})
    else:
        st.caption("No training results yet — check back after pipeline run")


def render_evaluation_results():
    st.subheader("Evaluation Results")

    results = load_pipeline_results(args.results)

    if not results or "轮次结果" not in results:
        st.caption("No evaluation results yet")
        return

    rounds = results["轮次结果"]
    if not rounds:
        st.caption("No round data")
        return

    last = rounds[-1]
    col1, col2, col3 = st.columns(3)

    with col1:
        pre = last.get("pre_纠错率")
        st.metric("Pre 纠错率", f"{pre:.3f}" if pre is not None else "N/A")

    with col2:
        post = last.get("post_纠错率")
        st.metric("Post 纠错率", f"{post:.3f}" if post is not None else "N/A")

    with col3:
        delta = last.get("Δ纠错率")
        if delta is not None:
            delta_str = f"{delta:+.3f}"
            st.metric("Δ纠错率", delta_str, delta=delta_str)
        else:
            st.metric("Δ纠错率", "N/A")


def render_rubric_status():
    st.subheader("Rubric Status")

    rubric_stats = load_rubric_stats(args.rubrics)

    if not rubric_stats:
        st.caption("No rubric data")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active", rubric_stats["active_count"])
    with col2:
        st.metric("Archived", rubric_stats["archived_count"])
    with col3:
        st.metric("Total", rubric_stats["total_count"])

    # Per-group breakdown
    if rubric_stats["by_group"]:
        st.caption("By category group:")
        for group, count in rubric_stats["by_group"].items():
            st.text(f"  {group}: {count}")

    # Active rubric names
    if rubric_stats["rubrics"]:
        with st.expander(f"Active Rubrics ({rubric_stats['active_count']})"):
            for r in rubric_stats["rubrics"]:
                st.text(f"  {r.get('名称', r.get('id', '?'))}")


def render_recent_conversations():
    st.subheader("Recent Conversations")

    p = Path(args.db)
    if not p.exists():
        st.caption("No conversation database")
        return

    try:
        from trainable_openclaw.logging.conversation_store import ConversationStore
        store = ConversationStore(str(p))
        sessions = store.list_sessions(limit=10)
        store.close()

        if not sessions:
            st.caption("No conversations yet")
            return

        for sess in sessions[:5]:
            sid = sess.get("session_id", "?")
            user = sess.get("user_id", "?")
            msg_count = sess.get("message_count", 0)
            updated = sess.get("updated_at", "")

            with st.expander(f"{user} · {msg_count} msgs · {updated[:19] if updated else '?'}"):
                try:
                    store2 = ConversationStore(str(p))
                    messages = store2.get_messages(sid)
                    store2.close()
                    for msg in messages:
                        role = msg.get("role", "?")
                        content = msg.get("content", "")[:200]
                        st.caption(f"**{role}**: {content}")
                except Exception:
                    st.caption("(messages unavailable)")
    except Exception as e:
        st.caption(f"Store error: {e}")


# --- Main ---

def main():
    render_header()

    if args.refresh > 0:
        st.markdown(
            f'<meta http-equiv="refresh" content="{args.refresh}">',
            unsafe_allow_html=True,
        )

    # Top row: server status + rubric stats
    col1, col2 = st.columns(2)
    with col1:
        render_mode_status()
    with col2:
        render_rubric_status()

    st.divider()

    # Mid row: training progress + evaluation results
    col3, col4 = st.columns(2)
    with col3:
        render_training_progress()
    with col4:
        render_evaluation_results()

    st.divider()

    # Bottom: recent conversations
    render_recent_conversations()

    st.caption(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
