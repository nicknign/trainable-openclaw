"""
SQLite-backed conversation log store.

Stores user sessions and messages for LLM input/output logging.
Designed for local operation — zero external dependencies beyond
Python's built-in sqlite3.  WAL mode supports concurrent reads
during writes, and B-tree indexes power fast user/time queries.

Usage::

    store = ConversationStore("data/conversations.db")
    sid = store.create_session("user_42", model="qwen3-4b")
    store.add_message(sid, "user", "What is 2+2?", token_count=6)
    store.add_message(sid, "assistant", "4", token_count=1, latency_ms=123)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    model         TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    message_count INTEGER DEFAULT 0,
    metadata      TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_time
    ON sessions(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role           TEXT NOT NULL,
    content        TEXT NOT NULL,
    token_count    INTEGER,
    latency_ms     REAL,
    temperature    REAL,
    max_tokens     INTEGER,
    stop_reason    TEXT,
    created_at     REAL NOT NULL,
    metadata       TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_messages_content
    ON messages(content);
"""


class ConversationStore:
    """Thread-safe conversation logger backed by a local SQLite file.

    Parameters:
        db_path: Path to the ``.db`` file (directory is created automatically).
    """

    def __init__(self, db_path: str = "data/conversations.db") -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()
        logger.info("ConversationStore opened at %s", db_path)

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def create_session(
        self,
        user_id: str,
        model: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Create a new conversation session and return its UUID."""
        sid = uuid.uuid4().hex
        now = time.time()
        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (id, user_id, model, created_at, updated_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (sid, user_id, model, now, now, meta_json),
            )
            self._conn.commit()
        return sid

    def get_session(self, session_id: str) -> dict | None:
        """Return a session dict or *None*."""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(
        self,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Return recent sessions, optionally filtered by user."""
        if user_id:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (user_id, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and cascade-delete its messages."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Message CRUD
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        token_count: int | None = None,
        latency_ms: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_reason: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Append a message to *session_id* and bump ``updated_at`` + ``message_count``.

        Returns the new message's integer id.
        """
        now = time.time()
        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages "
                "(session_id, role, content, token_count, latency_ms, "
                "temperature, max_tokens, stop_reason, created_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, role, content, token_count, latency_ms,
                 temperature, max_tokens, stop_reason, now, meta_json),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ?, message_count = message_count + 1 "
                "WHERE id = ?",
                (now, session_id),
            )
            self._conn.commit()
        return cur.lastrowid

    def get_messages(self, session_id: str) -> list[dict]:
        """Return all messages for a session, ordered by time ascending."""
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Analysis queries  (B1 — feedback analysis)
    # ------------------------------------------------------------------

    def query_messages(
        self,
        user_id: str | None = None,
        role: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return messages matching the given filters, newest first."""
        clauses = []
        params: list[Any] = []

        if user_id is not None:
            clauses.append("s.user_id = ?")
            params.append(user_id)
        if role is not None:
            clauses.append("m.role = ?")
            params.append(role)
        if start_time is not None:
            clauses.append("m.created_at >= ?")
            params.append(start_time)
        if end_time is not None:
            clauses.append("m.created_at <= ?")
            params.append(end_time)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT m.*, s.user_id FROM messages m "
            f"JOIN sessions s ON m.session_id = s.id "
            f"{where} ORDER BY m.created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def search_content(self, keyword: str, limit: int = 50) -> list[dict]:
        """Full-text-style search on message content via LIKE."""
        rows = self._conn.execute(
            "SELECT m.*, s.user_id FROM messages m "
            "JOIN sessions s ON m.session_id = s.id "
            "WHERE m.content LIKE ? "
            "ORDER BY m.created_at DESC LIMIT ?",
            (f"%{keyword}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Return aggregate statistics for dashboard / analysis."""
        total_sessions = self._conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]
        total_messages = self._conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]
        total_users = self._conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM sessions"
        ).fetchone()[0]

        # Role distribution
        role_rows = self._conn.execute(
            "SELECT role, COUNT(*) as cnt FROM messages GROUP BY role ORDER BY cnt DESC"
        ).fetchall()
        role_dist = {r["role"]: r["cnt"] for r in role_rows}

        # User breakdown
        user_rows = self._conn.execute(
            "SELECT user_id, COUNT(*) as sessions, "
            "SUM(message_count) as msgs "
            "FROM sessions GROUP BY user_id ORDER BY msgs DESC LIMIT 20"
        ).fetchall()

        # Time range
        first = self._conn.execute(
            "SELECT MIN(created_at) FROM messages"
        ).fetchone()[0]
        last = self._conn.execute(
            "SELECT MAX(created_at) FROM messages"
        ).fetchone()[0]

        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_users": total_users,
            "role_distribution": role_dist,
            "user_breakdown": [dict(r) for r in user_rows],
            "first_message_at": first,
            "last_message_at": last,
        }

    # ------------------------------------------------------------------
    # Raw connection access (for CLI viewer)
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        """The underlying ``sqlite3.Connection`` (row_factory = sqlite3.Row)."""
        return self._conn

    def close(self) -> None:
        self._conn.close()
        logger.info("ConversationStore closed")
