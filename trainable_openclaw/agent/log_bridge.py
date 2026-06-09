"""
T3: Log bridge — dual-write between nanobot SessionManager (JSONL) and
our ConversationStore (SQLite).

When nanobot processes messages, we intercept session writes to also
record them in our SQLite store for training data extraction.

Usage::

    store = ConversationStore("data/conversations.db")
    bridge = LogBridge(store)

    # Option A: wrap nanobot's SessionManager
    bridge.wrap_session_manager(nanobot_sessions)

    # Option B: sync nanobot sessions → SQLite after the fact
    bridge.sync_from_nanobot(workspace=Path("~/.nanobot/workspace"))
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LogBridge:
    """Bridges nanobot's JSONL session format to our SQLite ConversationStore.

    Two modes:
    1. **Live wrapping** — patch ``Session.add_message`` to dual-write
    2. **Batch sync** — scan nanobot sessions dir and import to SQLite
    """

    def __init__(self, store: Any, auto_sync: bool = False):
        self.store = store
        self.auto_sync = auto_sync
        self._session_map: dict[str, str] = {}  # nanobot_key → sqlite_session_id

    # ------------------------------------------------------------------
    # Live wrapping
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_key_from_session(session: Any) -> str:
        return getattr(session, "key", "unknown")

    def on_message_added(
        self,
        session_key: str,
        role: str,
        content: str,
        **kwargs: Any,
    ) -> None:
        """Called when nanobot adds a message to a session.

        Mirrors the message into our SQLite store.
        """
        if not self.auto_sync:
            return

        sid = self._session_map.get(session_key)
        if sid is None:
            # Get or create the SQLite session
            channel, chat_id = (
                session_key.split(":", 1)
                if ":" in session_key
                else (session_key, "")
            )
            user_id = channel or "nanobot"
            sid = self.store.create_session(
                user_id=user_id,
                model="serve_ppo/qwen3-4b",
                metadata={"source": "nanobot", "session_key": session_key},
            )
            self._session_map[session_key] = sid

        self.store.add_message(
            session_id=sid,
            role=role,
            content=content,
            metadata=kwargs or None,
        )

    def wrap_session_manager(self, session_manager: Any) -> None:
        """Monkey-patch nanobot's SessionManager.save to also write to SQLite."""
        original_save = session_manager.save
        bridge = self

        def _dual_write_save(session: Any, **save_kwargs: Any) -> None:
            original_save(session, **save_kwargs)
            try:
                bridge._sync_session(session)
            except Exception:
                logger.debug("dual-write sync failed for %s", session.key, exc_info=True)

        session_manager.save = _dual_write_save  # type: ignore[method-assign]
        logger.info("LogBridge: wrapped SessionManager.save for dual-write")

    def _sync_session(self, session: Any) -> None:
        """Sync a single nanobot session into SQLite."""
        session_key = getattr(session, "key", "")
        if not session_key:
            return

        channel, chat_id = (
            session_key.split(":", 1) if ":" in session_key else (session_key, "")
        )
        user_id = channel or "nanobot"

        sid = self._session_map.get(session_key)
        if sid is None:
            sid = self.store.create_session(
                user_id=user_id,
                model="serve_ppo/qwen3-4b",
                metadata={"source": "nanobot", "session_key": session_key},
            )
            self._session_map[session_key] = sid

        # Sync new messages (append-only by created_at heuristic)
        existing = self.store.get_messages(sid)
        existing_count = len(existing)
        messages = getattr(session, "messages", [])
        new_msgs = messages[existing_count:]

        for msg in new_msgs:
            self.store.add_message(
                session_id=sid,
                role=msg.get("role", "unknown"),
                content=str(msg.get("content", "")),
                metadata={
                    k: v
                    for k, v in msg.items()
                    if k not in ("role", "content")
                },
            )

        if new_msgs:
            logger.debug(
                "Synced %d messages for session %s", len(new_msgs), session_key,
            )

    # ------------------------------------------------------------------
    # Batch sync (offline import)
    # ------------------------------------------------------------------

    def sync_from_nanobot(
        self,
        workspace: Path,
        *,
        max_sessions: int = 50,
    ) -> int:
        """Scan nanobot sessions dir and import all sessions to SQLite.

        Returns the count of newly imported messages.
        """
        sessions_dir = workspace / "sessions"
        if not sessions_dir.exists():
            logger.warning("nanobot sessions dir not found: %s", sessions_dir)
            return 0

        total_imported = 0
        paths = sorted(
            sessions_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:max_sessions]

        for jsonl_path in paths:
            try:
                session = self._load_nanobot_session(jsonl_path)
                if session and session.get("messages"):
                    count = self._import_session(session)
                    total_imported += count
            except Exception:
                logger.debug("Failed to import %s", jsonl_path, exc_info=True)

        logger.info(
            "Batch sync: %d sessions, %d messages imported",
            len(paths),
            total_imported,
        )
        return total_imported

    @staticmethod
    def _load_nanobot_session(path: Path) -> dict[str, Any] | None:
        """Load a nanobot JSONL session file."""
        try:
            messages: list[dict] = []
            metadata: dict = {}
            session_key = path.stem.replace("_", ":", 1)

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        session_key = data.get("key", session_key)
                    else:
                        messages.append(data)

            return {
                "key": session_key,
                "created_at": metadata.get("created_at"),
                "updated_at": metadata.get("updated_at"),
                "metadata": metadata,
                "messages": messages,
            }
        except Exception:
            logger.debug("Failed to load %s", path, exc_info=True)
            return None

    def _import_session(self, session: dict[str, Any]) -> int:
        """Import a loaded session dict into SQLite. Returns message count."""
        session_key = session["key"]
        channel = session_key.split(":", 1)[0] if ":" in session_key else session_key

        sid = self.store.create_session(
            user_id=channel,
            model="serve_ppo/qwen3-4b",
            metadata={
                "source": "nanobot_import",
                "session_key": session_key,
                "nanobot_metadata": session.get("metadata"),
            },
        )

        count = 0
        for msg in session.get("messages", []):
            self.store.add_message(
                session_id=sid,
                role=msg.get("role", "unknown"),
                content=str(msg.get("content", "")),
                metadata={
                    k: v
                    for k, v in msg.items()
                    if k not in ("role", "content")
                },
            )
            count += 1

        return count

    # ------------------------------------------------------------------
    # Export from SQLite → nanobot JSONL (reverse direction)
    # ------------------------------------------------------------------

    def export_to_nanobot(
        self,
        session_id: str,
        workspace: Path,
    ) -> Path | None:
        """Export a ConversationStore session to nanobot JSONL format."""
        sess = self.store.get_session(session_id)
        if not sess:
            logger.warning("Session %s not found", session_id)
            return None

        messages = self.store.get_messages(session_id)
        sessions_dir = workspace / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        safe_key = session_id.replace(":", "_")
        jsonl_path = sessions_dir / f"{safe_key}.jsonl"

        with open(jsonl_path, "w", encoding="utf-8") as f:
            # Metadata line
            meta = {
                "_type": "metadata",
                "key": f"imported:{session_id}",
                "created_at": datetime.fromtimestamp(sess["created_at"]).isoformat(),
                "updated_at": datetime.fromtimestamp(sess["updated_at"]).isoformat(),
                "metadata": {
                    "source": "sqlite_export",
                    "user_id": sess["user_id"],
                    "model": sess.get("model", ""),
                },
                "last_consolidated": 0,
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            for msg in messages:
                entry = {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": datetime.fromtimestamp(msg["created_at"]).isoformat(),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info("Exported session %s → %s (%d messages)", session_id, jsonl_path, len(messages))
        return jsonl_path
