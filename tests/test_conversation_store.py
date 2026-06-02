"""Tests for trainable_openclaw.logging.conversation_store — no GPU required."""

import os
import tempfile
import threading

import pytest

from trainable_openclaw.logging.conversation_store import ConversationStore


@pytest.fixture
def store():
    """Create a store backed by a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = ConversationStore(path)
    yield s
    s.close()
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_idempotent(store):
    """Creating the store twice on the same file does not error."""
    store2 = ConversationStore(store._db_path)
    # Should not raise
    store2.close()


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


def test_create_and_get_session(store):
    sid = store.create_session("alice", model="test-model", metadata={"tag": "test"})
    s = store.get_session(sid)
    assert s is not None
    assert s["user_id"] == "alice"
    assert s["model"] == "test-model"
    assert s["message_count"] == 0
    assert s["created_at"] > 0
    assert s["updated_at"] > 0


def test_get_session_missing(store):
    assert store.get_session("nonexistent") is None


def test_list_sessions_empty(store):
    assert store.list_sessions() == []


def test_list_sessions(store):
    store.create_session("alice")
    store.create_session("bob")
    store.create_session("alice")
    all_s = store.list_sessions()
    assert len(all_s) == 3
    # newest first
    assert all_s[0]["created_at"] >= all_s[-1]["created_at"]


def test_list_sessions_by_user(store):
    store.create_session("alice")
    store.create_session("bob")
    store.create_session("alice")
    alice_s = store.list_sessions(user_id="alice")
    assert len(alice_s) == 2
    bob_s = store.list_sessions(user_id="bob")
    assert len(bob_s) == 1


def test_list_sessions_limit_offset(store):
    for _ in range(5):
        store.create_session("alice")
    assert len(store.list_sessions(limit=3)) == 3
    assert len(store.list_sessions(limit=3, offset=3)) == 2


def test_delete_session_cascades(store):
    sid = store.create_session("alice")
    store.add_message(sid, "user", "hello")
    store.add_message(sid, "assistant", "hi")
    assert store.delete_session(sid) is True
    assert store.get_session(sid) is None
    assert store.get_messages(sid) == []


def test_delete_session_missing(store):
    assert store.delete_session("nonexistent") is False


# ---------------------------------------------------------------------------
# Message CRUD
# ---------------------------------------------------------------------------


def test_add_and_get_messages(store):
    sid = store.create_session("alice")
    mid1 = store.add_message(sid, "user", "What is 2+2?", token_count=6)
    mid2 = store.add_message(sid, "assistant", "4", token_count=1, latency_ms=123,
                             temperature=0.7, max_tokens=2048, stop_reason="stop")
    assert mid1 > 0
    assert mid2 > mid1

    msgs = store.get_messages(sid)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "What is 2+2?"
    assert msgs[0]["token_count"] == 6
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["latency_ms"] == 123
    assert msgs[1]["temperature"] == 0.7
    assert msgs[1]["max_tokens"] == 2048
    assert msgs[1]["stop_reason"] == "stop"


def test_get_messages_ordered(store):
    sid = store.create_session("alice")
    for i in range(5):
        store.add_message(sid, "user", f"msg {i}")
    msgs = store.get_messages(sid)
    assert [m["content"] for m in msgs] == [f"msg {i}" for i in range(5)]


def test_message_bumps_session(store):
    sid = store.create_session("alice")
    old_updated = store.get_session(sid)["updated_at"]
    assert store.get_session(sid)["message_count"] == 0
    store.add_message(sid, "user", "hello")
    s = store.get_session(sid)
    assert s["message_count"] == 1
    assert s["updated_at"] >= old_updated


def test_metadata_json(store):
    sid = store.create_session("alice", metadata={"key": "value"})
    msgs = store.get_messages(sid)
    # metadata stored as JSON text in sessions, but returned as-is
    s = store.get_session(sid)
    assert '"key"' in s["metadata"]  # JSON string


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def test_query_messages_by_user(store):
    s1 = store.create_session("alice")
    s2 = store.create_session("bob")
    store.add_message(s1, "user", "alice msg")
    store.add_message(s2, "user", "bob msg")

    results = store.query_messages(user_id="alice")
    assert len(results) == 1
    assert results[0]["content"] == "alice msg"


def test_query_messages_by_role(store):
    sid = store.create_session("alice")
    store.add_message(sid, "user", "q1")
    store.add_message(sid, "assistant", "a1")
    store.add_message(sid, "user", "q2")

    assert len(store.query_messages(role="user")) == 2
    assert len(store.query_messages(role="assistant")) == 1
    assert len(store.query_messages(role="system")) == 0


def test_query_messages_by_time(store):
    import time
    t0 = time.time()
    sid = store.create_session("alice")
    store.add_message(sid, "user", "old")
    time.sleep(0.01)
    t1 = time.time()
    store.add_message(sid, "assistant", "new")

    results = store.query_messages(start_time=t1)
    assert len(results) == 1
    assert results[0]["content"] == "new"

    results = store.query_messages(end_time=t0 + 0.005)
    assert len(results) == 1
    assert results[0]["content"] == "old"


def test_query_messages_limit(store):
    sid = store.create_session("alice")
    for i in range(10):
        store.add_message(sid, "user", f"msg {i}")
    assert len(store.query_messages(limit=3)) == 3


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_content(store):
    sid = store.create_session("alice")
    store.add_message(sid, "user", "How do I fix a NullPointerException?")
    store.add_message(sid, "assistant", "Check if the object is null first.")
    store.add_message(sid, "user", "Thanks, that worked.")

    results = store.search_content("NullPointer")
    assert len(results) == 1

    results = store.search_content("null")
    assert len(results) == 2  # SQLite LIKE is case-insensitive for ASCII: matches both messages

    results = store.search_content("worked")
    assert len(results) == 1

    results = store.search_content("nonexistent")
    assert len(results) == 0


def test_search_content_limit(store):
    sid = store.create_session("alice")
    for i in range(5):
        store.add_message(sid, "user", f"error code {i}")
    assert len(store.search_content("error", limit=3)) == 3


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_statistics_empty(store):
    stats = store.get_statistics()
    assert stats["total_sessions"] == 0
    assert stats["total_messages"] == 0
    assert stats["total_users"] == 0


def test_statistics(store):
    s1 = store.create_session("alice")
    s2 = store.create_session("bob")
    store.add_message(s1, "user", "a")
    store.add_message(s1, "assistant", "b")
    store.add_message(s2, "user", "c")

    stats = store.get_statistics()
    assert stats["total_sessions"] == 2
    assert stats["total_messages"] == 3
    assert stats["total_users"] == 2
    assert stats["role_distribution"]["user"] == 2
    assert stats["role_distribution"]["assistant"] == 1
    assert len(stats["user_breakdown"]) == 2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_writes(store):
    errors = []

    def writer(n):
        try:
            sid = store.create_session(f"user_{n}")
            for i in range(10):
                store.add_message(sid, "user", f"msg {n}.{i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    stats = store.get_statistics()
    assert stats["total_sessions"] == 10
    assert stats["total_messages"] == 100


# ---------------------------------------------------------------------------
# Unicode & special characters
# ---------------------------------------------------------------------------


def test_unicode_content(store):
    sid = store.create_session("测试用户")
    store.add_message(sid, "user", "你好，世界！")
    store.add_message(sid, "assistant", "Hello 世界 🌍")
    msgs = store.get_messages(sid)
    assert len(msgs) == 2
    assert msgs[0]["content"] == "你好，世界！"
    assert msgs[1]["content"] == "Hello 世界 🌍"


def test_special_sql_characters(store):
    sid = store.create_session("alice")
    tricky = "What's up? SELECT * FROM users; -- DROP TABLE messages;"
    store.add_message(sid, "user", tricky)
    msgs = store.get_messages(sid)
    assert msgs[0]["content"] == tricky


def test_long_content(store):
    sid = store.create_session("alice")
    long_text = "Hello " * 10000
    store.add_message(sid, "user", long_text)
    msgs = store.get_messages(sid)
    assert len(msgs[0]["content"]) == len(long_text)


def test_empty_content(store):
    sid = store.create_session("alice")
    store.add_message(sid, "user", "")
    msgs = store.get_messages(sid)
    assert msgs[0]["content"] == ""


def test_search_unicode(store):
    sid = store.create_session("alice")
    store.add_message(sid, "user", "检查代码中的异常处理")
    store.add_message(sid, "user", "testing error handling")
    results = store.search_content("异常")
    assert len(results) == 1
    results = store.search_content("error")
    assert len(results) == 1


def test_zero_token_count(store):
    sid = store.create_session("alice")
    mid = store.add_message(sid, "user", "hi", token_count=0)
    msgs = store.get_messages(sid)
    assert msgs[0]["token_count"] == 0


def test_many_messages_in_session(store):
    sid = store.create_session("alice")
    for i in range(500):
        store.add_message(sid, "user" if i % 2 == 0 else "assistant", f"msg {i}")
    assert len(store.get_messages(sid)) == 500
    s = store.get_session(sid)
    assert s["message_count"] == 500


def test_statistics_after_delete(store):
    sid = store.create_session("alice")
    store.add_message(sid, "user", "a")
    store.add_message(sid, "assistant", "b")
    store.delete_session(sid)
    stats = store.get_statistics()
    assert stats["total_sessions"] == 0
    assert stats["total_messages"] == 0


def test_create_session_without_model(store):
    sid = store.create_session("alice")
    s = store.get_session(sid)
    assert s["model"] is None


def test_add_message_without_optionals(store):
    sid = store.create_session("alice")
    mid = store.add_message(sid, "system", "Ready.")
    msgs = store.get_messages(sid)
    m = msgs[0]
    assert m["token_count"] is None
    assert m["latency_ms"] is None
    assert m["temperature"] is None
    assert m["max_tokens"] is None
    assert m["stop_reason"] is None


def test_user_id_and_content_in_query_result(store):
    """query_messages should include user_id from JOIN with sessions."""
    sid = store.create_session("bob")
    store.add_message(sid, "user", "hello")
    results = store.query_messages()
    assert len(results) >= 1
    assert results[0]["user_id"] == "bob"
    assert results[0]["content"] == "hello"


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def test_close(store):
    store.close()
    # closing twice is safe with CPython but we just verify no crash
    store.close()


# ---------------------------------------------------------------------------
# Telemetry events (Phase 3)
# ---------------------------------------------------------------------------


class TestTelemetry:
    """Tests for record_telemetry / get_telemetry / get_telemetry_stats."""

    def test_record_and_get_telemetry(self, store):
        sid = store.create_session("alice")
        eid = store.record_telemetry(sid, "user.accepted", rating=5, trace_id="t-123")
        assert eid > 0

        events = store.get_telemetry(session_id=sid)
        assert len(events) == 1
        assert events[0]["event_type"] == "user.accepted"
        assert events[0]["rating"] == 5
        assert events[0]["trace_id"] == "t-123"

    def test_filter_by_event_type(self, store):
        sid = store.create_session("alice")
        store.record_telemetry(sid, "user.accepted", rating=5)
        store.record_telemetry(sid, "user.corrected", rating=3, correction="请修正")
        store.record_telemetry(sid, "user.abandoned", rating=1)

        accepted = store.get_telemetry(session_id=sid, event_type="user.accepted")
        assert len(accepted) == 1

        corrected = store.get_telemetry(session_id=sid, event_type="user.corrected")
        assert len(corrected) == 1
        assert corrected[0]["correction"] == "请修正"

    def test_telemetry_stats(self, store):
        sid = store.create_session("alice")
        store.record_telemetry(sid, "user.accepted", rating=5)
        store.record_telemetry(sid, "user.accepted", rating=4)
        store.record_telemetry(sid, "user.corrected", rating=3)

        stats = store.get_telemetry_stats()
        assert stats["total_events"] == 3
        assert stats["by_type"]["user.accepted"] == 2
        assert stats["by_type"]["user.corrected"] == 1
        assert stats["avg_rating"] == 4.0

    def test_correction_stored(self, store):
        sid = store.create_session("alice")
        store.record_telemetry(
            sid, "user.corrected", rating=2,
            correction="名字取错了，应该叫 get_name",
        )
        events = store.get_telemetry(session_id=sid)
        assert events[0]["correction"] == "名字取错了，应该叫 get_name"

    def test_cascade_on_session_delete(self, store):
        sid = store.create_session("alice")
        store.record_telemetry(sid, "user.accepted", rating=5)
        store.delete_session(sid)
        events = store.get_telemetry(session_id=sid)
        assert len(events) == 0

    def test_empty_telemetry(self, store):
        assert store.get_telemetry() == []
        stats = store.get_telemetry_stats()
        assert stats["total_events"] == 0
        assert stats["by_type"] == {}


# ---------------------------------------------------------------------------
# Step export (Phase 3 — Trajectory-inspired)
# ---------------------------------------------------------------------------


class TestExportSteps:
    """Tests for export_steps — self-contained training sample extraction."""

    def test_empty_store(self, store):
        assert store.export_steps() == []

    def test_single_turn_no_feedback(self, store):
        sid = store.create_session("alice")
        store.add_message(sid, "user", "什么是机器学习？")
        store.add_message(sid, "assistant", "机器学习是AI的一个分支...")

        steps = store.export_steps()
        assert len(steps) == 1
        s = steps[0]
        assert len(s["messages_so_far"]) == 1
        assert s["messages_so_far"][0]["content"] == "什么是机器学习？"
        assert s["agent_action"]["content"] == "机器学习是AI的一个分支..."
        assert s["is_training_sample"] is False
        assert s["feedback"] is None

    def test_multi_turn(self, store):
        sid = store.create_session("alice")
        store.add_message(sid, "user", "Q1")
        store.add_message(sid, "assistant", "A1")
        store.add_message(sid, "user", "Q2")
        store.add_message(sid, "assistant", "A2")

        steps = store.export_steps()
        assert len(steps) == 2
        # Step 1: context=[Q1], action=A1
        assert len(steps[0]["messages_so_far"]) == 1
        assert steps[0]["agent_action"]["content"] == "A1"
        # Step 2: context=[Q1, A1, Q2], action=A2
        assert len(steps[1]["messages_so_far"]) == 3
        assert steps[1]["agent_action"]["content"] == "A2"

    def test_with_feedback(self, store):
        sid = store.create_session("alice")
        store.add_message(sid, "user", "写一个排序")
        store.add_message(sid, "assistant", "def sort(arr): return sorted(arr)")
        store.record_telemetry(sid, "user.corrected", rating=3, correction="变量名不表意")

        steps = store.export_steps()
        assert len(steps) == 1
        assert steps[0]["is_training_sample"] is True
        assert steps[0]["feedback"]["event_type"] == "user.corrected"
        assert steps[0]["feedback"]["correction"] == "变量名不表意"

    def test_with_feedback_only_filter(self, store):
        sid1 = store.create_session("alice")
        store.add_message(sid1, "user", "Q1")
        store.add_message(sid1, "assistant", "A1")
        # No feedback on sid1

        sid2 = store.create_session("bob")
        store.add_message(sid2, "user", "Q2")
        store.add_message(sid2, "assistant", "A2")
        store.record_telemetry(sid2, "user.accepted", rating=5)

        steps = store.export_steps(with_feedback_only=True)
        assert len(steps) == 1
        assert steps[0]["user_id"] == "bob"

    def test_session_filter(self, store):
        sid1 = store.create_session("alice")
        store.add_message(sid1, "user", "Q1")
        store.add_message(sid1, "assistant", "A1")

        sid2 = store.create_session("bob")
        store.add_message(sid2, "user", "Q2")
        store.add_message(sid2, "assistant", "A2")

        steps = store.export_steps(session_id=sid1)
        assert len(steps) == 1
        assert steps[0]["user_id"] == "alice"

    def test_limit(self, store):
        for i in range(5):
            sid = store.create_session(f"user_{i}")
            store.add_message(sid, "user", f"Q{i}")
            store.add_message(sid, "assistant", f"A{i}")

        steps = store.export_steps(limit=3)
        assert len(steps) == 3

    def test_metadata_preserved(self, store):
        sid = store.create_session("alice")
        store.add_message(sid, "user", "Q1")
        store.add_message(sid, "assistant", "A1", token_count=42, latency_ms=500.0)

        steps = store.export_steps()
        assert steps[0]["agent_action"]["token_count"] == 42
        assert steps[0]["agent_action"]["latency_ms"] == 500.0
