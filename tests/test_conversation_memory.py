"""对话记忆分层策略测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.conversation_memory import ConversationMemory


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.last_query = ""
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.last_query = query
        self.rowcount = 3 if query.strip().upper().startswith("DELETE") else 0

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConn:
    def cursor(self):
        return FakeCursor()

    def commit(self):
        pass


def _memory() -> ConversationMemory:
    memory = ConversationMemory.__new__(ConversationMemory)
    memory.pg = type("PG", (), {"conn": FakeConn()})()
    memory.max_hot_messages = 4
    memory.max_warm_messages = 4
    memory.max_context_tokens = 80
    memory.summary_batch_size = 3
    return memory


class TestConversationMemoryTiers:
    def test_partition_message_ids(self):
        memory = _memory()
        rows = [(i,) for i in range(10, 0, -1)]

        class Conn(FakeConn):
            def cursor(self_inner):
                return FakeCursor(rows)

        memory.pg.conn = Conn()
        hot, warm, cold = memory._partition_message_ids("s1")

        assert hot == [10, 9, 8, 7]
        assert warm == [6, 5, 4, 3]
        assert cold == [2, 1]

    def test_context_trim_keeps_recent_tail(self):
        memory = _memory()
        text = "\n".join(f"消息{i}: " + "很长的内容" * 10 for i in range(20))

        trimmed = memory._trim_text_to_token_budget(text, max_tokens=60)

        assert "消息19" in trimmed
        assert "消息0" not in trimmed

    def test_extractive_summary_uses_role_prefixes(self):
        memory = _memory()
        rows = [
            (1, "user", "我喜欢清淡饮食"),
            (2, "assistant", "已记录你的偏好"),
        ]

        summary = memory._build_extractive_summary(rows)

        assert "用户: 我喜欢清淡饮食" in summary
        assert "助手: 已记录你的偏好" in summary

    def test_cleanup_retention_deletes_expired_cold(self):
        memory = _memory()

        deleted = memory._delete_expired_cold("s1", cold_ttl_days=180)

        assert deleted == 3

    def test_warm_summary_waits_until_batch_threshold(self):
        memory = _memory()
        memory._get_warm_summary_metadata = lambda session_id: {
            "source_start_id": 3,
            "source_end_id": 6,
        }
        refreshed = []
        memory._refresh_warm_summary = lambda session_id, warm_ids: refreshed.append(warm_ids)

        memory._maybe_refresh_warm_summary("s1", [8, 7, 6, 5])

        assert refreshed == []

    def test_warm_summary_refreshes_at_batch_threshold(self):
        memory = _memory()
        memory._get_warm_summary_metadata = lambda session_id: {
            "source_start_id": 3,
            "source_end_id": 6,
        }
        refreshed = []
        memory._refresh_warm_summary = lambda session_id, warm_ids: refreshed.append(warm_ids)

        memory._maybe_refresh_warm_summary("s1", [9, 8, 7, 6])

        assert refreshed == [[9, 8, 7, 6]]

    def test_context_includes_pending_warm_messages(self):
        memory = _memory()
        memory.get_warm_summary = lambda session_id: "旧的温层摘要"
        memory.get_pending_warm_context = lambda session_id: [
            {"role": "user", "content": "尚未进入摘要的偏好"},
        ]
        memory.get_context = lambda session_id, limit: [
            {"role": "assistant", "content": "最近回复"},
        ]

        context = memory.get_context_text("s1", max_tokens=500)

        assert "旧的温层摘要" in context
        assert "尚未进入摘要的偏好" in context
        assert "最近回复" in context
