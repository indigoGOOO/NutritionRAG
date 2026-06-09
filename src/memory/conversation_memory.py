"""对话历史记忆 - 基于PostgreSQL的会话管理

存储每次问答的完整对话历史，支持：
- 按 session_id 存/取对话
- 热/温/冷分层，不删除历史原文
- 热层：最近消息原文，直接进入 LLM 上下文
- 温层：较近历史生成摘要，辅助上下文
- 冷层：更早历史归档保存，不进入 LLM 上下文
- 按 token 预算裁剪最终上下文
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from src.memory.base import BaseMemory, ConversationTurn, MemoryItem
from src.storage.pg_client import PostgreSQLClient

logger = logging.getLogger(__name__)


class ConversationMemory(BaseMemory):
    """对话历史记忆"""

    def __init__(
        self,
        pg: PostgreSQLClient,
        max_turns: int = 30,
        max_warm_messages: int = 120,
        max_context_tokens: int = 3000,
        cold_ttl_days: int = 180,
        max_messages_per_session: int = 2000,
        summary_batch_size: int = 10,
    ):
        self.pg = pg
        # max_turns 兼容旧参数名；这里实际表示热层保留的 message 条数。
        self.max_hot_messages = max_turns
        self.max_warm_messages = max_warm_messages
        self.max_context_tokens = max_context_tokens
        self.cold_ttl_days = cold_ttl_days
        self.max_messages_per_session = max_messages_per_session
        self.summary_batch_size = summary_batch_size
        self._init_table()

    def _init_table(self):
        with self.pg.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tier VARCHAR(20) DEFAULT 'hot',
                    archived_at TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_conv_session
                    ON conversations(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_conv_session_tier
                    ON conversations(session_id, tier, created_at DESC);
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    tier VARCHAR(20) NOT NULL DEFAULT 'warm',
                    summary TEXT NOT NULL,
                    source_start_id INTEGER,
                    source_end_id INTEGER,
                    message_count INTEGER DEFAULT 0,
                    token_count INTEGER DEFAULT 0,
                    pending_message_count INTEGER DEFAULT 0,
                    summary_version INTEGER DEFAULT 1,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, tier)
                );
            """)
            cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS tier VARCHAR(20) DEFAULT 'hot';")
            cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP;")
            cur.execute(
                "ALTER TABLE conversation_summaries "
                "ADD COLUMN IF NOT EXISTS pending_message_count INTEGER DEFAULT 0;"
            )
            cur.execute(
                "ALTER TABLE conversation_summaries "
                "ADD COLUMN IF NOT EXISTS summary_version INTEGER DEFAULT 1;"
            )
        self.pg.conn.commit()
        logger.debug("ConversationMemory: 表已就绪")

    # ---- 公开 API ----

    def add_turn(self, session_id: str, turn: ConversationTurn):
        """添加一轮对话"""
        self._insert(session_id, turn)
        self._refresh_tiers(session_id)
        self.cleanup_retention(session_id=session_id)
        logger.debug(f"[Conv] 记录对话: {session_id} / {turn.role}")

    def add_user_message(self, session_id: str, content: str):
        self.add_turn(session_id, ConversationTurn(role="user", content=content))

    def add_assistant_message(self, session_id: str, content: str, metadata: dict | None = None):
        self.add_turn(
            session_id,
            ConversationTurn(role="assistant", content=content, metadata=metadata or {}),
        )

    def get_context(self, session_id: str, limit: int = 30) -> list[dict]:
        """获取最近 limit 条热层对话原文。"""
        rows = self._query_hot(session_id, limit)
        return [{"role": r[1], "content": r[2]} for r in rows]

    def get_context_text(self, session_id: str, limit: int = 30, max_tokens: int | None = None) -> str:
        """获取用于 LLM 的纯文本上下文：温层摘要 + 热层原文，并按 token 裁剪。"""
        max_tokens = max_tokens or self.max_context_tokens
        parts = []

        warm_summary = self.get_warm_summary(session_id)
        if warm_summary:
            parts.append(f"温层历史摘要:\n{warm_summary}")

        pending_warm = self.get_pending_warm_context(session_id)
        if pending_warm:
            pending_lines = []
            for turn in pending_warm:
                prefix = "用户" if turn["role"] == "user" else "助手"
                pending_lines.append(f"{prefix}: {turn['content']}")
            parts.append("尚未归入温层摘要的较早对话:\n" + "\n".join(pending_lines))

        turns = self.get_context(session_id, limit)
        lines = []
        for t in turns:
            prefix = "用户" if t["role"] == "user" else "助手"
            lines.append(f"{prefix}: {t['content']}")
        if lines:
            parts.append("最近对话:\n" + "\n".join(lines))

        return self._trim_text_to_token_budget("\n\n".join(parts), max_tokens)

    def get_warm_summary(self, session_id: str) -> str:
        """获取温层摘要。"""
        with self.pg.conn.cursor() as cur:
            cur.execute(
                """SELECT summary FROM conversation_summaries
                   WHERE session_id = %s AND tier = 'warm'""",
                (session_id,),
            )
            row = cur.fetchone()
            return row[0] if row else ""

    def get_pending_warm_context(self, session_id: str) -> list[dict]:
        """Return warm messages added after the latest summary snapshot."""
        metadata = self._get_warm_summary_metadata(session_id)
        if not metadata:
            return []

        source_end_id = metadata.get("source_end_id")
        if source_end_id is None:
            return []

        with self.pg.conn.cursor() as cur:
            cur.execute(
                """SELECT id, role, content
                   FROM conversations
                   WHERE session_id = %s
                     AND tier = 'warm'
                     AND id > %s
                   ORDER BY id ASC""",
                (session_id, source_end_id),
            )
            rows = cur.fetchall()
        return [{"role": row[1], "content": row[2]} for row in rows]

    def clear_session(self, session_id: str):
        """清空某会话的全部历史"""
        with self.pg.conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM conversation_summaries WHERE session_id = %s", (session_id,))
        self.pg.conn.commit()
        logger.info(f"[Conv] 清空会话: {session_id}")

    def count_turns(self, session_id: str) -> int:
        with self.pg.conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM conversations WHERE session_id = %s", (session_id,),
            )
            return cur.fetchone()[0]

    def count_by_tier(self, session_id: str) -> dict[str, int]:
        """统计某会话各层消息数。"""
        with self.pg.conn.cursor() as cur:
            cur.execute(
                """SELECT tier, COUNT(*) FROM conversations
                   WHERE session_id = %s GROUP BY tier""",
                (session_id,),
            )
            return {row[0]: row[1] for row in cur.fetchall()}

    def cleanup_retention(
        self,
        session_id: str | None = None,
        cold_ttl_days: int | None = None,
        max_messages_per_session: int | None = None,
    ) -> dict[str, int]:
        """清理会话原文，防止 conversations 无限增长。

        策略：
        - 删除 archived_at 超过 cold_ttl_days 的 cold 消息
        - 每个 session 最多保留 max_messages_per_session 条，超出时优先删除最早 cold
        """
        cold_ttl_days = cold_ttl_days if cold_ttl_days is not None else self.cold_ttl_days
        max_messages_per_session = (
            max_messages_per_session
            if max_messages_per_session is not None
            else self.max_messages_per_session
        )

        deleted_by_ttl = self._delete_expired_cold(session_id, cold_ttl_days)
        deleted_by_limit = self._enforce_session_limit(session_id, max_messages_per_session)
        return {"expired_cold": deleted_by_ttl, "session_limit": deleted_by_limit}

    # ---- 基类接口 ----

    def add(self, item: MemoryItem) -> str:
        meta = item.metadata or {}
        turn = ConversationTurn(
            role=meta.get("role", "assistant"),
            content=item.content,
            metadata=meta,
        )
        self.add_turn(meta.get("session_id", "default"), turn)
        return item.id

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """对话记忆不支持语义搜索，返回最近记录"""
        turns = self.get_context("default", limit)
        return [
            MemoryItem(
                id=f"conv_{i}",
                content=t["content"],
                metadata={"role": t["role"], "source": "conversation"},
                timestamp=datetime.now(),
            )
            for i, t in enumerate(turns)
        ]

    def remove(self, item_id: str) -> bool:
        return False  # 对话记忆不支持单条删除

    # ---- 内部 ----

    def _insert(self, session_id: str, turn: ConversationTurn):
        with self.pg.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO conversations (session_id, role, content, metadata, created_at)
                   VALUES (%s, %s, %s, %s, %s)""",
                (
                    session_id,
                    turn.role,
                    turn.content,
                    json.dumps(turn.metadata, ensure_ascii=False),
                    turn.timestamp,
                ),
            )
        self.pg.conn.commit()

    def _query_hot(self, session_id: str, limit: int) -> list[tuple]:
        with self.pg.conn.cursor() as cur:
            cur.execute(
                """SELECT id, role, content, created_at
                   FROM conversations
                   WHERE session_id = %s
                     AND tier = 'hot'
                   ORDER BY id DESC
                   LIMIT %s""",
                (session_id, limit),
            )
            rows = cur.fetchall()
        return list(reversed(rows))  # 按时间正序返回

    def _refresh_tiers(self, session_id: str):
        """刷新热/温/冷分层。历史原文保留，不做 DELETE。"""
        hot_ids, warm_ids, cold_ids = self._partition_message_ids(session_id)

        with self.pg.conn.cursor() as cur:
            if hot_ids:
                cur.execute(
                    "UPDATE conversations SET tier = 'hot', archived_at = NULL WHERE id = ANY(%s)",
                    (hot_ids,),
                )
            if warm_ids:
                cur.execute(
                    "UPDATE conversations SET tier = 'warm', archived_at = NULL WHERE id = ANY(%s)",
                    (warm_ids,),
                )
            if cold_ids:
                cur.execute(
                    """UPDATE conversations
                       SET tier = 'cold',
                           archived_at = COALESCE(archived_at, CURRENT_TIMESTAMP)
                       WHERE id = ANY(%s)""",
                    (cold_ids,),
                )
        self.pg.conn.commit()
        self._maybe_refresh_warm_summary(session_id, warm_ids)

    def _partition_message_ids(self, session_id: str) -> tuple[list[int], list[int], list[int]]:
        """按新到旧划分：最近为热层，其后为温层，再早为冷层。"""
        with self.pg.conn.cursor() as cur:
            cur.execute(
                """SELECT id FROM conversations
                   WHERE session_id = %s
                   ORDER BY id DESC""",
                (session_id,),
            )
            ids_desc = [row[0] for row in cur.fetchall()]

        hot_ids = ids_desc[: self.max_hot_messages]
        warm_end = self.max_hot_messages + self.max_warm_messages
        warm_ids = ids_desc[self.max_hot_messages : warm_end]
        cold_ids = ids_desc[warm_end:]
        return hot_ids, warm_ids, cold_ids

    def _maybe_refresh_warm_summary(self, session_id: str, warm_ids: list[int]):
        """Refresh the warm summary only after enough window changes."""
        if not warm_ids:
            self._refresh_warm_summary(session_id, warm_ids)
            return

        metadata = self._get_warm_summary_metadata(session_id)
        if not metadata:
            self._refresh_warm_summary(session_id, warm_ids)
            return

        source_start_id = metadata.get("source_start_id")
        source_end_id = metadata.get("source_end_id")
        summarized_ids = {
            message_id
            for message_id in warm_ids
            if source_start_id is not None
            and source_end_id is not None
            and source_start_id <= message_id <= source_end_id
        }
        changed_count = len(warm_ids) - len(summarized_ids)
        batch_size = max(1, int(getattr(self, "summary_batch_size", 10)))

        if changed_count >= batch_size:
            self._refresh_warm_summary(session_id, warm_ids)
            return

        with self.pg.conn.cursor() as cur:
            cur.execute(
                """UPDATE conversation_summaries
                   SET pending_message_count = %s
                   WHERE session_id = %s AND tier = 'warm'""",
                (changed_count, session_id),
            )
        self.pg.conn.commit()

    def _get_warm_summary_metadata(self, session_id: str) -> dict | None:
        with self.pg.conn.cursor() as cur:
            cur.execute(
                """SELECT source_start_id, source_end_id, message_count,
                          pending_message_count, summary_version, updated_at
                   FROM conversation_summaries
                   WHERE session_id = %s AND tier = 'warm'""",
                (session_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "source_start_id": row[0],
            "source_end_id": row[1],
            "message_count": row[2],
            "pending_message_count": row[3],
            "summary_version": row[4],
            "updated_at": row[5],
        }

    def _refresh_warm_summary(self, session_id: str, warm_ids: list[int]):
        """用温层消息生成轻量摘要。没有外部LLM时采用抽取式摘要。"""
        if not warm_ids:
            with self.pg.conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversation_summaries WHERE session_id = %s AND tier = 'warm'",
                    (session_id,),
                )
            self.pg.conn.commit()
            return

        with self.pg.conn.cursor() as cur:
            cur.execute(
                """SELECT id, role, content FROM conversations
                   WHERE id = ANY(%s)
                   ORDER BY id ASC""",
                (warm_ids,),
            )
            rows = cur.fetchall()

        summary = self._build_extractive_summary(rows)
        token_count = self._count_tokens(summary)
        with self.pg.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO conversation_summaries
                   (session_id, tier, summary, source_start_id, source_end_id,
                    message_count, token_count, pending_message_count, summary_version, updated_at)
                   VALUES (%s, 'warm', %s, %s, %s, %s, %s, 0, 1, CURRENT_TIMESTAMP)
                   ON CONFLICT (session_id, tier) DO UPDATE SET
                     summary = EXCLUDED.summary,
                     source_start_id = EXCLUDED.source_start_id,
                     source_end_id = EXCLUDED.source_end_id,
                     message_count = EXCLUDED.message_count,
                     token_count = EXCLUDED.token_count,
                     pending_message_count = 0,
                     summary_version = conversation_summaries.summary_version + 1,
                     updated_at = CURRENT_TIMESTAMP""",
                (session_id, summary, rows[0][0], rows[-1][0], len(rows), token_count),
            )
        self.pg.conn.commit()

    def _build_extractive_summary(self, rows: list[tuple]) -> str:
        """抽取式温层摘要：保留较早上下文的关键信息，控制长度。"""
        lines = []
        for _, role, content in rows:
            prefix = "用户" if role == "user" else "助手"
            compact = " ".join(content.split())
            lines.append(f"{prefix}: {compact[:240]}")

        text = "\n".join(lines)
        return self._trim_text_to_token_budget(text, max(400, self.max_context_tokens // 3))

    def _trim_text_to_token_budget(self, text: str, max_tokens: int) -> str:
        """按 token 预算从尾部保留文本，确保最近内容优先。"""
        if self._count_tokens(text) <= max_tokens:
            return text

        lines = text.splitlines()
        kept = []
        total = 0
        for line in reversed(lines):
            line_tokens = self._count_tokens(line)
            if kept and total + line_tokens > max_tokens:
                break
            kept.append(line)
            total += line_tokens
        return "\n".join(reversed(kept))

    @staticmethod
    def _count_tokens(text: str) -> int:
        try:
            import tiktoken

            tokenizer = tiktoken.get_encoding("cl100k_base")
            return len(tokenizer.encode(text))
        except Exception:
            return max(1, int(len(text) * 0.7))

    def _delete_expired_cold(self, session_id: str | None, cold_ttl_days: int) -> int:
        cutoff = datetime.now() - timedelta(days=cold_ttl_days)
        with self.pg.conn.cursor() as cur:
            if session_id:
                cur.execute(
                    """DELETE FROM conversations
                       WHERE session_id = %s
                         AND tier = 'cold'
                         AND archived_at IS NOT NULL
                         AND archived_at < %s""",
                    (session_id, cutoff),
                )
            else:
                cur.execute(
                    """DELETE FROM conversations
                       WHERE tier = 'cold'
                         AND archived_at IS NOT NULL
                         AND archived_at < %s""",
                    (cutoff,),
                )
            deleted = cur.rowcount if cur.rowcount is not None else 0
        self.pg.conn.commit()
        return deleted

    def _enforce_session_limit(self, session_id: str | None, max_messages: int) -> int:
        session_ids = [session_id] if session_id else self._list_session_ids()
        deleted_total = 0
        for sid in session_ids:
            overflow = self.count_turns(sid) - max_messages
            if overflow <= 0:
                continue
            with self.pg.conn.cursor() as cur:
                cur.execute(
                    """DELETE FROM conversations
                       WHERE id IN (
                         SELECT id FROM conversations
                         WHERE session_id = %s
                           AND tier = 'cold'
                         ORDER BY id ASC
                         LIMIT %s
                       )""",
                    (sid, overflow),
                )
                deleted_total += cur.rowcount if cur.rowcount is not None else 0
            self.pg.conn.commit()
        return deleted_total

    def _list_session_ids(self) -> list[str]:
        with self.pg.conn.cursor() as cur:
            cur.execute("SELECT DISTINCT session_id FROM conversations")
            return [row[0] for row in cur.fetchall()]
