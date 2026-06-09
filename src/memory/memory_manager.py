"""Memory manager: unified entry point for conversation, profile, and knowledge memory."""

from __future__ import annotations

import logging

from src.memory.base import MemoryItem
from src.memory.conversation_memory import ConversationMemory
from src.memory.knowledge_memory import KnowledgeMemory
from src.memory.profile_memory import ProfileMemory
from src.storage.milvus_client import MilvusClient
from src.storage.pg_client import PostgreSQLClient

logger = logging.getLogger(__name__)


class MemoryManager:
    """Unified memory facade used by the agent graph."""

    def __init__(
        self,
        pg: PostgreSQLClient,
        milvus: MilvusClient,
        user_id: str = "default",
    ):
        self.user_id = user_id
        self.conversation = ConversationMemory(pg)
        self.profile = ProfileMemory(pg, neo4j=None)
        self.knowledge = KnowledgeMemory(pg, milvus)

    # ==================== Conversation memory ====================

    def on_user_query(self, session_id: str, query: str):
        """Record the user query.

        Long-term profile extraction is handled by the graph through
        planner/confirm_profile_node, not by regex extraction here.
        """
        self.conversation.add_user_message(session_id, query)

    def on_assistant_answer(self, session_id: str, answer: str, metadata: dict | None = None):
        """Record the assistant answer."""
        self.conversation.add_assistant_message(session_id, answer, metadata)

    def get_conversation_context(self, session_id: str, limit: int = 10) -> str:
        """Return recent conversation context."""
        return self.conversation.get_context_text(session_id, limit)

    # ==================== Knowledge memory ====================

    def search_knowledge(self, query: str, limit: int = 3) -> list[MemoryItem]:
        """Search reusable historical QA memory."""
        return self.knowledge.search(query, limit)

    def store_qa(
        self,
        question: str,
        answer: str,
        entities: list[str] | None = None,
        tags: list[str] | None = None,
        citations: list[dict] | None = None,
        evidence_count: int = 0,
    ) -> int | None:
        """Store a high-quality QA pair into knowledge memory."""
        return self.knowledge.store_qa(
            question=question,
            answer=answer,
            entities=entities,
            tags=tags,
            citations=citations,
            evidence_count=evidence_count,
        )

    def find_reusable_answer(self, query: str, intent: str | None = None) -> MemoryItem | None:
        """Find a directly reusable low-risk historical answer."""
        return self.knowledge.find_reusable_answer(query, intent)

    # ==================== User profile ====================

    def set_user_preference(self, key: str, value: str):
        """Set a user profile preference."""
        self.profile.set_preference(self.user_id, key, value)

    def add_user_allergy(self, ingredient: str):
        """Add a user allergy."""
        self.profile.add_allergy(self.user_id, ingredient)

    def get_user_profile_text(self) -> str:
        """Format the user profile for prompts."""
        return self.profile.profile_to_text(self.user_id)

    def check_ingredient(self, ingredient: str) -> dict:
        """Check whether an ingredient is safe for this user."""
        return self.profile.check_ingredient_safety(self.user_id, ingredient)

    # ==================== Prompt context formatting ====================

    def format_personal_memory_context(self, session_id: str = "default") -> str:
        """Format only conversation history and user profile for fallback answers."""
        parts = []

        history = self.conversation.get_context_text(
            session_id, limit=self.conversation.max_hot_messages,
        )
        if history:
            parts.append(f"## 最近对话历史\n{history}")

        profile_text = self.profile.profile_to_text(self.user_id)
        if profile_text and profile_text != "暂无用户画像信息":
            parts.append(f"## 用户画像\n{profile_text}")

        return "\n\n".join(parts) if parts else ""

    def format_memory_context(
        self,
        query: str,
        session_id: str = "default",
        intent: str = "",
    ) -> str:
        """Format conversation, profile, and knowledge memory for Answer."""
        parts = []

        if hasattr(self, "conversation"):
            history = self.conversation.get_context_text(
                session_id, limit=self.conversation.max_hot_messages,
            )
            if history:
                parts.append(f"## 最近对话历史\n{history}")

        if hasattr(self, "profile"):
            profile_text = self.profile.profile_to_text(self.user_id)
            if profile_text and profile_text != "暂无用户画像信息":
                parts.append(f"## 用户画像\n{profile_text}")

        items = self.knowledge.search(query, limit=3) if hasattr(self, "knowledge") else []
        if items:
            direct_lines = []
            context_lines = []
            for item in items:
                policy = item.metadata.get("reuse_policy", "context_only")
                if intent and not self.knowledge.is_safe_reuse_intent(intent):
                    policy = "context_only"

                if policy == "direct_reuse":
                    direct_lines.append(
                        f"Q: {item.metadata.get('question', '')}\n"
                        f"A: {item.content}\n"
                        f"相似度: {item.score:.3f}\n"
                        "使用方式: 可作为主要参考，但必须核对当前检索证据；"
                        "当前证据不足时，可以引用该历史问答，并明确标注为“历史问答参考”。"
                    )
                else:
                    context_lines.append(
                        f"Q: {item.metadata.get('question', '')}\n"
                        f"摘要: {self._summarize_memory_answer(item.content)}\n"
                        f"相似度: {item.score:.3f}\n"
                        "使用方式: 只能辅助理解用户意图和保持口径一致；"
                        "不可作为事实依据，不允许作为来源引用。"
                    )

            knowledge_parts = []
            if direct_lines:
                knowledge_parts.append(
                    "## 可强参考的历史问答\n"
                    "说明：以下历史问答相似度高且低风险，可作为主要参考，但仍需核对当前检索证据。\n"
                    + "\n---\n".join(direct_lines)
                )
            if context_lines:
                knowledge_parts.append(
                    "## 仅供理解意图的历史问答\n"
                    "说明：以下内容只能帮助理解用户意图或保持表达一致，"
                    "不可作为事实依据，不允许作为来源引用。\n"
                    + "\n---\n".join(context_lines)
                )
            parts.append("\n\n".join(knowledge_parts))

        return "\n\n".join(parts) if parts else ""

    def format_answer_memory_context(self, query: str, intent: str = "") -> str:
        """Backward-compatible alias used by older tests/callers."""
        return self.format_memory_context(query=query, intent=intent)

    @staticmethod
    def _summarize_memory_answer(answer: str, max_chars: int = 180) -> str:
        """Create a short summary for context-only historical answers."""
        compact = " ".join(answer.split())
        if len(compact) <= max_chars:
            return compact
        return compact[:max_chars].rstrip() + "..."
