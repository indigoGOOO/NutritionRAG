"""Request-level runtime logging for agent runs.

This module stores a compact snapshot of each chat request. It intentionally
keeps evidence as summaries instead of full prompts or full memory context.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from src.storage.pg_client import PostgreSQLClient

logger = logging.getLogger(__name__)

MAX_EVIDENCE_ITEMS = 8
MAX_PREVIEW_CHARS = 160


class RuntimeLogger:
    """Best-effort runtime logger backed by PostgreSQL."""

    def __init__(self, pg: PostgreSQLClient):
        self.pg = pg
        self._tables_ready = False

    @staticmethod
    def new_request_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def now_ms() -> int:
        return int(time.perf_counter() * 1000)

    def log_chat(
        self,
        *,
        request_id: str,
        query: str,
        session_id: str,
        user_id: str,
        result: dict[str, Any],
        latency_ms: int,
        status: str = "success",
        error: str = "",
    ) -> bool:
        """Persist one chat runtime snapshot. Returns False on best-effort failure."""
        try:
            self._ensure_tables()
            self.pg.insert_runtime_log(
                build_runtime_log(
                    request_id=request_id,
                    query=query,
                    session_id=session_id,
                    user_id=user_id,
                    result=result,
                    latency_ms=latency_ms,
                    status=status,
                    error=error,
                )
            )
            return True
        except Exception as exc:
            logger.warning("[RuntimeLogger] failed to write runtime log: %s", exc)
            return False

    def _ensure_tables(self) -> None:
        if self._tables_ready:
            return
        self.pg.init_runtime_tables()
        self._tables_ready = True


def build_runtime_log(
    *,
    request_id: str,
    query: str,
    session_id: str,
    user_id: str,
    result: dict[str, Any],
    latency_ms: int,
    status: str = "success",
    error: str = "",
) -> dict[str, Any]:
    policy = result.get("personalization_policy", {}) or {}
    return {
        "request_id": request_id,
        "session_id": result.get("session_id", session_id),
        "user_id": result.get("user_id", user_id),
        "query": query,
        "intent": result.get("intent", ""),
        "intent_confidence": result.get("intent_confidence", 0.0),
        "personalization_mode": policy.get("mode", "none"),
        "private_content_required": bool(policy.get("private_content_required", False)),
        "private_content_found": bool(policy.get("private_content_found", False)),
        "planned_routes": result.get("planned_routes", []) or [],
        "executed_routes": result.get("executed_routes", []) or [],
        "fallback_routes": result.get("fallback_routes", []) or [],
        "route_status": result.get("route_status", {}) or {},
        "route_errors": result.get("route_errors", []) or [],
        "route_decision": result.get("route_decision", {}) or {},
        "trace": _compact_trace(result.get("trace", {}) or {}),
        "evidence_summary": summarize_evidence(result.get("reranked_evidence", []) or []),
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []) or [],
        "latency_ms": latency_ms,
        "status": status,
        "error": error,
    }


def summarize_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create privacy-conscious evidence summaries for runtime logs."""
    summary = []
    for item in evidence[:MAX_EVIDENCE_ITEMS]:
        content = str(item.get("content") or "")
        user_context = item.get("user_content_context")
        summary.append({
            "source_type": item.get("source_type", ""),
            "chunk_id": item.get("chunk_id"),
            "source_doc_id": item.get("source_doc_id") or _metadata_value(item, "source_doc_id"),
            "doc_category": item.get("doc_category", ""),
            "score": item.get("rerank_score", item.get("score")),
            "personalization_weight": item.get("personalization_weight"),
            "user_content_type": (
                user_context.get("content_type")
                if isinstance(user_context, dict)
                else _metadata_value(item, "user_content_type")
            ),
            "content_preview": content[:MAX_PREVIEW_CHARS] if content else _relation_preview(item),
        })
    return summary


def _metadata_value(item: dict[str, Any], key: str) -> Any:
    metadata = item.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    properties = item.get("properties")
    if isinstance(properties, dict):
        nested = properties.get("metadata")
        if isinstance(nested, dict):
            return nested.get(key)
    return None


def _relation_preview(item: dict[str, Any]) -> str:
    subject = str(item.get("subject") or "")
    predicate = str(item.get("predicate") or "")
    obj = str(item.get("object") or "")
    if subject or predicate or obj:
        return f"{subject} {predicate} {obj}".strip()[:MAX_PREVIEW_CHARS]
    return ""


def _compact_trace(trace: dict[str, Any]) -> dict[str, Any]:
    events = trace.get("events", [])
    if not isinstance(events, list):
        return trace
    return {**trace, "events": events[-20:]}
