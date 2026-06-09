"""AgentState definition for LangGraph shared state."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state passed between graph nodes."""

    # Input
    query: str
    session_id: str
    user_id: str
    session_history: list[dict[str, str]]

    # Planner
    intent: str
    intent_confidence: float
    intent_reason: str
    entities: list[dict[str, str]]
    has_profile_signal: bool
    planned_routes: list[str]
    original_planned_routes: list[str]
    executed_routes: list[str]
    fallback_routes: list[str]
    route_errors: list[dict[str, Any]]
    route_reason: dict[str, Any]
    route_status: dict[str, dict[str, Any]]
    trace: dict[str, Any]
    route_context: dict[str, Any]
    personalization_policy: dict[str, Any]
    semantic_discovered_entities: list[dict[str, str]]
    route_step_count: int
    route_decision: dict[str, Any]

    # Clarification
    clarification_needed: bool
    clarification_question: str
    clarification_answer: str

    # Retrieval evidence pools
    evidence: dict[str, list[dict[str, Any]]]

    # Rerank
    reranked_evidence: list[dict[str, Any]]
    safety_warnings: list[dict[str, Any]]

    # Answer
    answer: str
    citations: list[dict[str, Any]]

    # Runtime routing
    next_route: str


def create_initial_state(
    query: str,
    session_history: list[dict[str, str]] | None = None,
    session_id: str = "default",
    user_id: str = "default",
) -> AgentState:
    """Create the initial graph state."""
    return {
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
        "session_history": session_history or [],
        "intent": "",
        "intent_confidence": 0.0,
        "intent_reason": "",
        "entities": [],
        "has_profile_signal": False,
        "planned_routes": [],
        "original_planned_routes": [],
        "executed_routes": [],
        "fallback_routes": [],
        "route_errors": [],
        "route_reason": {},
        "route_status": {},
        "trace": {},
        "route_context": {},
        "personalization_policy": {"mode": "none", "private_content_required": False},
        "semantic_discovered_entities": [],
        "route_step_count": 0,
        "route_decision": {},
        "clarification_needed": False,
        "clarification_question": "",
        "clarification_answer": "",
        "evidence": {"semantic": [], "relation": [], "text2sql": []},
        "reranked_evidence": [],
        "safety_warnings": [],
        "answer": "",
        "citations": [],
        "next_route": "",
    }
