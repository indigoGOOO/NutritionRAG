"""Helpers for collecting evaluation rows from a running Agent."""

from __future__ import annotations

from typing import Any, Protocol

from src.evaluation.dataset import EvaluationExample


class AgentLike(Protocol):
    """Minimal protocol implemented by NutritionAgent."""

    def run(self, query: str, session_history: list[dict] | None = None) -> dict:
        ...


def run_agent_examples(
    agent: AgentLike,
    gold_examples: list[EvaluationExample],
) -> list[EvaluationExample]:
    """Run an agent for each gold example and return filled evaluation examples.

    The gold example keeps labels such as ground_truth, expected routes,
    expected context ids, user profile, and forbidden ingredients. Runtime fields
    are filled from the agent response.
    """
    results = []
    for example in gold_examples:
        response = agent.run(example.question)
        results.append(example_from_agent_response(example, response))
    return results


def example_from_agent_response(
    gold: EvaluationExample,
    response: dict[str, Any],
) -> EvaluationExample:
    citations = response.get("citations", []) or []
    contexts = []
    citation_ids = []
    for citation in citations:
        if isinstance(citation, dict):
            content = citation.get("content", "")
            if content:
                contexts.append(str(content))
            source_id = citation.get("source_id")
            if source_id is not None:
                citation_ids.append(int(source_id))

    # Capture new router and safety fields
    safety_warnings = response.get("safety_warnings", []) or []
    actual_safety_items = []
    for w in safety_warnings:
        if isinstance(w, dict) and w.get("matched_ingredients"):
            actual_safety_items.extend(w["matched_ingredients"])
        elif isinstance(w, str):
            actual_safety_items.append(w)

    discovered = response.get("semantic_discovered_entities", []) or []
    actual_dynamic_entities = []
    for e in discovered:
        if isinstance(e, dict) and e.get("name"):
            actual_dynamic_entities.append(e["name"])
        elif isinstance(e, str):
            actual_dynamic_entities.append(e)

    route_status = response.get("route_status", {}) or {}
    trace = response.get("trace", {}) or {}
    executed_routes = response.get("executed_routes", []) or []
    fallback_routes = response.get("fallback_routes", []) or []

    return EvaluationExample(
        question=gold.question,
        answer=str(response.get("answer", "")),
        contexts=contexts or gold.contexts,
        ground_truth=gold.ground_truth,
        intent=str(response.get("intent", gold.intent)),
        expected_intent=gold.expected_intent,
        expected_routes=gold.expected_routes,
        expected_route_order=gold.expected_route_order,
        actual_routes=_coerce_routes(response),
        expected_context_ids=gold.expected_context_ids,
        retrieved_context_ids=_coerce_context_ids(response) or gold.retrieved_context_ids,
        citations=citation_ids or gold.citations,
        user_profile=gold.user_profile,
        forbidden_ingredients=gold.forbidden_ingredients,
        memory_policies=gold.memory_policies,
        expected_safety_warnings=gold.expected_safety_warnings,
        actual_safety_warnings=actual_safety_items,
        expected_dynamic_entities=gold.expected_dynamic_entities,
        actual_dynamic_entities=actual_dynamic_entities,
        expected_dynamic_routes=gold.expected_dynamic_routes,
        actual_dynamic_routes=_detect_dynamic_routes(response, gold),
        executed_routes=executed_routes,
        fallback_routes=fallback_routes,
        route_status=route_status,
        trace=trace,
        metadata={
            **gold.metadata,
            "entities": response.get("entities", []),
        },
    )


def _coerce_routes(response: dict[str, Any]) -> list[str]:
    routes = response.get("planned_routes", response.get("routes", [])) or []
    return [str(route) for route in routes]


def _coerce_context_ids(response: dict[str, Any]) -> list[str]:
    ids = response.get("retrieved_context_ids", []) or []
    if ids:
        return [str(item) for item in ids]
    citations = response.get("citations", []) or []
    result = []
    for citation in citations:
        if isinstance(citation, dict):
            source = citation.get("source_id", citation.get("source", ""))
            if source != "":
                result.append(str(source))
    return result


def _detect_dynamic_routes(
    response: dict[str, Any],
    gold: EvaluationExample,
) -> list[str]:
    """Detect routes that were dynamically added by the router beyond planner originals."""
    original = set(gold.expected_routes or [])
    executed = set(response.get("executed_routes", []) or [])
    fallback = set(response.get("fallback_routes", []) or [])
    # Dynamic routes are those executed or fallbacked that were not in original plan
    dynamic = (executed | fallback) - original
    # Remove standard default routes
    dynamic -= {"rerank", "answer"}
    return sorted(dynamic)
