"""LangGraph definition for the nutrition RAG agent."""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.answer_node import answer_node
from src.agent.clarify_node import clarify_node
from src.agent.confirm_profile_node import confirm_profile_node
from src.agent.pg_relation_retrieval_node import pg_relation_retrieval_node
from src.agent.planner_node import planner_node
from src.agent.rerank_node import rerank_node
from src.agent.safety_filter_node import safety_filter_node
from src.agent.semantic_retrieval_node import semantic_retrieval_node
from src.agent.state import AgentState, create_initial_state
from src.agent.text2sql_node import text2sql_node
from src.indexing.llm_client import BaseLLMClient
from src.memory.memory_manager import MemoryManager
from src.storage.milvus_client import MilvusClient
from src.storage.pg_client import PostgreSQLClient

logger = logging.getLogger(__name__)

VALID_ROUTES = ("semantic", "relation", "text2sql")
MAX_ROUTER_STEPS = 6
STRONG_INTENT_ROUTES = {
    "disease_diet": ["semantic", "relation"],
    "safety_check": ["semantic", "relation"],
    "nutrition_filter": ["text2sql"],
    "recipe_instruction": ["semantic"],
}
HIGH_RISK_INTENTS = {"disease_diet", "safety_check"}
INTENT_SEMANTIC_TERMS = {
    "disease_diet": ["diet advice", "risk", "contraindication", "suitable", "avoid"],
    "safety_check": ["allergy", "contraindication", "risk", "safety"],
    "recipe_instruction": ["recipe", "pairing", "nutrition", "suitable pairing", "conflict"],
    "nutrition_filter": ["nutrition facts", "protein", "fat", "calories", "sodium", "sugar"],
}
INTENT_RELATION_PREDICATES = {
    "disease_diet": ["contraindicated_for", "suitable_for", "conflicts_with", "recommends"],
    "safety_check": ["contraindicated_for", "conflicts_with", "contains"],
    "recipe_instruction": ["pairs_with", "conflicts_with", "contains", "cooked_by"],
    "nutrition_filter": ["contains", "source_of", "belongs_to"],
}
STRUCTURED_QUERY_TERMS = {
    "大于", "小于", "低于", "高于", "超过", "少于", "排名", "最多", "最少",
    "protein", "fat", "calorie", "sodium", "sugar",
}
FALLBACK_ROUTES = {
    "text2sql": ["semantic"],
    "semantic": ["relation"],
    "relation": ["semantic"],
}


def build_graph(
    llm: BaseLLMClient,
    pg: PostgreSQLClient,
    milvus: MilvusClient,
    cross_encoder=None,
    memory_manager: MemoryManager | None = None,
) -> CompiledStateGraph:
    """Build the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", lambda s: planner_node(s, llm, memory_manager))
    workflow.add_node("clarify", lambda s: clarify_node(s, llm))
    workflow.add_node("router", _router_node)
    workflow.add_node("semantic_retrieval", lambda s: semantic_retrieval_node(s, milvus))
    workflow.add_node("relation_retrieval", lambda s: pg_relation_retrieval_node(s, pg))
    workflow.add_node("text2sql", lambda s: text2sql_node(s, llm, pg))
    workflow.add_node("rerank", lambda s: rerank_node(s, cross_encoder))
    workflow.add_node("safety_filter", lambda s: safety_filter_node(s, memory_manager))
    workflow.add_node("answer", lambda s: answer_node(s, llm, memory_manager))
    workflow.add_node("confirm_profile", lambda s: confirm_profile_node(s, llm, memory_manager))
    workflow.add_node("confirm_profile_after_answer", lambda s: confirm_profile_node(s, llm, memory_manager))

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        _decide_after_planner,
        {"clarify": "clarify", "confirm": "confirm_profile", "dispatch": "router"},
    )
    workflow.add_edge("clarify", "planner")
    workflow.add_edge("confirm_profile", "answer")

    workflow.add_conditional_edges(
        "router",
        _dispatch_route,
        {
            "semantic": "semantic_retrieval",
            "relation": "relation_retrieval",
            "text2sql": "text2sql",
            "clarify": "clarify",
            "rerank": "rerank",
            "answer": "answer",
        },
    )
    workflow.add_edge("semantic_retrieval", "router")
    workflow.add_edge("relation_retrieval", "router")
    workflow.add_edge("text2sql", "router")

    workflow.add_edge("rerank", "safety_filter")
    workflow.add_edge("safety_filter", "answer")

    workflow.add_conditional_edges(
        "answer",
        _decide_confirm_profile,
        {"confirm": "confirm_profile_after_answer", "end": END},
    )
    workflow.add_edge("confirm_profile_after_answer", END)

    return workflow.compile()


def _decide_after_planner(state: AgentState) -> Literal["clarify", "confirm", "dispatch"]:
    if state.get("clarification_needed"):
        return "clarify"
    if (
        state.get("intent") == "profile_management"
        and state.get("has_profile_signal")
        and not state.get("planned_routes", [])
    ):
        return "confirm"
    return "dispatch"


def _router_node(state: AgentState) -> dict:
    """Clean planned routes, dispatch the next route, and add conservative fallbacks."""
    route_step_count = int(state.get("route_step_count", 0)) + 1
    raw_routes = state.get("planned_routes", [])
    cleaned_routes, invalid_routes = _sanitize_routes(raw_routes)
    route_errors = list(state.get("route_errors", []))

    if invalid_routes:
        route_errors.append({
            "type": "invalid_route_filtered",
            "routes": invalid_routes,
        })
        logger.warning(f"[Router] filtered invalid routes: {invalid_routes}")

    cleaned_routes, intent_update = _enforce_intent_routes(state, cleaned_routes)
    route_errors.extend(intent_update.get("route_errors", []))

    cleaned_routes, dynamic_update = _append_relation_for_semantic_entities(state, cleaned_routes)
    route_errors.extend(dynamic_update.get("route_errors", []))

    cleaned_routes, coverage_update = _append_semantic_for_relation_entity_gaps(state, cleaned_routes)
    route_errors.extend(coverage_update.get("route_errors", []))

    cleaned_routes, fallback_update = _append_fallback_if_last_route_failed(state, cleaned_routes)
    route_errors.extend(fallback_update.get("route_errors", []))
    route_context = _merge_route_context_updates(
        state,
        intent_update,
        dynamic_update,
        coverage_update,
        fallback_update,
    )

    route_decision = _judge_route_decision(state, cleaned_routes, route_step_count)
    route_errors.extend(route_decision.get("route_errors", []))
    if route_decision.get("force_stop"):
        cleaned_routes = []
    elif route_decision.get("clarify"):
        trace = _append_trace_event(state, "router_clarify", {
            "reason": route_decision.get("reason", ""),
        })
        update = {
            "next_route": "clarify",
            "planned_routes": cleaned_routes,
            "route_errors": route_errors,
            "trace": trace,
            "route_context": route_context,
            "route_step_count": route_step_count,
            "route_decision": route_decision,
            "clarification_needed": True,
            "clarification_question": route_decision.get(
                "question",
                "我需要再确认一下你的具体情况，才能更稳妥地回答。",
            ),
        }
        if intent_update.get("planned_routes") is not None:
            update["planned_routes"] = cleaned_routes
        return update

    if cleaned_routes:
        next_route = cleaned_routes[0]
        remaining = cleaned_routes[1:]
        executed = [*state.get("executed_routes", []), next_route]
        trace = _append_trace_event(state, "router_dispatch", {
            "next_route": next_route,
            "remaining_routes": remaining,
            "route_decision": route_decision,
        })
        update = {
            "next_route": next_route,
            "planned_routes": remaining,
            "executed_routes": executed,
            "route_errors": route_errors,
            "trace": trace,
            "route_context": route_context,
            "route_step_count": route_step_count,
            "route_decision": route_decision,
        }
        if _is_first_router_pass(state):
            update["original_planned_routes"] = [*executed, *remaining]
        if fallback_update.get("fallback_routes") is not None:
            update["fallback_routes"] = fallback_update["fallback_routes"]
        if dynamic_update.get("entities") is not None:
            update["entities"] = dynamic_update["entities"]
        if coverage_update.get("entities") is not None:
            update["entities"] = coverage_update["entities"]
        if dynamic_update.get("planned_routes") is not None:
            update["planned_routes"] = update["planned_routes"]
        return update

    next_route = "answer" if _should_skip_rerank(state) else "rerank"
    update = {
        "next_route": next_route,
        "planned_routes": [],
        "route_errors": route_errors,
        "trace": _append_trace_event(state, "router_done", {
            "next_route": next_route,
            "route_decision": route_decision,
        }),
        "route_context": route_context,
        "route_step_count": route_step_count,
        "route_decision": route_decision,
    }
    if _is_first_router_pass(state):
        update["original_planned_routes"] = []
    if fallback_update.get("fallback_routes") is not None:
        update["fallback_routes"] = fallback_update["fallback_routes"]
    if dynamic_update.get("entities") is not None:
        update["entities"] = dynamic_update["entities"]
    if coverage_update.get("entities") is not None:
        update["entities"] = coverage_update["entities"]
    return update


def _dispatch_route(state: AgentState) -> str:
    return state.get("next_route", "rerank")


def _sanitize_routes(routes: list[str]) -> tuple[list[str], list[str]]:
    cleaned = []
    invalid = []
    for route in routes or []:
        route = str(route)
        if route not in VALID_ROUTES:
            invalid.append(route)
            continue
        if route not in cleaned:
            cleaned.append(route)
    return cleaned, invalid


def _append_fallback_if_last_route_failed(
    state: AgentState,
    planned_routes: list[str],
) -> tuple[list[str], dict]:
    last_route = state.get("next_route", "")
    if last_route not in VALID_ROUTES:
        return planned_routes, {}
    route_status = state.get("route_status", {}).get(last_route, {})
    if route_status.get("status") == "success" or _route_has_evidence(state, last_route):
        return planned_routes, {}

    executed = state.get("executed_routes", [])
    if not executed or executed[-1] != last_route:
        return planned_routes, {}

    fallback_routes = list(state.get("fallback_routes", []))
    route_errors = [{
        "type": "route_failed_or_empty",
        "route": last_route,
        "status": route_status.get("status", "empty"),
        "error": route_status.get("error", ""),
        "reason": route_status.get("reason", ""),
        "retry_strategy": route_status.get("retry_strategy", ""),
        "fallback_added": None,
    }]
    blocked = set(executed) | set(planned_routes) | set(fallback_routes)

    for fallback in _fallback_candidates_for_route(last_route, route_status):
        if fallback in blocked:
            continue
        logger.info(f"[Router] route={last_route} returned no evidence, fallback to {fallback}")
        fallback_routes.append(fallback)
        route_errors[-1]["fallback_added"] = fallback
        return [fallback, *planned_routes], {
            "fallback_routes": fallback_routes,
            "route_errors": route_errors,
            "route_context": _build_fallback_route_context(state, fallback, last_route),
        }

    logger.info(f"[Router] route={last_route} returned no evidence, no fallback left")
    return planned_routes, {
        "fallback_routes": fallback_routes,
        "route_errors": route_errors,
    }


def _fallback_candidates_for_route(route: str, route_status: dict) -> list[str]:
    if route == "text2sql":
        reason = route_status.get("reason", "")
        if reason == "schema_mismatch":
            return ["relation", "semantic"]
        return ["semantic"]
    return FALLBACK_ROUTES.get(route, [])


def _merge_route_context_updates(state: AgentState, *updates: dict) -> dict:
    route_context = {**state.get("route_context", {})}
    for update in updates:
        for route, context in update.get("route_context", {}).items():
            existing = route_context.get(route, {})
            if isinstance(existing, dict) and isinstance(context, dict):
                route_context[route] = {**existing, **context}
            else:
                route_context[route] = context
    return route_context


def _build_intent_route_context(state: AgentState, routes: list[str], intent: str) -> dict:
    context = {}
    entities = state.get("entities", [])
    if "semantic" in routes:
        terms = INTENT_SEMANTIC_TERMS.get(intent, ["nutrition", "diet", "food"])
        context["semantic"] = _build_semantic_context(
            state,
            _compose_query(state.get("query", ""), _entity_names(entities), terms),
            terms,
            f"intent_{intent}",
        )["semantic"]
    if "relation" in routes:
        context["relation"] = _build_relation_context(
            state,
            entities,
            f"intent_{intent}",
        )["relation"]
    if "text2sql" in routes:
        context["text2sql"] = _build_text2sql_context(state, f"intent_{intent}")["text2sql"]
    return context


def _build_fallback_route_context(state: AgentState, fallback: str, failed_route: str) -> dict:
    reason = f"{failed_route}_fallback"
    if fallback == "semantic":
        names = _entity_names(state.get("entities", []))
        return _build_semantic_context(
            state,
            _compose_query(state.get("query", ""), names, [failed_route, "fallback"]),
            names,
            reason,
        )
    if fallback == "relation":
        return _build_relation_context(state, state.get("entities", []), reason)
    if fallback == "text2sql":
        return _build_text2sql_context(state, reason)
    return {}


def _build_semantic_context(
    state: AgentState,
    query: str,
    expansion_terms: list[str],
    reason: str,
) -> dict:
    current = state.get("route_context", {}).get("semantic", {})
    retry_count = int(current.get("retry_count", 0))
    if reason.endswith("_retry"):
        retry_count += 1
    return {
        "semantic": {
            "query": query,
            "expansion_terms": _dedupe_strings(expansion_terms),
            "reason": reason,
            "retry_count": retry_count,
        }
    }


def _build_relation_context(state: AgentState, entities: list[dict], reason: str) -> dict:
    intent = state.get("intent", "")
    predicates = INTENT_RELATION_PREDICATES.get(intent, ["related_to", "contains"])
    return {
        "relation": {
            "entities": entities,
            "predicates": predicates,
            "reason": reason,
        }
    }


def _build_text2sql_context(state: AgentState, reason: str) -> dict:
    query = state.get("query", "")
    constraints = _infer_text2sql_constraints(query)
    return {
        "text2sql": {
            "query": query,
            "constraints": constraints,
            "reason": reason,
        }
    }


def _can_retry_semantic_with_expansion(state: AgentState, missing_entities: list[str]) -> bool:
    if not missing_entities:
        return False
    current = state.get("route_context", {}).get("semantic", {})
    retry_count = int(current.get("retry_count", 0))
    return retry_count < 1


def _build_expanded_semantic_query(state: AgentState, missing_entities: list[str]) -> str:
    intent = state.get("intent", "")
    terms = INTENT_SEMANTIC_TERMS.get(intent, ["nutrition", "diet", "food", "pairing"])
    names = _entity_names(state.get("entities", []))
    return _compose_query(state.get("query", ""), names, missing_entities, terms)


def _compose_query(*parts: object) -> str:
    tokens: list[str] = []
    for part in parts:
        if isinstance(part, str):
            tokens.extend(part.split())
        elif isinstance(part, list):
            tokens.extend(str(item) for item in part if str(item).strip())
    return " ".join(_dedupe_strings(tokens))


def _entity_names(entities: list[dict]) -> list[str]:
    names = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name", "")).strip()
        if name:
            names.append(name)
    return _dedupe_strings(names)


def _dedupe_strings(items: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def _infer_text2sql_constraints(query: str) -> dict:
    constraints = {}
    nutrient_aliases = {
        "protein": ("protein", "蛋白质"),
        "fat": ("fat", "脂肪"),
        "calorie": ("calorie", "calories", "热量", "能量"),
        "sodium": ("sodium", "钠"),
        "sugar": ("sugar", "糖"),
        "carbohydrate": ("carbohydrate", "carb", "碳水"),
    }
    lowered = query.lower()
    for field, aliases in nutrient_aliases.items():
        if any(alias in lowered or alias in query for alias in aliases):
            constraints[field] = "mentioned"
    if any(term in query or term in lowered for term in STRUCTURED_QUERY_TERMS):
        constraints["structured_query"] = True
    return constraints


def _enforce_intent_routes(
    state: AgentState,
    planned_routes: list[str],
) -> tuple[list[str], dict]:
    intent = state.get("intent", "")
    required_routes = STRONG_INTENT_ROUTES.get(intent, [])
    if not required_routes:
        return planned_routes, {}

    blocked = set(state.get("executed_routes", [])) | set(planned_routes)
    missing = [route for route in required_routes if route not in blocked]
    if not missing:
        return planned_routes, {}

    route_errors = [{
        "type": "intent_route_enforced",
        "intent": intent,
        "routes": missing,
    }]
    route_context = _build_intent_route_context(state, missing, intent)
    return [*missing, *planned_routes], {
        "planned_routes": [*missing, *planned_routes],
        "route_errors": route_errors,
        "route_context": route_context,
    }


def _append_relation_for_semantic_entities(
    state: AgentState,
    planned_routes: list[str],
) -> tuple[list[str], dict]:
    last_route = state.get("next_route", "")
    if last_route != "semantic":
        return planned_routes, {}
    discovered = state.get("semantic_discovered_entities", [])
    if not discovered:
        return planned_routes, {}

    entities = _merge_entities(state.get("entities", []), discovered)
    blocked = set(state.get("executed_routes", [])) | set(planned_routes)
    if "relation" in blocked:
        return planned_routes, {
            "entities": entities,
            "route_context": _build_relation_context(state, entities, "semantic_discovered_entities"),
        }

    logger.info("[Router] semantic discovered new entities, scheduling relation")
    return ["relation", *planned_routes], {
        "entities": entities,
        "route_context": _build_relation_context(state, entities, "semantic_discovered_entities"),
        "route_errors": [{
            "type": "dynamic_route_added",
            "route": "relation",
            "reason": "semantic_discovered_entities",
            "entities": [e.get("name", "") for e in discovered],
        }],
    }


def _append_semantic_for_relation_entity_gaps(
    state: AgentState,
    planned_routes: list[str],
) -> tuple[list[str], dict]:
    last_route = state.get("next_route", "")
    if last_route != "relation":
        return planned_routes, {}

    entities = state.get("entities", [])
    entity_names = [str(e.get("name", "")).strip() for e in entities if isinstance(e, dict)]
    entity_names = [name for name in entity_names if name]
    if not entity_names:
        return planned_routes, {}

    relation_items = state.get("evidence", {}).get("relation", [])
    covered = _covered_entities_by_relation(entity_names, relation_items)
    coverage = len(covered) / max(len(set(entity_names)), 1)
    if coverage >= 0.6:
        return planned_routes, {}

    blocked = set(state.get("executed_routes", [])) | set(planned_routes)
    if "semantic" in blocked:
        if not _can_retry_semantic_with_expansion(state, missing := [
            name for name in entity_names if name not in covered
        ]):
            return planned_routes, {}
        expanded_query = _build_expanded_semantic_query(state, missing)
        logger.info("[Router] relation missed entities, retrying semantic with expansion")
        return ["semantic", *planned_routes], {
            "route_context": _build_semantic_context(
                state,
                expanded_query,
                missing,
                "relation_entity_coverage_low_retry",
            ),
            "route_errors": [{
                "type": "dynamic_route_added",
                "route": "semantic",
                "reason": "relation_entity_coverage_low_with_query_expansion",
                "coverage": coverage,
                "missing_entities": missing,
                "semantic_query": expanded_query,
            }],
        }

    missing = [name for name in entity_names if name not in covered]
    expanded_query = _build_expanded_semantic_query(state, missing)
    logger.info("[Router] relation missed key entities, scheduling semantic")
    return ["semantic", *planned_routes], {
        "route_context": _build_semantic_context(
            state,
            expanded_query,
            missing,
            "relation_entity_coverage_low",
        ),
        "route_errors": [{
            "type": "dynamic_route_added",
            "route": "semantic",
            "reason": "relation_entity_coverage_low",
            "coverage": coverage,
            "missing_entities": missing,
            "semantic_query": expanded_query,
        }],
    }


def _covered_entities_by_relation(entity_names: list[str], relation_items: list[dict]) -> set[str]:
    covered = set()
    for item in relation_items:
        text = " ".join(
            str(item.get(key, ""))
            for key in ("subject", "predicate", "object", "entity", "attribute", "value", "content")
        )
        for name in entity_names:
            if name and name in text:
                covered.add(name)
    return covered


def _judge_route_decision(
    state: AgentState,
    planned_routes: list[str],
    route_step_count: int,
) -> dict:
    if route_step_count > MAX_ROUTER_STEPS:
        return {
            "can_answer": _has_any_evidence(state),
            "force_stop": True,
            "reason": "max_router_steps_exceeded",
            "route_errors": [{
                "type": "max_router_steps_exceeded",
                "max_steps": MAX_ROUTER_STEPS,
                "executed_routes": state.get("executed_routes", []),
            }],
        }

    intent = state.get("intent", "")
    if _semantic_evidence_is_sufficient(state) and _can_converge_early(state, planned_routes):
        return {
            "can_answer": True,
            "force_stop": True,
            "reason": "semantic_evidence_sufficient",
            "route_errors": [{
                "type": "early_converged",
                "reason": "semantic_evidence_sufficient",
                "skipped_routes": planned_routes,
            }],
        }

    if not planned_routes and not _has_any_evidence(state):
        if intent in HIGH_RISK_INTENTS:
            return {
                "can_answer": False,
                "clarify": True,
                "reason": "high_risk_no_evidence",
                "question": "这个问题涉及健康风险，我需要更多背景或明确的食材/疾病信息后再回答。",
            }
        return {
            "can_answer": True,
            "degraded": True,
            "reason": "no_evidence_degraded_answer",
        }

    if not planned_routes and _evidence_quality_low(state):
        return {
            "can_answer": True,
            "degraded": True,
            "reason": "low_quality_evidence_degraded_answer",
        }

    return {
        "can_answer": False,
        "reason": "continue_routing" if planned_routes else "ready_for_rerank",
    }


def _semantic_evidence_is_sufficient(state: AgentState) -> bool:
    semantic_items = state.get("evidence", {}).get("semantic", [])
    if len(semantic_items) >= 15:
        return True
    if len(semantic_items) < 8:
        return False

    scores = [_extract_score(item) for item in semantic_items[:8]]
    scores = [score for score in scores if score is not None]
    if not scores:
        return len(semantic_items) >= 12
    return sum(scores) / len(scores) >= 0.75


def _can_converge_early(state: AgentState, planned_routes: list[str]) -> bool:
    if not planned_routes:
        return False
    if state.get("intent") in HIGH_RISK_INTENTS:
        return False
    return "semantic" in state.get("executed_routes", [])


def _evidence_quality_low(state: AgentState) -> bool:
    items = []
    for route_items in state.get("evidence", {}).values():
        items.extend(route_items or [])
    if len(items) < 2:
        return True

    scores = [_extract_score(item) for item in items]
    scores = [score for score in scores if score is not None]
    if not scores:
        return False
    return max(scores) < 0.45


def _extract_score(item: dict) -> float | None:
    for key in ("rerank_score", "score", "similarity", "confidence"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    distance = item.get("distance")
    if isinstance(distance, (int, float)):
        return max(0.0, 1.0 - float(distance))
    return None


def _merge_entities(existing: list[dict], discovered: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    for item in [*existing, *discovered]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        entity_type = str(item.get("type", "unknown") or "unknown").strip()
        if not name or name in seen:
            continue
        merged.append({"name": name, "type": entity_type})
        seen.add(name)
    return merged


def _route_has_evidence(state: AgentState, route: str) -> bool:
    evidence = state.get("evidence", {})
    return bool(evidence.get(route, []))


def _should_skip_rerank(state: AgentState) -> bool:
    return (
        not _has_any_evidence(state)
        and state.get("intent") in {"profile_management", "content_save"}
    )


def _has_any_evidence(state: AgentState) -> bool:
    evidence = state.get("evidence", {})
    return any(bool(items) for items in evidence.values())


def _is_first_router_pass(state: AgentState) -> bool:
    return not state.get("executed_routes") and not state.get("next_route")


def _append_trace_event(state: AgentState, event_type: str, payload: dict) -> dict:
    trace = {**state.get("trace", {})}
    events = list(trace.get("events", []))
    events.append({"type": event_type, **payload})
    trace["events"] = events
    return trace


def _decide_confirm_profile(state: AgentState) -> Literal["confirm", "end"]:
    return "confirm" if state.get("has_profile_signal") else "end"


class NutritionAgent:
    """Convenience wrapper around the compiled graph."""

    def __init__(
        self,
        pg_client: PostgreSQLClient,
        milvus_client: MilvusClient,
        llm_client: BaseLLMClient,
        cross_encoder=None,
        memory_manager: MemoryManager | None = None,
    ):
        self.pg = pg_client
        self.milvus = milvus_client
        self.llm = llm_client
        self.graph = build_graph(
            llm=llm_client,
            pg=pg_client,
            milvus=milvus_client,
            cross_encoder=cross_encoder,
            memory_manager=memory_manager,
        )
        self.memory = memory_manager

    def run(
        self,
        query: str,
        session_history: list[dict] | None = None,
        session_id: str = "default",
        user_id: str = "default",
    ) -> dict:
        """Run the agent synchronously."""
        if self.memory:
            self.memory.user_id = user_id
            self.memory.on_user_query(session_id, query)

        initial_state = create_initial_state(
            query=query,
            session_history=session_history,
            session_id=session_id,
            user_id=user_id,
        )
        final_state = self.graph.invoke(initial_state)

        answer = final_state.get("answer", "")
        evidence = final_state.get("reranked_evidence", [])
        result = {
            "answer": answer,
            "citations": final_state.get("citations", []),
            "entities": final_state.get("entities", []),
            "intent": final_state.get("intent", ""),
            "intent_confidence": final_state.get("intent_confidence", 0.0),
            "intent_reason": final_state.get("intent_reason", ""),
            "session_id": session_id,
            "user_id": user_id,
            "planned_routes": final_state.get("original_planned_routes", []),
            "executed_routes": final_state.get("executed_routes", []),
            "fallback_routes": final_state.get("fallback_routes", []),
            "route_errors": final_state.get("route_errors", []),
            "route_status": final_state.get("route_status", {}),
            "trace": final_state.get("trace", {}),
            "route_context": final_state.get("route_context", {}),
            "route_decision": final_state.get("route_decision", {}),
            "personalization_policy": final_state.get("personalization_policy", {}),
            "semantic_discovered_entities": final_state.get("semantic_discovered_entities", []),
            "safety_warnings": final_state.get("safety_warnings", []),
            "route_reason": final_state.get("route_reason", {}),
            "has_profile_signal": final_state.get("has_profile_signal", False),
            "evidence_count": len(evidence),
            "reranked_evidence": evidence,
            "retrieved_context_ids": _extract_context_ids(evidence),
            "contexts": _extract_contexts(evidence),
        }

        if self.memory and answer:
            self.memory.user_id = user_id
            self.memory.on_assistant_answer(
                session_id,
                answer,
                {
                    "intent": result["intent"],
                    "entities": [e.get("name", "") for e in result["entities"]],
                    "planned_routes": result["planned_routes"],
                    "executed_routes": result["executed_routes"],
                    "fallback_routes": result["fallback_routes"],
                    "route_errors": result["route_errors"],
                    "route_status": result["route_status"],
                    "route_context": result["route_context"],
                    "route_decision": result["route_decision"],
                    "personalization_policy": result["personalization_policy"],
                    "semantic_discovered_entities": [
                        e.get("name", "") for e in result["semantic_discovered_entities"]
                    ],
                    "safety_warnings": result["safety_warnings"],
                },
            )

        return result

    def run_stream(
        self,
        query: str,
        session_id: str = "default",
        user_id: str = "default",
    ):
        """Stream graph node events."""
        if self.memory:
            self.memory.user_id = user_id
            self.memory.on_user_query(session_id, query)

        initial_state = create_initial_state(query, session_id=session_id, user_id=user_id)
        for event in self.graph.stream(initial_state):
            node_name = list(event.keys())[0]
            yield {"node": node_name, "state": event[node_name]}


def _extract_context_ids(evidence: list[dict]) -> list[str]:
    ids = []
    for idx, item in enumerate(evidence, 1):
        value = (
            item.get("evidence_id")
            or item.get("chunk_id")
            or item.get("source_id")
            or item.get("id")
            or idx
        )
        ids.append(str(value))
    return ids


def _extract_contexts(evidence: list[dict]) -> list[str]:
    contexts = []
    for item in evidence:
        content = item.get("content")
        if content:
            contexts.append(str(content))
            continue

        subject = item.get("subject", "")
        predicate = item.get("predicate", "")
        obj = item.get("object", "")
        if subject or predicate or obj:
            contexts.append(f"{subject} {predicate} {obj}".strip())
            continue

        entity = item.get("entity", "")
        attribute = item.get("attribute", "")
        value = item.get("value", "")
        if entity or attribute or value:
            contexts.append(f"{entity} {attribute} {value}".strip())
    return contexts
