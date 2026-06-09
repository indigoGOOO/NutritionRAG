import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph_definition import _router_node
from src.agent.state import create_initial_state


def test_router_adds_semantic_when_relation_misses_entities():
    state = create_initial_state("tomato and egg")
    state["next_route"] = "relation"
    state["executed_routes"] = ["relation"]
    state["planned_routes"] = []
    state["entities"] = [
        {"name": "tomato", "type": "ingredient"},
        {"name": "egg", "type": "ingredient"},
    ]
    state["evidence"]["relation"] = [
        {"subject": "tomato", "predicate": "contains", "object": "lycopene"}
    ]
    state["route_status"]["relation"] = {"status": "success", "count": 1, "error": ""}

    result = _router_node(state)

    assert result["next_route"] == "semantic"
    assert result["planned_routes"] == []
    assert result["executed_routes"] == ["relation", "semantic"]
    assert result["route_errors"] == [
        {
            "type": "dynamic_route_added",
            "route": "semantic",
            "reason": "relation_entity_coverage_low",
            "coverage": 0.5,
            "missing_entities": ["egg"],
            "semantic_query": "tomato and egg nutrition diet food pairing",
        }
    ]
    assert result["route_context"]["semantic"]["query"] == (
        "tomato and egg nutrition diet food pairing"
    )
    assert result["route_context"]["semantic"]["expansion_terms"] == ["egg"]


def test_router_converges_early_when_semantic_evidence_is_sufficient():
    state = create_initial_state("high quality semantic")
    state["intent"] = "recipe_instruction"
    state["next_route"] = "semantic"
    state["executed_routes"] = ["semantic"]
    state["planned_routes"] = ["relation", "text2sql"]
    state["evidence"]["semantic"] = [
        {"content": f"context {idx}", "score": 0.9} for idx in range(15)
    ]
    state["route_status"]["semantic"] = {"status": "success", "count": 15, "error": ""}

    result = _router_node(state)

    assert result["next_route"] == "rerank"
    assert result["planned_routes"] == []
    assert result["route_decision"]["reason"] == "semantic_evidence_sufficient"
    assert result["route_errors"] == [
        {
            "type": "early_converged",
            "reason": "semantic_evidence_sufficient",
            "skipped_routes": ["relation", "text2sql"],
        }
    ]


def test_router_does_not_converge_early_for_high_risk_intent():
    state = create_initial_state("hypertension and pickles")
    state["intent"] = "disease_diet"
    state["next_route"] = "semantic"
    state["executed_routes"] = ["semantic"]
    state["planned_routes"] = ["relation"]
    state["evidence"]["semantic"] = [
        {"content": f"context {idx}", "score": 0.9} for idx in range(15)
    ]
    state["route_status"]["semantic"] = {"status": "success", "count": 15, "error": ""}

    result = _router_node(state)

    assert result["next_route"] == "relation"
    assert result["planned_routes"] == []


def test_router_stops_when_max_steps_exceeded():
    state = create_initial_state("loop")
    state["route_step_count"] = 6
    state["planned_routes"] = ["semantic"]
    state["evidence"]["semantic"] = [{"content": "some evidence"}]

    result = _router_node(state)

    assert result["next_route"] == "rerank"
    assert result["planned_routes"] == []
    assert result["route_decision"]["reason"] == "max_router_steps_exceeded"


def test_router_retries_semantic_with_query_expansion_after_relation_gap():
    state = create_initial_state("tomato nutrition")
    state["next_route"] = "relation"
    state["executed_routes"] = ["semantic", "relation"]
    state["planned_routes"] = []
    state["entities"] = [
        {"name": "tomato", "type": "ingredient"},
        {"name": "egg", "type": "ingredient"},
    ]
    state["evidence"]["relation"] = [
        {"subject": "tomato", "predicate": "contains", "object": "lycopene"}
    ]
    state["route_status"]["relation"] = {"status": "success", "count": 1, "error": ""}

    result = _router_node(state)

    assert result["next_route"] == "semantic"
    assert result["executed_routes"] == ["semantic", "relation", "semantic"]
    assert result["route_context"]["semantic"]["retry_count"] == 1
    assert result["route_context"]["semantic"]["query"] == (
        "tomato nutrition egg diet food pairing"
    )
    assert result["route_errors"] == [
        {
            "type": "dynamic_route_added",
            "route": "semantic",
            "reason": "relation_entity_coverage_low_with_query_expansion",
            "coverage": 0.5,
            "missing_entities": ["egg"],
            "semantic_query": "tomato nutrition egg diet food pairing",
        }
    ]


def test_router_writes_relation_context_for_semantic_discovered_entities():
    state = create_initial_state("tomato nutrition")
    state["next_route"] = "semantic"
    state["executed_routes"] = ["semantic"]
    state["planned_routes"] = []
    state["entities"] = [{"name": "tomato", "type": "ingredient"}]
    state["semantic_discovered_entities"] = [{"name": "egg", "type": "ingredient"}]
    state["evidence"]["semantic"] = [{"content": "tomato pairs with egg", "score": 0.7}]
    state["route_status"]["semantic"] = {"status": "success", "count": 1, "error": ""}

    result = _router_node(state)

    assert result["next_route"] == "relation"
    assert result["route_context"]["relation"]["entities"] == [
        {"name": "tomato", "type": "ingredient"},
        {"name": "egg", "type": "ingredient"},
    ]
    assert result["route_context"]["relation"]["reason"] == "semantic_discovered_entities"


def test_router_writes_text2sql_context_for_nutrition_filter():
    state = create_initial_state("protein higher than 20g and fat low")
    state["intent"] = "nutrition_filter"
    state["planned_routes"] = []

    result = _router_node(state)

    assert result["next_route"] == "text2sql"
    assert result["route_context"]["text2sql"]["query"] == "protein higher than 20g and fat low"
    assert result["route_context"]["text2sql"]["constraints"]["protein"] == "mentioned"
    assert result["route_context"]["text2sql"]["constraints"]["fat"] == "mentioned"


def test_router_fallbacks_text2sql_schema_mismatch_to_relation_first():
    state = create_initial_state("protein foods")
    state["next_route"] = "text2sql"
    state["executed_routes"] = ["text2sql"]
    state["planned_routes"] = []
    state["entities"] = [{"name": "egg", "type": "ingredient"}]
    state["route_status"]["text2sql"] = {
        "status": "error",
        "count": 0,
        "error": "column protein_value does not exist",
        "reason": "schema_mismatch",
        "retry_strategy": "",
    }

    result = _router_node(state)

    assert result["next_route"] == "relation"
    assert result["fallback_routes"] == ["relation"]
    assert result["route_errors"] == [
        {
            "type": "route_failed_or_empty",
            "route": "text2sql",
            "status": "error",
            "error": "column protein_value does not exist",
            "reason": "schema_mismatch",
            "retry_strategy": "",
            "fallback_added": "relation",
        }
    ]
