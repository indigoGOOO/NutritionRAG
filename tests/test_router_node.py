import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph_definition import _router_node
from src.agent.state import create_initial_state


def test_router_deduplicates_and_filters_invalid_routes():
    state = create_initial_state("test")
    state["planned_routes"] = ["semantic", "bad", "semantic", "text2sql", "unknown"]

    result = _router_node(state)

    assert result["next_route"] == "semantic"
    assert result["planned_routes"] == ["text2sql"]
    assert result["executed_routes"] == ["semantic"]
    assert result["original_planned_routes"] == ["semantic", "text2sql"]
    assert result["route_errors"] == [
        {"type": "invalid_route_filtered", "routes": ["bad", "unknown"]}
    ]


def test_router_sends_profile_management_with_no_routes_to_answer():
    state = create_initial_state("帮我记住，我花生过敏")
    state["intent"] = "profile_management"
    state["has_profile_signal"] = True
    state["planned_routes"] = []

    result = _router_node(state)

    assert result["next_route"] == "answer"
    assert result["planned_routes"] == []
    assert result["original_planned_routes"] == []


def test_router_adds_fallback_when_text2sql_returns_empty_evidence():
    state = create_initial_state("蛋白质大于10g的食物有哪些？")
    state["next_route"] = "text2sql"
    state["executed_routes"] = ["text2sql"]
    state["planned_routes"] = []
    state["evidence"]["text2sql"] = []
    state["route_status"]["text2sql"] = {"status": "empty", "count": 0, "error": ""}

    result = _router_node(state)

    assert result["next_route"] == "semantic"
    assert result["planned_routes"] == []
    assert result["executed_routes"] == ["text2sql", "semantic"]
    assert result["fallback_routes"] == ["semantic"]
    assert result["route_errors"] == [
        {
            "type": "route_failed_or_empty",
            "route": "text2sql",
            "status": "empty",
            "error": "",
            "reason": "",
            "retry_strategy": "",
            "fallback_added": "semantic",
        }
    ]


def test_router_does_not_repeat_executed_fallback():
    state = create_initial_state("test")
    state["next_route"] = "semantic"
    state["executed_routes"] = ["semantic"]
    state["fallback_routes"] = ["relation"]
    state["planned_routes"] = []
    state["evidence"]["semantic"] = []
    state["route_status"]["semantic"] = {"status": "empty", "count": 0, "error": ""}

    result = _router_node(state)

    assert result["next_route"] == "rerank"
    assert result["fallback_routes"] == ["relation"]
    assert result["route_errors"] == [
        {
            "type": "route_failed_or_empty",
            "route": "semantic",
            "status": "empty",
            "error": "",
            "reason": "",
            "retry_strategy": "",
            "fallback_added": None,
        }
    ]


def test_router_adds_relation_when_semantic_discovers_new_entities():
    state = create_initial_state("番茄有什么营养？")
    state["next_route"] = "semantic"
    state["executed_routes"] = ["semantic"]
    state["planned_routes"] = []
    state["evidence"]["semantic"] = [{"content": "番茄相宜鸡蛋。"}]
    state["route_status"]["semantic"] = {"status": "success", "count": 1, "error": ""}
    state["entities"] = [{"name": "番茄", "type": "ingredient"}]
    state["semantic_discovered_entities"] = [{"name": "鸡蛋", "type": "ingredient"}]

    result = _router_node(state)

    assert result["next_route"] == "relation"
    assert result["executed_routes"] == ["semantic", "relation"]
    assert result["entities"] == [
        {"name": "番茄", "type": "ingredient"},
        {"name": "鸡蛋", "type": "ingredient"},
    ]
    assert result["route_errors"] == [
        {
            "type": "dynamic_route_added",
            "route": "relation",
            "reason": "semantic_discovered_entities",
            "entities": ["鸡蛋"],
        }
    ]
