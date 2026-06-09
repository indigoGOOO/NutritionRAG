import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph_definition import _decide_after_planner
from src.agent.state import create_initial_state


def test_profile_management_without_routes_goes_to_confirm():
    state = create_initial_state("帮我记住，我花生过敏")
    state["intent"] = "profile_management"
    state["has_profile_signal"] = True
    state["planned_routes"] = []

    assert _decide_after_planner(state) == "confirm"


def test_profile_signal_with_retrieval_routes_dispatches_normally():
    state = create_initial_state("我花生过敏，可以吃这个吗？")
    state["intent"] = "safety_check"
    state["has_profile_signal"] = True
    state["planned_routes"] = ["semantic", "relation"]

    assert _decide_after_planner(state) == "dispatch"


def test_clarification_still_has_priority():
    state = create_initial_state("这个怎么样？")
    state["clarification_needed"] = True
    state["intent"] = "profile_management"
    state["has_profile_signal"] = True

    assert _decide_after_planner(state) == "clarify"
